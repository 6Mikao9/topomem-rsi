"""Bind the existing recursive controller to fixed real SGLang DP2 cases."""

import atexit
from hashlib import sha256
import math
from pathlib import Path

from integrations.topology_memory import sglang_ingress_gate as ingress
from rsi_memory import execution_client, execution_worker, memory_program, topology_rsi
from rsi_memory.execution_client import ExecutionError, PersistentExecutor
from rsi_memory.execution_worker import atomic, digest
from rsi_memory.memory_program import MemoryProgram, generator_context
from rsi_memory.topology import fingerprint
from rsi_memory.topology_rsi import Domain

from . import contract, hot_config
from .contract import EVALUATOR, FEATURES, NCCL_SHA256, SCOPE, checked_case, checked_id, checked_inputs, checked_sha, load_case, load_external, read_json
from .hot_config import ACTIONS, BACKING_FEATURES, HOT_FEATURES, PARITY_KEYS, compile_memory, resident_control_program, stock_control_program


RULE_SEMANTICS = """An optional configure/hot_capacity rule runs once at startup and picks a common measured native hot size.
With no configure rule, 4096 slots is the default. All eligible hot choices are real measured configurations. Lower score wins in
the first eligible rule. Incremental observed coordinator metadata consumes the same device budget as cold slabs and relay scratch.
An optional configure/backing_mode rule chooses tiered (candidate_backing_is_resident=0) or full resident (=1).
Resident mode needs the qualified 6144-slot hot setting, omits all allocate/place rules and requires the full committed
prompt plus maximum generated length of every fixed request to fit the native page-reserved hot pool. It allocates no
cold slabs and uses a separately budgeted CUDA sentinel for each layer. The evaluator checks producer bytes and actual
attention-selected addresses. A resident choice without a valid case proof is infeasible.
Tiered mode is the default. In tiered mode allocate/place visits slabs in fixed external rank and layer order under the remaining budget.
A slab holds every packed 656-byte cold KV row for one layer on one worker, with one authoritative storage tier.
Each enabled destination has a fixed read/write route or a zero-hop local copy. Lower score wins in the first eligible rule.
The compiler checks all slab caps and real relay scratch. Zero-hop routes report zero estimated bandwidth and transfer time.
Measured route estimates use the whole slab and are topology features. They do not equal observed selected-row serving time.
Native HiSparse retains request slots, hot-cache replacement and DSA attention. Resident only skips proved unnecessary cold backups.
Rank placement, scheduler, arrivals, model,
tokenization, memory caps and the measured window are fixed. This replay has independent requests; physical prefix retention,
copy-on-write, KV migration over time, prefetch and cache replacement are outside this backend's capabilities.
Parity checks real selected packed bytes, output IDs, actual dispatch and every active global DP forward before any scoring.
Score is sum of measured replay window milliseconds across fixed cases, including complete model prefill and decode after warmup.
The default native baseline uses 4096 slots and all-host backing. The strong stock control uses 6144 slots and all-host backing.
The program may choose an equivalent simple native memory configuration. Training miss counts are diagnostic context."""


class SGLangDomain:
    def __init__(self, config, directory, *, executor_factory=PersistentExecutor):
        fields = {"schema", "experiment_id", "source_manifest_sha256", "launch_sha256", "executor", "receipt_directory",
                  "local_cases", "cases", "token_inputs", "paired_block_ids", "initial_program", "hot_profile", "deadline", "job_timeout"}
        schema = config.get("schema") if type(config) is dict else None
        if schema == 3:
            fields.update({"manual_program", "manual_selection", "controller_preregistration"})
        if type(schema) is not int or schema not in (2, 3) or config.keys() != fields:
            raise ValueError("invalid isolated SGLang domain configuration")
        self.config = config
        self.experiment = checked_id(config["experiment_id"])
        self.source = checked_sha(config["source_manifest_sha256"])
        self.launch = checked_sha(config["launch_sha256"])
        directory = Path(directory)
        self.output = (directory / config["receipt_directory"]).resolve()
        self.executor_factory = executor_factory
        self.executor = None
        self.parities = {}
        self.native_repeats = {}
        self.deadline, self.timeout = config["deadline"], config["job_timeout"]
        if (type(self.deadline) is not int or type(self.timeout) is not int
                or not 1 <= self.deadline <= 1800 or not 2 * self.deadline < self.timeout <= 3600):
            raise ValueError("job timeout must cover both serving phases within the child process")
        if (type(config["executor"]) is not dict or config["executor"].keys() not in (
                {"host", "worker", "root", "python", "source", "gpu_indices", "gpu_uuids"},
                {"host", "worker", "root", "python", "source", "gpu_indices", "gpu_uuids", "environment"})
                or config["executor"]["gpu_indices"] != [0, 1]):
            raise ValueError("SGLang requires the qualified two-device launch configuration")
        self.cases = config["cases"]
        if type(self.cases) is not dict or self.cases.keys() != {"train", "validation", "test"}:
            raise ValueError("require distinct frozen train, validation, and test serving cases")
        all_cases = []
        for cases in self.cases.values():
            if type(cases) is not list or not 1 <= len(cases) <= 32:
                raise ValueError("each serving split needs a bounded nonempty case list")
            all_cases.extend(checked_case(spec) for spec in cases)
        if any(len({spec[key] for spec in all_cases}) != len(all_cases) for key in ("id", "workload_sha256", "scenario_sha256")):
            raise ValueError("serving splits cannot reuse the same frozen case or workload identity")
        if any(len({spec[key] for spec in all_cases}) != 1 for key in ("target_sha256", "topology_sha256")):
            raise ValueError("all serving cases must share one fixed model, topology and backing budget")
        self.token_inputs = config["token_inputs"]
        if type(self.token_inputs) is not dict or self.token_inputs.keys() != {spec["id"] for spec in all_cases}:
            raise ValueError("tokenization receipt does not bind all frozen serving cases")
        for inputs in self.token_inputs.values():
            checked_inputs(inputs)
        self.blocks = config["paired_block_ids"]
        if type(self.blocks) is not dict or self.blocks.keys() != {"validation", "test"} or any(
                type(ids) is not list or not ids or len({checked_id(value) for value in ids}) != len(ids) for ids in self.blocks.values()):
            raise ValueError("validation and test require separate predeclared block IDs")
        self.initial = MemoryProgram.from_dict(config["initial_program"])
        self.manual = stock_control_program()
        self.profile = config["hot_profile"]
        local = (directory / config["local_cases"]).resolve(strict=True)
        self.geometry_by_case = {}
        for spec in all_cases:
            path = local / spec["id"]
            target, topology, scenario, _ = load_case(path, spec)
            external, _ = load_external(path, spec, expected_inputs=self.token_inputs[spec["id"]], require_gate=True)
            self.geometry_by_case[spec["id"]] = target, topology, scenario, external
        training = []
        self.preflight = None
        for spec in self.cases["train"]:
            target, topology, scenario, external = self.geometry_by_case[spec["id"]]
            if self.preflight is None:
                self.preflight = (target, topology, scenario, external)
                expected_uuids = [binding["uuid"] for binding in sorted(target["gpu_bindings"].values(), key=lambda row: row["ordinal"])]
                if expected_uuids != config["executor"]["gpu_uuids"]:
                    raise ValueError("leased CUDA UUIDs differ from the fixed SGLang target")
                for program in (None, self.initial, self.manual):
                    self._compile(program)
            context = generator_context(topology, scenario, supported_actions=ACTIONS,
                                        supported_features=FEATURES | HOT_FEATURES | BACKING_FEATURES)
            try:
                self._compile(resident_control_program(), case=spec)
            except (ValueError, MemoryError):
                resident_eligible = False
            else:
                resident_eligible = True
            training.append({"case": spec["id"], "target_sha256": spec["target_sha256"], "pool": target["pool"],
                             "allowed_nodes": target["allowed_nodes"], "resident_6144_eligible": resident_eligible,
                             "case_max_request_tokens": hot_config.case_token_bounds(external)[1], **context})
        eligible = self._compile(None).audit["startup"]["eligible_hot_slots"]
        capacities = {str(slots): self._compile(None, slots).audit["startup"]["metadata_bytes_reserved_by_node"] for slots in eligible}
        self.training_context = {"cases": training, "runtime_semantics": RULE_SEMANTICS,
                                 "candidate_extra_metadata_bytes_by_node": capacities,
                                 "features_by_action": {"configure/hot_capacity": sorted(HOT_FEATURES),
                                                        "configure/backing_mode": sorted(BACKING_FEATURES),
                                                        "allocate/place": sorted(FEATURES)},
                                 "score_unit": "sum_replay_window_ms", "lower_is_better": True}
        adapter_sources = {
            "topomem_sglang_rsi/domain.py": __file__,
            "topomem_sglang_rsi/contract.py": contract.__file__,
            "topomem_sglang_rsi/hot_config.py": hot_config.__file__,
            "rsi_memory/execution_client.py": execution_client.__file__,
            "rsi_memory/execution_worker.py": execution_worker.__file__,
            "rsi_memory/memory_program.py": memory_program.__file__,
            "rsi_memory/topology_rsi.py": topology_rsi.__file__,
        }
        self.contract = {"schema": 1, "kind": SCOPE, "experiment_id": self.experiment, "source_manifest_sha256": self.source,
                         "launch_sha256": self.launch, "nccl_sha256": NCCL_SHA256, "cases": self.cases, "paired_block_ids": self.blocks,
                         "tokenization_sha256": fingerprint(self.token_inputs),
                         "ingress_contract_sha256_by_case": {case: fingerprint(geometry[3][ingress.EXTERNAL_FIELD])
                                                             for case, geometry in self.geometry_by_case.items()},
                         "metadata_profile_sha256": fingerprint(self.profile),
                         "native_comparators": {"default": {"hot_slots": 4096, "backing": "all_host"},
                                                "manual": {"hot_slots": 6144, "backing": "all_host", "backing_mode": "tiered"}},
                         "score": "sum measured post-warmup replay window ms, lower is better", "deadline": self.deadline,
                         "job_timeout": self.timeout, "independent_block": "one fresh leased process per case and arm",
                         "native_qualification": "per-case same-capacity native A/A; tuned native also matches that case's default global forward, dispatch, inputs, output and KV",
                         "paired_order": "predeclared prefix of deterministic balanced Latin order; reversed rows for odd counts",
                         "adapter_sources_sha256": {name: sha256(Path(path).read_bytes()).hexdigest() for name, path in adapter_sources.items()},
                         "initial_program_sha256": self.initial.fingerprint(), "manual_program_sha256": self.manual.fingerprint()}
        if schema == 3:
            from . import manual_selection, manual_sweep

            self.manual, selection, preregistration = manual_selection.verify_selection(self)
            self.contract["manual_program_sha256"] = self.manual.fingerprint()
            self.contract["manual_selection"] = selection
            self.contract["controller_preregistration"] = preregistration
            self.contract["native_comparators"]["manual"] = {key: selection[key] for key in ("hot_slots", "layer_tiers_by_rank", "backing_mode")}
            for module in (manual_selection, manual_sweep):
                path = Path(module.__file__)
                self.contract["adapter_sources_sha256"][f"topomem_sglang_rsi/{path.name}"] = sha256(path.read_bytes()).hexdigest()
            self.training_context["runtime_semantics"] = RULE_SEMANTICS.replace(
                "The strong stock control uses 6144 slots and all-host backing.",
                "The manual control was selected from nine predeclared hand policies on training only.")
            self.training_context["manual_control"] = {key: selection[key] for key in ("hot_slots", "layer_tiers_by_rank", "backing_mode")}
            self.training_context["minimum_paired_improvement_ms"] = preregistration["min_improvement_ms"]
        self.identity = fingerprint(self.contract)
        atexit.register(self.close)

    def close(self):
        if self.executor is not None:
            self.executor.close()
            self.executor = None

    def _client(self):
        if self.executor is None:
            self.executor = self.executor_factory(**self.config["executor"], source_manifest_sha256=self.source, audit_directory=self.output)
        return self.executor

    def _compile(self, program, native_hot=None, *, case=None):
        geometry = self.preflight if case is None else self.geometry_by_case[case["id"]]
        return compile_memory(*geometry, self.profile, program, native_hot=native_hot)

    def effective_program_fingerprint(self, program):
        """Identify the physical startup and cold-placement behavior under frozen training geometry."""
        effects = {}
        for case in self.cases["train"]:
            compiled = self._compile(program, case=case)
            mode = compiled.audit["startup"]["backing_mode"]
            actions = compiled.plan["actions"] if mode == "tiered" else ()
            placements = [{"rank": action["rank"], "layer": action["layer"], "node": action["node"],
                           "tier": compiled.workers[action["rank"]].layer_tiers[action["layer"]],
                           "read_links": action["read_links"], "write_links": action["write_links"]}
                          for action in actions]
            placements.sort(key=lambda row: (row["rank"], row["layer"]))
            effects[case["id"]] = {"hot_slots": compiled.audit["startup"]["hot_slots"],
                                   "backing_mode": mode, "placements": placements}
        return fingerprint({"schema": 2, "training_effects": effects})

    def _job(self, case, arm, phase, program, identity, qualification, *, native_hot=None, cross_cap=None):
        payload = {"schema": 1, "case": case, "arm": arm, "phase": phase,
                   "program": None if program is None else program.to_dict(), "qualification": qualification,
                   "deadline": self.deadline, "launch_sha256": self.launch, "source_manifest_sha256": self.source,
                   "expected_inputs": self.token_inputs[case["id"]], "hot_profile": self.profile,
                   "native_hot": native_hot, "cross_cap_reference": cross_cap}
        job_id = "sg-" + fingerprint({"contract": self.identity, "identity": identity, "payload": payload})[:48]
        client = self._client()
        previous = client.status(job_id)
        if previous["status"] == "not_found":
            receipt = client.run(job_id, EVALUATOR, payload, timeout=self.timeout)
        else:
            receipt = previous
        expected_request = digest({"operation": "run", "job_id": job_id, "module": EVALUATOR, "payload": payload,
                                   "timeout": self.timeout, **client.config})
        if receipt["status"] != "completed" or receipt.get("request_sha256") != expected_request:
            raise ExecutionError("trusted SGLang job failed or is unresolved; inspect receipt " + job_id)
        if previous["status"] == "completed":
            local = self.output / f"{job_id}.json"
            self.output.mkdir(parents=True, exist_ok=True)
            if not local.exists():
                atomic(local, receipt)
        result = receipt["result"]
        expected = self._compile(program, native_hot, case=case).audit
        invariant = ("startup", "metadata_profile_sha256", "base_target_sha256", "base_topology_sha256", "cold_program_sha256",
                     "derived_target_sha256", "derived_topology_sha256", "total_incremental_bytes_by_node", "placement_rule_indices_in_full_program",
                     "equivalent_to_strong_stock_control", "ingress_contract_sha256", "external_scheduler_sha256",
                     "derived_scenario_sha256", "derived_schedule_sha256", "cold_plan_sha256", "cold_plan_role")
        if (result.get("scope") != SCOPE or result.get("case") != case or result.get("arm") != arm or result.get("phase") != phase
                or type(result.get("feasible")) is not bool or result.get("program_sha256") != expected["full_program_sha256"]
                or any(result.get("memory_compilation", {}).get(key) != expected[key] for key in invariant)):
            raise ExecutionError("trusted SGLang result is inconsistent with job " + job_id)
        checked_sha(result.get("raw_result_sha256"))
        if result["feasible"]:
            checked_sha(result.get("metrics", {}).get("global_ticks_sha256"))
            if result["metrics"].get("input_tokenization_sha256") != fingerprint(self.token_inputs[case["id"]]):
                raise ExecutionError("trusted SGLang result changed the case tokenizer commitment " + job_id)
        return job_id, result

    def _parity(self, case, arm, program, *, slots=None):
        slots = self._compile(program, slots, case=case).audit["startup"]["hot_slots"]
        key = (case["id"], arm, slots, None if program is None else program.fingerprint())
        if key not in self.parities:
            native = self._proof(*self._parity(case, "native", None, slots=slots)) if arm != "native" else None
            native_hot = slots if arm == "native" and slots != 4096 else None
            cross_cap = self._proof(*self._parity(case, "native", None)) if native_hot is not None else None
            qualified = self._job(case, arm, "parity", program, "immutable_qualification", native,
                                  native_hot=native_hot, cross_cap=cross_cap)
            if arm == "native":
                repeat_id, repeat = self._job(case, arm, "parity", program, "independent_native_repeat", None,
                                              native_hot=native_hot, cross_cap=cross_cap)
                first = qualified[1]
                if (not first["feasible"] or not repeat["feasible"] or any(first["metrics"][field] != repeat["metrics"][field]
                        for field in (*PARITY_KEYS, "input_tokenization_sha256"))
                        or native_hot is not None and any(row.get("cross_capacity_native_parity", {}).get("matched") is not True
                                                          for row in (first, repeat))):
                    raise ExecutionError("fresh native parity did not repeat; inspect receipt " + repeat_id)
                self.native_repeats[case["id"], slots] = repeat_id
            self.parities[key] = qualified
        return self.parities[key]

    @staticmethod
    def _proof(job_id, result):
        if not result["feasible"]:
            raise ExecutionError("cannot time a failed SGLang qualification " + job_id)
        return {"job_id": job_id, "raw_result_sha256": result["raw_result_sha256"]}

    def _trial(self, split, program, label, identity):
        arm = "native" if program is None else "manual" if label == "manual" else "generated"
        if arm == "manual" and program.fingerprint() != self.manual.fingerprint():
            raise ValueError("paired manual arm changed from its frozen control")
        try:
            compilations = {case["id"]: self._compile(program, case=case).audit for case in self.cases[split]}
        except (ValueError, MemoryError) as exc:
            return {"feasible": False, "score": None, "metrics": {"preflight_rejected": True, "receipts": []},
                    "feedback": {"failure": str(exc)} if split == "train" else {}}
        feedback, receipts, scores = [], [], {}
        for case in self.cases[split]:
            slots = compilations[case["id"]]["startup"]["hot_slots"]
            parity_id, parity = self._parity(case, arm, program)
            receipts.extend((parity_id, self.native_repeats[case["id"], slots]))
            if not parity["feasible"]:
                return {"feasible": False, "score": None, "metrics": {"receipts": receipts},
                        "feedback": {"failure": parity["failure"]} if split == "train" else {}}
            timed_id, timed = self._job(case, arm, "performance", program, identity, self._proof(parity_id, parity))
            receipts.append(timed_id)
            score = timed["score"]
            if not timed["feasible"] or type(score) not in (int, float) or not math.isfinite(score) or score <= 0:
                raise ExecutionError("bound SGLang timed job has no measured score " + timed_id)
            scores[case["id"]] = score
            if split == "train":
                feedback.append({"case": case["id"], "window_ms": score, "cold_miss_rows_by_rank_layer": parity["feedback"]["cold_miss_rows_by_rank_layer"],
                                 "ranks": parity["metrics"]["ranks"], "timing": timed["metrics"]["timing"],
                                 "startup": parity["memory_compilation"]["startup"]})
        return {"feasible": True, "score": sum(scores.values()),
                "metrics": {"receipts": receipts, "case_scores_ms": scores, "score_unit": "sum_replay_window_ms",
                            "equivalent_to_strong_stock_control": all(row["equivalent_to_strong_stock_control"]
                                                                      for row in compilations.values())},
                "feedback": {"cases": feedback} if split == "train" else {}}

    def evaluate(self, split, program, eval_id):
        if split not in self.cases:
            raise ValueError("unknown frozen SGLang split")
        return self._trial(split, program, "candidate", {"eval_id": eval_id, "mode": "unpaired"})

    def evaluate_pair(self, split, variants, eval_id):
        if split not in self.blocks or len(variants) < 2 or "native" not in variants or variants["native"] is not None:
            raise ValueError("paired SGLang trials require a frozen split and native baseline")
        labels = sorted(variants)
        count = len(labels)
        offset = int(sha256(eval_id.encode()).hexdigest()[:8], 16) % count
        base = [0] + [(position + 1) // 2 if position % 2 else count - position // 2 for position in range(1, count)]
        orders = [[labels[(item + shift + offset) % count] for item in base] for shift in range(count)]
        if count % 2:
            orders.extend(list(reversed(order)) for order in orders.copy())
        paired = {label: {"feasible": True, "samples": [], "metrics": {"receipts": [], "execution_order": []}} for label in labels}
        # Keep instrumentation and initial parity work outside all measured pair blocks.
        for label in labels:
            program = variants[label]
            try:
                for case in self.cases[split]:
                    self._compile(program, case=case)
            except (ValueError, MemoryError):
                continue
            arm = "native" if program is None else "manual" if label == "manual" else "generated"
            for case in self.cases[split]:
                self._parity(case, arm, program)
        for index, block in enumerate(self.blocks[split]):
            order = orders[index % len(orders)]
            for label in order:
                trial = self._trial(split, variants[label], label, {"eval_id": eval_id, "label": label, "block": block})
                row = paired[label]
                row["feasible"] = row["feasible"] and trial["feasible"]
                row["metrics"]["execution_order"].append({"block_id": block, "order": order})
                row["metrics"]["receipts"].extend(trial["metrics"]["receipts"])
                if trial["feasible"]:
                    row["metrics"]["equivalent_to_strong_stock_control"] = trial["metrics"]["equivalent_to_strong_stock_control"]
                    row["samples"].append({"block_id": block, "score": trial["score"], "case_scores_ms": trial["metrics"]["case_scores_ms"]})
        return paired

    def domain(self):
        return Domain(self.contract, self.training_context, ACTIONS, self.evaluate, self.initial, self.evaluate_pair, self.manual,
                      effective_program_fingerprint=self.effective_program_fingerprint)


def factory(path):
    path = Path(path).resolve(strict=True)
    return SGLangDomain(read_json(path), path.parent).domain()
