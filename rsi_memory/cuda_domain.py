"""Trusted adapter binding a recursive search to immutable real CUDA cases."""

from __future__ import annotations

import atexit
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median

from .execution_client import PersistentExecutor, ExecutionError
from .memory_program import ACTION_DOMAIN, FEATURE_UNITS, MemoryProgram
from .topology import fingerprint
from .topology_rsi import Domain


RULE_SEMANTICS = """An allocation first creates physical KV at the externally fixed compute rank.
allocate/place picks an authoritative owner route including staying local; a moving place drops
the producer copy unless allocate/replicate explicitly retains its existing zero-hop local copy.
Retain or replicate a hot block explicitly if useful. Live authoritative
owners cannot be evicted. allocate/demote and access/demote can instead move an unprotected
live owner out of the actually blocking pool, only when an allocation needs space. They
score victim/destination pairs; lower score wins. Active selections, the producer and
in-flight blocks are protected. A demotion frees its source only after completion. An
already valid immutable copy at the destination permits a zero-transfer ownership handoff
(target_has_copy=1, route_hops=0); otherwise the real directed route and relay budget apply.
Demotions cannot cycle through the same owner in one pressure resolution. Without pressure
these rules do no work; they differ from proactive access/migrate.
access/replicate|migrate|prefetch chooses a route to the compute node
on demand; the same access rules may also execute once after attention. hint rules see only
previously observed selections after an external append, never future selections. Prefetch's
destination is the local compute node. Target/source features describe each legal candidate
route. Multi-hop relays reserve real pinned-host slots until completed. allocate/access evict
or reclaim rules select safe physical victims under pressure; release/reclaim and idle/reclaim
free zero-reference blocks. A program without all necessary demand, eviction, and reclamation
rules can be infeasible. Rules are first matching per event/action, lower score wins; AST
when must be positive. All shared/COW references, bytes/order, query/selection, rank placement,
scheduling, correctness checks, memory caps and score definition are outside your control.
The objective measures real elapsed work including policy overhead, production, transfers,
attention and mandatory final reclamation; no modeled GPU times replace observations. Pools
are preallocated equally: node budgets bound managed KV slots including relay/COW staging;
common attention workspace is reported separately through actual PyTorch CUDA peaks.
diagnostic_work_counts describes verified policy work by event/action. Separate uninstrumented
trials measure elapsed time and supply the score.
Workloads execute an
ordered logical clock without waiting for external wall-time timestamps. KV/Q values are
deterministically derived fixtures, not model inference or a LongBench quality score."""

_WORK_GROUPS = tuple(f"{event}/{action}" for event, actions in ACTION_DOMAIN.items() for action in actions)
_WORK_FIELDS = ("rows_scanned", "choice_calls", "candidate_records", "predicate_evals", "score_evals",
                "indexed_choices", "static_route_choices", "static_producer_retain_choices")


def _diagnostic_work_counts(diagnostic):
    raw = diagnostic.get("work_counts")
    if type(raw) is not dict:
        return {}
    projected = {}
    for group in _WORK_GROUPS:
        counts = raw.get(group)
        if type(counts) is not dict:
            continue
        row = {name: counts[name] for name in _WORK_FIELDS if type(counts.get(name)) is int and 0 <= counts[name] < 2**63}
        if row:
            projected[group] = row
    return projected


def _feedback(result):
    if not result.get("feasible"):
        return {"failure_type": result.get("error_type"), "failure": result.get("error")}
    cases = []
    for case in result["cases"]:
        diagnostic = case["diagnostic"]
        timings = case["timings"]
        counts = Counter((row.get("rule"), row["action"]) for row in diagnostic["actions"])
        cases.append({"name": case["name"], "fixture_kind": diagnostic["fixture_kind"],
                      "median_elapsed_ms": case["median_elapsed_ns"] / 1e6,
                      "median_transfer_ms": median(row["transfer_wall_ns"] for row in timings) / 1e6,
                      "fetch_misses": diagnostic["total_fetch_misses"],
                      "diagnostic_work_counts": _diagnostic_work_counts(diagnostic),
                      "page_activity": sorted(diagnostic["page_activity"],
                                              key=lambda row: (-row["attention_reads"], row["layer"], row["page"]))[:64],
                      "page_activity_total_entries": len(diagnostic["page_activity"]),
                      "rule_counts": [{"rule": rule, "action": action, "count": count}
                                      for (rule, action), count in counts.items()],
                      "decision_examples": [{key: value for key, value in action.items()
                                             if key in {"step", "action", "block", "src", "dst", "node", "links", "rule"}}
                                            for action in diagnostic["actions"] if action.get("rule") is not None][:8],
                      "link_bytes": diagnostic["link_bytes"],
                      "managed_peak_bytes_by_node": diagnostic["managed_peak_bytes_by_node"],
                      "residual_before_teardown_bytes_by_node": diagnostic["residual_before_teardown_bytes_by_node"],
                      "action_counts": {action: sum(row["action"] == action for row in diagnostic["actions"])
                                        for action in sorted({row["action"] for row in diagnostic["actions"]})}})
    return {"cases": cases, "objective": "sum_case_median_ms", "lower_is_better": True}


class CudaDomain:
    def __init__(self, config, config_directory):
        self.config = config
        self.directory = (config_directory / config["output_directory"]).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cases = config["cases"]
        if set(self.cases) != {"train", "validation", "test"} or any(not x for x in self.cases.values()):
            raise ValueError("three nonempty sealed CUDA splits are required")
        self.contract = {
            "schema": 1, "kind": "real_cuda_fixed_block_sparse",
            "experiment_id": config["experiment_id"],
            "source_manifest_sha256": config["source_manifest_sha256"],
            "split_case_sha256": {split: [fingerprint(case) for case in cases] for split, cases in self.cases.items()},
            "paired_block_ids": config["paired_block_ids"],
            "score": "sum of per-case median total elapsed milliseconds; lower is better",
            "score_scope": "CUDA text-derived attention with physical KV lifecycles; no model-serving claim",
            "memory_budget_scope": "preallocated managed KV slots, including copy/COW/relay slots; fixed attention workspace is common and separately measured",
            "timing_scope": "all externally ordered work and mandatory final reclamation; allocation and compilation outside timing",
            "diagnostic_work_count_scope": "verified policy work only; separate uninstrumented trials supply the score",
            "paired_order": "Williams balanced Latin rows, labels offset by fixed eval-ID SHA; odd label counts also use reversed rows",
            "paired_order_source": "rsi_memory.cuda_domain.CudaDomain.evaluate_pair",
            "warmup": config.get("warmup", 2), "samples": config.get("samples", 3),
            "independent_block": "new OS process and fresh managed CUDA/host pools for every arm in every block",
            "initial_program_sha256": fingerprint(config["initial_program"]),
            "manual_program_sha256": fingerprint(config["manual_program"]),
            "manual_comparator": "explicit hand-written MemoryProgram executed through the identical generated-program backend",
            "adapter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        self.executor = None
        atexit.register(self.close)

    def close(self):
        if self.executor is not None:
            self.executor.close()
            self.executor = None

    def _executor(self):
        if self.executor is None:
            self.executor = PersistentExecutor(
                **self.config["executor"], source_manifest_sha256=self.config["source_manifest_sha256"],
                audit_directory=self.directory / "receipts")
        return self.executor

    def _run(self, split, program, eval_id, label, block="train"):
        payload = {"schema": 1, "arm": "native_lru" if program is None else "generated",
                   "program": None if program is None else program.to_dict(), "cases": self.cases[split],
                   "warmup": self.contract["warmup"], "samples": self.contract["samples"],
                   "expected_gpu_uuids": self.config["executor"]["gpu_uuids"]}
        identity = {"experiment": self.config["experiment_id"], "contract": self.contract,
                    "evaluation": eval_id, "label": label, "block": block, "payload": payload}
        job_id = "eval-" + fingerprint(identity)[:48]
        receipt = self._executor().run(job_id, "integrations.topology_memory.torch_eval", payload,
                                       timeout=self.config.get("timeout", 900))
        if receipt["status"] != "completed" or not isinstance(receipt.get("result"), dict):
            raise ExecutionError("trusted GPU evaluation failed; inspect receipt " + job_id)
        result = receipt["result"]
        if result.get("scope") != "cuda_text_derived_fixed_block_sparse":
            raise ExecutionError("unexpected measurement scope")
        if result["feasible"]:
            if len(result["cases"]) != len(self.cases[split]):
                raise ExecutionError("missing required case result")
            for case, expected in zip(result["cases"], self.cases[split]):
                if case["case_sha256"] != fingerprint(expected) or case["diagnostic"]["status"] != "verified":
                    raise ExecutionError("GPU result did not verify its fixed case")
        return job_id, result

    def evaluate(self, split, program, eval_id):
        if program is not None:
            from integrations.topology_memory.torch_runtime import SUPPORTED_FEATURES
            try:
                program.validate_features(SUPPORTED_FEATURES)
            except ValueError as exc:
                return {"feasible": False, "score": None, "feedback": {"failure": str(exc)},
                        "metrics": {"preflight_rejection": True, "receipts": []}}
        job_id, result = self._run(split, program, eval_id, "native" if program is None else "candidate")
        return {"feasible": result["feasible"], "score": result["score"],
                "feedback": _feedback(result) if split == "train" else {},
                "metrics": {"receipts": [job_id], "scope": result["scope"], "score_unit": "sum_case_median_ms"}}

    def evaluate_pair(self, split, variants, eval_id):
        labels = sorted(variants)
        count = len(labels)
        offset = int(hashlib.sha256(eval_id.encode()).hexdigest()[:8], 16) % count
        base = [0]
        for position in range(1, count):
            base.append((position+1)//2 if position % 2 else count-position//2)
        orders = [[labels[(item+shift+offset) % count] for item in base] for shift in range(count)]
        if count % 2:
            orders += [list(reversed(order)) for order in orders.copy()]
        results = {label: {"feasible": True, "samples": [], "metrics": {"receipts": [], "execution_order": []}}
                   for label in labels}
        for block_index, block_id in enumerate(self.contract["paired_block_ids"][split]):
            order = orders[block_index % len(orders)]
            observed = {}
            for label in order:
                job_id, result = self._run(split, variants[label], eval_id, label, block_id)
                row = results[label]
                row["metrics"]["receipts"].append(job_id)
                row["metrics"]["execution_order"].append({"block_id": block_id, "order": order})
                row["feasible"] = row["feasible"] and result["feasible"]
                if result["feasible"]:
                    observed[label] = [(case["case_sha256"], case["diagnostic"]["selected_and_output_sha256"])
                                       for case in result["cases"]]
                    row["samples"].append({"block_id": block_id, "score": result["score"], "receipt_id": job_id,
                                           "case_scores_ms": {case["name"]: case["median_elapsed_ns"] / 1e6
                                                              for case in result["cases"]}})
            if len({fingerprint(value) for value in observed.values()}) > 1:
                raise ExecutionError("paired arms consumed different selected bytes or produced different attention outputs")
        return results

    def domain(self):
        from integrations.topology_memory.torch_runtime import SUPPORTED_ACTIONS, SUPPORTED_FEATURES
        contexts = [{"name": case["name"], "topology": case["topology"], "geometry": case["compile"],
                     "workload_kind": case["workload"]["kind"], "compute_node": case["compute_node"]}
                    for case in self.cases["train"]]
        return Domain(self.contract, {"cases": contexts,
                                     "feature_units": {name: FEATURE_UNITS[name] for name in sorted(SUPPORTED_FEATURES)},
                                     "runtime_semantics": RULE_SEMANTICS,
                                     "score_unit": "sum_case_median_ms", "lower_is_better": True},
                      tuple(sorted(SUPPORTED_ACTIONS)), self.evaluate,
                      MemoryProgram.from_dict(self.config["initial_program"]), self.evaluate_pair,
                      MemoryProgram.from_dict(self.config["manual_program"]))


def factory(path):
    path = Path(path).resolve(strict=True)
    return CudaDomain(json.loads(path.read_text(encoding="utf-8")), path.parent).domain()
