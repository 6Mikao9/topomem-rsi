"""Add repeated training selection to the existing bounded generator loop."""

from dataclasses import dataclass, replace
import difflib
from hashlib import sha256
import math
from pathlib import Path
from statistics import median

from . import selection_protocol
from .memory_program import MemoryProgram
from .selection_protocol import Measurement, Proposal, Sample, TrainingArchive, TrainingPlan
from .topology import fingerprint
from .topology_rsi import ARCHIVE_VIEW, Config as BaseConfig, META_SYSTEM, SEED_SOURCE, _atomic_json, _entry, _sha
from .topology_rsi_contract import ContractTopologyRSI


@dataclass(frozen=True)
class Config(BaseConfig):
    min_paired_blocks: int = 3
    training_blocks: int = 3
    training_margin_ms: float = 1.0

    def validate(self):
        super().validate()
        if type(self.training_blocks) is not int or not 3 <= self.training_blocks <= 100:
            raise ValueError("training selection requires 3–100 repeated blocks")
        if type(self.training_margin_ms) not in (int, float) or not math.isfinite(self.training_margin_ms) or self.training_margin_ms <= 0:
            raise ValueError("training selection requires a positive finite noise margin")
        if not self.min_paired_blocks:
            raise ValueError("v4 requires paired validation and test")


class TrainingMatrixIncomplete(RuntimeError):
    pass


_LEGACY_META_SELECTION = (
    "Both sides may retain the best measured training program in the common starting archive. A revised\n"
    "generator must produce a distinct selected output and improve over both the parent output and the incumbent."
)
if not META_SYSTEM.endswith(_LEGACY_META_SELECTION):
    raise RuntimeError("the inherited meta selection text differs from the bound v4 predecessor")

META_SYSTEM_V41 = META_SYSTEM.removesuffix(_LEGACY_META_SELECTION) + (
    "The trusted_selection_contract describes the fixed selection and validation procedure. "
    "The parent and child each select a fresh program from their own proposals. "
    "The historical manager incumbent is a separate paired validation reference. "
    "The child must beat both references under the frozen validation gate. "
    "The structure check replays both generator entries on the shared round-start archive."
)

_SELECTION_CONTRACT = {
    "version": "4.1",
    "round_start": "Adaptive parent and child share a frozen starting archive. Fixed branches use their own seed archive.",
    "later_slots": "Later slots see earlier rough measurements from their own branch. These are single-evaluation training feedback.",
    "fresh_outputs": "Parent and child select from their own newly emitted programs using the complete repeated training matrix.",
    "incumbent": "The historical manager incumbent is evaluated as a separate validation reference.",
    "admission": "The fresh child requires a distinct training identity and a stable paired validation gain over the fresh parent and historical incumbent.",
    "structure": "Both generator entries need a structural change on the common round-start context. Reordering identical evidence IDs does not count.",
    "finalists": {
        "manager_best": "Training-selected program from the adaptive discovery archive with its original source attribution.",
        "admitted_output": "Training-selected program actually emitted by an admitted source, with its editing parents recorded.",
        "fixed": "Training-selected program from the fixed seed archive.",
    },
}


def _orders(effects, key, blocks):
    count = len(effects)
    if not count:
        return [[] for _ in blocks]
    offset = int(fingerprint(key)[:8], 16) % count
    pattern = [0] + [(i + 1) // 2 if i % 2 else count - i // 2 for i in range(1, count)]
    rows = [[effects[(offset + row + i) % count] for i in pattern] for row in range(count)]
    if count % 2:
        rows.extend(list(reversed(row)) for row in rows.copy())
    return [rows[i % len(rows)] for i in range(len(blocks))]


def structural_fields(before, after):
    """Check observed generator decisions while ignoring wording and evidence order."""
    return sorted(key for key in before if key != "guidance" and (
        set(before[key]) != set(after[key]) if key == "evidence" else before[key] != after[key]))


class TopologyRSIV4(ContractTopologyRSI):
    """Keep manager discovery and generator-source evaluation on separate selection paths."""

    def __init__(self, output, config, domain, request, *, backend_source, seed_source=SEED_SOURCE):
        if type(config) is not Config or domain.effective_program_fingerprint is None:
            raise ValueError("v4 requires its repeated-training config and a trusted training effect callback")
        root = Path(__file__).resolve().parents[1]
        sources = (Path(__file__).resolve(), Path(selection_protocol.__file__).resolve())
        protocol = {
            "schema": 1, "kind": "training_paired_declared_identity_v41", "version": "4.1",
            "training_blocks": config.training_blocks, "training_margin_ms": config.training_margin_ms,
            "trusted_selection_contract": {**_SELECTION_CONTRACT, "training_blocks": config.training_blocks,
                                           "training_margin_ms": config.training_margin_ms},
            "rough_evaluations_per_valid_slot": 1,
            "same_start": "Each arm receives the same frozen starting view for its lane. Later slots see only their own rough results.",
            "training_identity": domain.contract.get("training_effect_identity", "trusted backend training effect callback"),
            "receipt_identity": "Canonical ordered evaluator-reported receipt IDs, linked to the complete evaluation record.",
            "archive_lanes": ["adaptive", "fixed"],
            "manager_and_admitted_outputs": "independent training selections",
            "source_sha256": {path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest() for path in sources},
        }
        if "selection_protocol" in domain.contract:
            raise ValueError("selection protocol must be bound by the v4 controller")
        domain = replace(domain, contract={**domain.contract, "selection_protocol": protocol})
        super().__init__(output, config, domain, request, backend_source=backend_source, seed_source=seed_source)

    def _api(self, key, system, envelope, source_sha):
        if system == META_SYSTEM:
            system = META_SYSTEM_V41
        contract = self.manifest["contract"]["selection_protocol"]["trusted_selection_contract"]
        return super()._api(key, system, {**envelope, "trusted_selection_contract": contract}, source_sha)

    def _seal(self, key, value):
        saved = self._get(key)
        if saved is not None and saved != value:
            raise ValueError(f"v4 ledger seal changed for {key}")
        if saved is None:
            self._save(key, value)
        return value

    def _training_effect(self, program):
        try:
            return self._effective_digest(program)
        except (ValueError, MemoryError):
            return None

    def _previous_context(self, previous, lane, native_train, bootstrap):
        evidence = {"native": {"score": self._score(native_train), "status": native_train["status"],
                              "feedback": native_train.get("result", {}).get("feedback", {})}}
        candidates, refs = {}, {}
        if bootstrap is not None:
            evidence[bootstrap["id"]] = {k: bootstrap[k] for k in ("score", "status", "feedback")}
            candidates[bootstrap["program_sha256"]] = {"program": bootstrap["program"], "score": bootstrap["score"]}
            refs[bootstrap["program_sha256"]] = bootstrap["id"]
        for row in previous:
            trials = row["fixed"] if lane == "fixed" else row["parent"] + row["child"]
            for trial in trials:
                repeated = row["v4_training"]["measurements"].get(trial["training_effect_sha256"])
                score = repeated["median_ms"] if repeated and repeated["status"] == "complete" else None
                evidence[trial["id"]] = {"score": score, "status": trial["status"], "feedback": trial["feedback"]}
        if previous:
            last = previous[-1]["v4_training"]
            pool = sorted(last["cohort_proposals"].values(), key=lambda trial: trial["id"])
            for trial in pool:
                if trial["lane"] != lane:
                    continue
                physical = trial["training_effect_sha256"]
                measured = last["measurements"].get(physical)
                if not measured or measured["status"] != "complete":
                    continue
                raw = trial["program_sha256"]
                if raw not in refs or refs[raw] == "bootstrap/initial-incumbent":
                    candidates[raw] = {"program": trial["program"], "score": measured["median_ms"]}
                    refs[raw] = trial["id"]
            if bootstrap is not None and refs[bootstrap["program_sha256"]] == bootstrap["id"]:
                measured = last["measurements"].get(self._training_effect(bootstrap["program"]))
                if measured is None or measured["status"] != "complete":
                    candidates.pop(bootstrap["program_sha256"])
                    refs.pop(bootstrap["program_sha256"])
                else:
                    candidates[bootstrap["program_sha256"]]["score"] = measured["median_ms"]
        ranked = sorted(candidates, key=lambda raw: (candidates[raw]["score"] is None, candidates[raw]["score"], raw))[:ARCHIVE_VIEW]
        recent = list(reversed(list(evidence)[1:]))[:ARCHIVE_VIEW - 1] + ["native"]
        return ({raw: candidates[raw] for raw in ranked}, {key: evidence[key] for key in recent},
                {raw: refs[raw] for raw in ranked})

    def _new_trials(self, prefix, label, lane, source, archive, evidence, refs, number, offset=0):
        rows = []
        for slot in range(self.config.slots):
            trial_id = f"{prefix}/{label}/{slot:02d}"
            self._seal("selection_context/" + trial_id, {
                "archive_sha256": fingerprint({"programs": archive, "evidence": evidence}),
                "parent_proposal_ids": refs,
            })
            trial = self._trial(trial_id, self._context(number, slot + offset, source, archive, evidence), source, archive)
            decision = trial["decision"]
            trial.update(lane=lane, parent_proposal_ids=[refs[raw] for raw in decision["parents"]] if decision else [],
                         training_effect_sha256=self._training_effect(trial["program"]) if trial["program"] is not None else None)
            rows.append(trial)
            if trial["status"] in ("evaluation_interrupted_unknown", "evaluation_failed"):
                self._seal(f"training_matrix/{prefix}/rough_stop", {"trial": trial})
                raise TrainingMatrixIncomplete("a rough training evaluation did not settle")
            if slot + 1 == self.config.slots:
                continue
            rough = {key: trial[key] for key in ("score", "status", "feedback")}
            evidence = {trial_id: {**rough, "measurement_kind": "single_rough_evaluation"}, **evidence}
            recent = [key for key in evidence if key != "native"][:ARCHIVE_VIEW - 1] + ["native"]
            evidence = {key: evidence[key] for key in recent}
            if trial["score"] is not None and trial["training_effect_sha256"] is not None:
                raw = trial["program_sha256"]
                current = {"program": trial["program"], "score": trial["score"],
                           "measurement_kind": "single_rough_evaluation"}
                older = [key for key in archive if key != raw][:ARCHIVE_VIEW - 1]
                archive = {raw: current, **{key: archive[key] for key in older}}
                refs = {raw: trial_id, **{key: refs[key] for key in older}}
        return rows

    def _training_matrix(self, prefix, cohort, anchor, all_trials, bootstrap):
        representatives = {}
        for trial in sorted(cohort.values(), key=lambda row: row["id"]):
            physical = trial["training_effect_sha256"]
            if physical is not None:
                representatives.setdefault(physical, trial)
        if anchor is not None:
            physical = self._training_effect(anchor["program"])
            if physical is not None:
                representatives.setdefault(physical, anchor)

        block_ids = [f"{prefix}-train-{number:02d}" for number in range(1, self.config.training_blocks + 1)]
        order = _orders(sorted(representatives), prefix, block_ids)
        plan = {"schema": 1, "round": prefix, "scope": "training", "block_ids": block_ids,
                "noise_margin_ms": self.config.training_margin_ms, "order": order,
                "representatives": {effect: {"trial_id": trial["id"], "program_sha256": trial["program_sha256"]}
                                    for effect, trial in representatives.items()}}
        scope = fingerprint({"contract": self.manifest["contract"], "plan": plan})
        plan["scope_sha256"] = scope
        self._seal(f"training_matrix/{prefix}/plan", plan)
        observations = {effect: [] for effect in representatives}
        states = {effect: "complete" for effect in representatives}
        stopped = False
        receipt_bundles = set()
        for block, row in zip(block_ids, order, strict=True):
            for effect in row:
                if stopped or states[effect] != "complete":
                    continue
                key = f"{prefix}/training-matrix/{block}/{effect}"
                program = MemoryProgram.from_dict(representatives[effect]["program"])
                measured = self._evaluate(key, "train", program)
                evidence = {"block_id": block, "evaluation_key": "eval/" + key, "status": measured["status"],
                            "evaluation_sha256": fingerprint({"evaluation_key": key, "record": measured})}
                observations[effect].append(evidence)
                if measured["status"] != "completed":
                    states[effect] = "incomplete"
                    stopped = True
                    continue
                result = measured["result"]
                evidence["feasible"] = result["feasible"]
                receipts = result["metrics"].get("receipts", [])
                evidence["reported_receipts"] = receipts
                if not result["feasible"]:
                    states[effect] = "correctness_failed"
                    continue
                if (result["score"] <= 0 or type(receipts) is not list or not receipts
                        or any(type(receipt) is not str or not receipt for receipt in receipts)):
                    evidence["measurement_error"] = "missing_or_invalid_timing_receipts"
                    states[effect] = "incomplete"
                    stopped = True
                    continue
                bundle = fingerprint(receipts)
                if bundle in receipt_bundles:
                    evidence["measurement_error"] = "reused_timing_receipt_bundle"
                    states[effect] = "incomplete"
                    stopped = True
                    continue
                receipt_bundles.add(bundle)
                evidence["receipt_bundle_sha256"] = bundle
                evidence["window_ms"] = result["score"]

        measures, summaries = [], {}
        for effect, rows in observations.items():
            status = states[effect]
            times = tuple(Sample(row["block_id"], row["window_ms"], row["receipt_bundle_sha256"])
                          for row in rows if "window_ms" in row)
            if status == "complete" and len(times) != len(block_ids):
                status = "incomplete"
            proof = fingerprint({"scope_sha256": scope, "effect_sha256": effect, "evaluations": rows})
            measures.append(Measurement(scope, effect, proof, status, () if status == "correctness_failed" else times))
            summaries[effect] = {"status": status, "median_ms": median(item.window_ms for item in times) if status == "complete" else None,
                                 "qualification_record_sha256": proof, "evaluations": rows}
        matrix = self._seal(f"training_matrix/{prefix}/result",
                            {"plan": plan, "measurements": summaries, "cohort_proposals": cohort})
        if any(row["status"] == "incomplete" for row in summaries.values()):
            raise TrainingMatrixIncomplete("an unknown or incomplete training matrix cannot be used for selection")
        proposals = [Proposal(trial["id"], trial["lane"], trial["harness_sha256"], trial["program_sha256"],
                              trial["training_effect_sha256"], tuple(trial["parent_proposal_ids"]))
                     for trial in all_trials.values()]
        archive = TrainingArchive(TrainingPlan(scope, tuple(block_ids), self.config.training_margin_ms), proposals, measures,
                                  trusted_anchor_ids=(bootstrap["id"],) if bootstrap is not None else ())
        return matrix, archive

    def _round(self, number, previous, native_train, native_validation, bootstrap):
        prefix = f"r{number:04d}"
        parent_sha = previous[-1]["active_after"] if previous else _sha(self.seed_source)
        parent = self._read_source(parent_sha)
        adaptive, evidence, adaptive_refs = self._previous_context(previous, "adaptive", native_train, bootstrap)
        fixed, fixed_evidence, fixed_refs = self._previous_context(previous, "fixed", native_train, bootstrap)
        self._seal(f"selection_context/{prefix}", {
            "adaptive_sha256": fingerprint({"programs": adaptive, "evidence": evidence}), "adaptive_parent_ids": adaptive_refs,
            "fixed_sha256": fingerprint({"programs": fixed, "evidence": fixed_evidence}), "fixed_parent_ids": fixed_refs,
        })
        active_context = self._context(number, 0, parent, adaptive, evidence)
        fixed_context = self._context(number, 0, self.seed_source, fixed, fixed_evidence)
        decision, child, plan, failure = self._meta(prefix + "/adaptive-meta", active_context, parent, adaptive)
        child_sha = self._source(child) if child is not None else None
        adaptive_plan_id = self._record_meta_plan(prefix, "adaptive", parent_sha, child_sha, decision, plan, failure)
        fixed_decision, discarded, fixed_plan, fixed_failure = self._meta(prefix + "/fixed-meta", fixed_context, self.seed_source, fixed)
        discarded_sha = _sha(discarded) if discarded is not None else None
        fixed_plan_id = self._record_meta_plan(prefix, "fixed", _sha(self.seed_source), discarded_sha, fixed_decision, fixed_plan, fixed_failure)
        changes = None
        if child is not None:
            try:
                changes = self._behavior(parent, child, number, adaptive, evidence)
            except (ValueError, TimeoutError) as error:
                failure = "invalid_child_entry_" + type(error).__name__

        sources = {"parent": parent, "child": child if failure is None else parent}
        execution = ("parent", "child") if number % 2 == 0 else ("child", "parent")
        adaptive_trials = {label: self._new_trials(prefix, label, "adaptive", sources[label], adaptive, evidence,
                                                  adaptive_refs, number) for label in execution}
        parent_trials, child_trials = (adaptive_trials[label] for label in ("parent", "child"))
        fixed_trials = (self._new_trials(prefix, "fixed-a", "fixed", self.seed_source, fixed, fixed_evidence, fixed_refs, number)
                        + self._new_trials(prefix, "fixed-b", "fixed", self.seed_source, fixed, fixed_evidence, fixed_refs,
                                           number, self.config.slots))
        fresh = [*parent_trials, *child_trials, *fixed_trials]
        history = {trial["id"]: trial for row in previous for trial in row["parent"] + row["child"] + row["fixed"]
                   if trial["program"] is not None}
        current = {trial["id"]: trial for trial in fresh if trial["program"] is not None}
        all_trials = {**history, **current}
        last = previous[-1]["v4_selection"] if previous else {}
        anchor_ids = {trial_id for name in ("manager_best", "admitted_output", "fixed")
                      if (trial_id := last.get(name, {}).get("proposal_id"))}
        cohort = {key: trial for key, trial in all_trials.items() if key in current or key in anchor_ids}
        manager_id = last.get("manager_best", {}).get("proposal_id")
        incumbent = history[manager_id] if manager_id else bootstrap
        matrix, archive = self._training_matrix(prefix, cohort, incumbent, all_trials, bootstrap)

        def pick(lane, ids, *, allowed=None, retained=None):
            choice = archive.choose(lane, proposal_ids=tuple(ids), admitted_sources=allowed, incumbent_id=retained)
            proposal = choice.proposal
            return {
                "status": choice.status, "proposal_id": proposal.proposal_id if proposal else None,
                "effect_sha256": proposal.effect_sha256 if proposal else None, "frontier_effects": list(choice.frontier_effects),
                "excluded_infeasible_proposals": list(choice.excluded_infeasible_proposals),
                "source_sha256": proposal.source_sha256 if proposal else None,
            }

        selections = {name: pick(lane, (trial["id"] for trial in trials if trial["program"] is not None))
                      for name, lane, trials in (("parent", "adaptive", parent_trials), ("child", "adaptive", child_trials))}
        selections["manager_best"] = pick("adaptive", cohort, retained=manager_id)
        selections["fixed"] = pick("fixed", cohort, allowed=(_sha(self.seed_source),), retained=last.get("fixed", {}).get("proposal_id"))
        outputs = {name: all_trials.get(selections[name]["proposal_id"]) for name in ("parent", "child", "fixed")}
        incumbent_effect = self._training_effect(incumbent["program"]) if incumbent else None
        if incumbent_effect and matrix["measurements"][incumbent_effect]["status"] != "complete":
            incumbent = None
        validation, paired = self._validate_round(prefix, {**outputs, "incumbent": incumbent})
        if failure is None and not all(change["ast_changed"] and any(
                structural_fields(row["before"], row["after"]) for row in change["decisions"]) for change in changes.values()):
            failure = "both_entries_need_structural_behavior_change"
        if failure is None and (self._score(native_validation) is None or any(
                validation[label] is None or validation[label]["score"] is None for label in ("parent", "child"))):
            failure = "missing_feasible_matched_validation"
        if failure is None and incumbent is None:
            failure = "missing_measured_incumbent"
        if failure is None and (selections["child"]["effect_sha256"] == selections["parent"]["effect_sha256"]
                                or selections["child"]["effect_sha256"] == incumbent_effect):
            failure = "identical_training_effect_child_and_reference"
        if failure is None:
            for reference in ("parent", "incumbent"):
                if not paired["comparisons"][f"child_vs_{reference}"]["passed"]:
                    failure = f"insufficient_stable_paired_improvement_vs_{reference}"
                    break
        admitted = {_sha(self.seed_source), *(row["active_after"] for row in previous if row["decision"] == "admitted")}
        if failure is None:
            admitted.add(child_sha)
        selections["admitted_output"] = pick("adaptive", cohort, allowed=admitted,
                                             retained=last.get("admitted_output", {}).get("proposal_id"))

        row = {"round": number, "active_before": parent_sha, "active_after": child_sha if failure is None else parent_sha,
               "proposed_child_sha256": child_sha, "decision": "admitted" if failure is None else "rejected",
               "reason": "matched_validation_improvement" if failure is None else failure,
               "source_diff": list(difflib.unified_diff(parent.splitlines(), child.splitlines(), n=2)) if child else [],
               "controlled_behavior": changes, "matched_context_sha256": fingerprint({"parents": adaptive, "evidence": evidence}),
               "adaptive_meta_decision": decision, "fixed_meta_decision": fixed_decision, "adaptive_meta_plan": plan,
               "fixed_meta_plan": fixed_plan, "adaptive_meta_plan_id": adaptive_plan_id, "fixed_meta_plan_id": fixed_plan_id,
               "fixed_meta_failure": fixed_failure, "fixed_meta_discarded_sha256": discarded_sha,
               "child_fallback_to_parent": sources["child"] == parent and child != parent,
               "parent": parent_trials, "child": child_trials, "fixed": fixed_trials,
               "parent_validation": validation["parent"], "child_validation": validation["child"],
               "fixed_validation": [validation["fixed"]], "incumbent_validation": validation["incumbent"],
               "paired_validation": paired, "v4_training": matrix, "v4_selection": selections,
               "v4_editing_ancestry_diagnostic": list(archive.fully_admitted_ancestry_ids(admitted))}
        if previous and previous[-1]["decision"] == "admitted":
            row["previous_admission_reuse"] = self._actual_reuse(
                prefix, parent_sha, previous[-1]["active_before"], decision, active_context, adaptive, parent_trials)
        self._seal("round/" + prefix, row)
        directory = self.output / "rounds"
        directory.mkdir(exist_ok=True)
        _atomic_json(directory / (prefix + ".json"), row)
        return row

    def _actual_reuse(self, prefix, active_sha, old_sha, decision, context, archive, trials):
        meta_id = f"api/{prefix}/adaptive-meta"
        meta_call = self._get(meta_id)
        task_ids = []
        for trial in trials:
            key = "api/" + trial["id"]
            call = self._get(key)
            if (call and call["status"] == "completed" and call["input"]["harness_sha256"] == active_sha
                    and trial["program"] is not None):
                task_ids.append(key)
        meta_completed = bool(meta_call and meta_call["status"] == "completed" and meta_call["input"]["harness_sha256"] == active_sha)
        old = self._read_source(old_sha)
        try:
            shadow = _entry(old, "build_meta_prompt", {**context, "harness_source": old}, archive)
        except (ValueError, TimeoutError):
            shadow = None
        fields = structural_fields(shadow, decision) if shadow is not None and decision is not None else []
        return {"source_sha256": active_sha, "task_api_completed": len(task_ids), "task_api_ids": task_ids,
                "task_program_trial_ids": [key.removeprefix("api/") for key in task_ids],
                "meta_api_completed": meta_completed, "meta_api_id": meta_id if meta_completed else None,
                "actual_meta_decision": decision, "previous_source_same_input": shadow,
                "previous_source_sha256": old_sha,
                "matched_context_sha256": self._get(f"selection_context/{prefix}")["adaptive_sha256"],
                "changed_structural_fields": fields,
                "recursive_behavior_observed": bool(task_ids and meta_completed and fields)}

    def run(self, round_limit=None):
        try:
            return super().run(round_limit)
        except TrainingMatrixIncomplete:
            self._lock()
            try:
                return self._report("training_matrix_incomplete")
            finally:
                self._unlock()

    def _report(self, status):
        report = super()._report(status)
        report["training_matrices"] = dict(self._records("training_matrix/"))
        report["selection_contexts"] = dict(self._records("selection_context/"))
        report["test_program_relations"] = self._get("test_program_relations")
        _atomic_json(self.output / "report.json", report)
        return report

    def freeze(self, *, allow_partial=False):
        self._lock()
        try:
            prior = self._get("frozen")
            if prior is not None:
                _atomic_json(self.output / "frozen.json", prior)
                return prior
            rows = [row for _, row in self._records("round/")]
            if not rows or (len(rows) != self.config.rounds and not allow_partial):
                raise ValueError("configured rounds are incomplete; explicitly allow a partial freeze")
            last = rows[-1]
            finalized = {}
            for label in ("manager_best", "admitted_output", "fixed"):
                choice = last["v4_selection"][label]
                trial_id = choice["proposal_id"]
                if trial_id is None:
                    raise ValueError(f"no eligible training finalist for {label}")
                trial = last["v4_training"]["cohort_proposals"][trial_id]
                measured = last["v4_training"]["measurements"][choice["effect_sha256"]]
                finalized[label] = {"trial": trial, "training_score": measured["median_ms"], "selection": choice}
            frozen = {"rounds": len(rows), "partial": len(rows) != self.config.rounds, **finalized,
                      "active_harness_sha256": last["active_after"],
                      "admitted_harness_sha256s": sorted({_sha(self.seed_source), *(r["active_after"] for r in rows if r["decision"] == "admitted")}),
                      "reused_harness_sha256s": sorted({reuse["source_sha256"] for r in rows
                                                       if (reuse := r.get("previous_admission_reuse")) and reuse["recursive_behavior_observed"]}),
                      "manifest_sha256": fingerprint(self._get("manifest")), "rounds_sha256": fingerprint(rows),
                      "training_matrix_sha256": fingerprint(last["v4_training"])}
            self._seal("frozen", frozen)
            _atomic_json(self.output / "frozen.json", frozen)
            self._report("frozen")
            return frozen
        finally:
            self._unlock()

    def test(self):
        self._lock()
        try:
            frozen = self._get("frozen")
            if frozen is None:
                raise ValueError("test is sealed until training finalists are frozen")
            candidates = {name: MemoryProgram.from_dict(frozen[name]["trial"]["program"])
                          for name in ("manager_best", "admitted_output", "fixed")}
            measured = self._evaluate_pair("frozen/test", "test", candidates, time_budget=False)
            if measured["status"] != "completed":
                return self._report("test_incomplete")
            if any(not arm["feasible"] for arm in measured["result"].values()):
                return self._report("test_infeasible")
            references = ["native", "fixed"] + (["manual"] if self.domain.manual_program is not None else [])
            self._seal("test_comparisons", {f"{name}_vs_{ref}": self._comparison(measured, name, ref)
                                          for name in ("manager_best", "admitted_output") for ref in references})
            declared = {**candidates, "native": None}
            if self.domain.manual_program is not None:
                declared["manual"] = self.domain.manual_program
            self._seal("test_program_relations", {
                f"{name}_vs_{ref}": {
                    "same_raw_program": declared[ref] is not None and candidates[name].fingerprint() == declared[ref].fingerprint(),
                    "same_training_effect": declared[ref] is not None and self._training_effect(candidates[name].to_dict())
                    == self._training_effect(declared[ref].to_dict()),
                } for name in ("manager_best", "admitted_output") for ref in references
            })
            return self._report("test_settled")
        finally:
            self._unlock()
