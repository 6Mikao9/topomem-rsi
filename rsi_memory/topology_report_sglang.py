"""Bind SGLang held-out scores to the recorded memory layouts of the leased jobs."""

from collections import Counter
import re


_SHA = re.compile(r"[0-9a-f]{64}")
_CASE_KEYS = {"id", "manifest_sha256", "target_sha256", "topology_sha256", "scenario_sha256", "workload_sha256", "schedule_sha256"}
_RANK_FIELDS = ("packed_device_payload_bytes", "device_physical_rows", "device_logical_row_capacity",
                "request_metadata_slots_including_pad", "hot_metadata_bytes", "pinned_host_bytes", "local_hbm_bytes", "peer_hbm_bytes")
_TIERS = {"host", "local_hbm", "peer_hbm", "peer_relay", "resident"}


def _sha(value):
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _nonnegative_int(value):
    return type(value) is int and value >= 0


def _rank_configuration(receipt):
    serving = receipt["result"]["serving"]
    startup = serving["memory_compilation"]["startup"]
    metrics = serving["metrics"]
    slots, mode = startup.get("hot_slots"), startup.get("backing_mode")
    ordinals, uuids, ranks = receipt.get("gpu_indices"), receipt.get("gpu_uuids"), metrics["ranks"]
    if (type(slots) is not int or slots <= 0 or mode not in ("tiered", "resident") or metrics.get("backing_mode") != mode
            or ordinals != [0, 1] or not isinstance(uuids, list) or len(uuids) != 2
            or any(not isinstance(uuid, str) or not uuid for uuid in uuids) or uuids[0] == uuids[1]
            or [rank.get("rank") for rank in ranks] != [0, 1]):
        return None
    layout = []
    for rank, uuid in zip(ranks, uuids, strict=True):
        tiers = rank.get("layer_tiers")
        if (rank.get("hot_slots") != slots or not isinstance(tiers, list) or len(tiers) != 6
                or any(not isinstance(tier, str) or tier not in _TIERS for tier in tiers)
                or ("resident" in tiers) != (mode == "resident") or (mode == "resident" and tiers != ["resident"] * 6)
                or any(not _nonnegative_int(rank.get(key)) for key in _RANK_FIELDS)):
            return None
        layout.append({"rank": rank["rank"], "gpu_uuid": uuid, "layer_tiers": tiers,
                       **{key: rank[key] for key in _RANK_FIELDS}})
    return {"hot_slots": slots, "backing_mode": mode, "ranks": layout}


def _observed(receipt, case, source, profile, program, arm, phase):
    if not receipt or receipt.get("canonical_result_matches") is not True or receipt.get("status") != "completed":
        return None
    if (receipt.get("returncode") != 0 or receipt.get("process_group_remaining") is not False
            or receipt.get("process_group_cleanup_required") is not False or receipt.get("remaining_compute_pids") != ""
            or receipt.get("source_manifest_sha256") != source or not _sha(receipt.get("request_sha256"))):
        return None
    result = receipt.get("result") or {}
    serving = result.get("serving") or {}
    comp = serving.get("memory_compilation", {})
    if (result.get("feasible") is not True or serving.get("case") != case or serving.get("arm") != arm
            or serving.get("phase") != phase or "program_sha256" not in serving or serving["program_sha256"] != program
            or "full_program_sha256" not in comp or comp["full_program_sha256"] != program
            or comp.get("metadata_profile_sha256") != profile or comp.get("base_target_sha256") != case["target_sha256"]
            or comp.get("base_topology_sha256") != case["topology_sha256"] or not _sha(serving.get("raw_result_sha256"))):
        return None
    return _rank_configuration(receipt)


def _arm_configurations(arm, label, program, blocks, cases, receipts, timing_counts, source, profile):
    refs = arm.get("metrics", {}).get("receipts", [])
    samples = arm.get("samples", [])
    if (arm.get("feasible") is not True or not isinstance(refs, list) or len(refs) != 3 * len(blocks) * len(cases)
            or any(not isinstance(ref, str) for ref in refs) or [sample.get("block_id") for sample in samples] != blocks):
        return None
    receipt_arm = label if label in ("manual", "native") else "generated"
    jobs = iter(refs)
    configurations = {case["id"]: [] for case in cases}
    case_ids = configurations.keys()
    for sample in samples:
        scores = sample.get("case_scores_ms")
        if (not isinstance(scores, dict) or scores.keys() != case_ids or any(type(score) not in (int, float) or score <= 0
                for score in scores.values()) or sum(scores[case["id"]] for case in cases) != sample["score"]):
            return None
        for case in cases:
            parity, native_repeat, timed = next(jobs), next(jobs), next(jobs)
            timing = receipts.get(timed)
            layout = _observed(timing, case, source, profile, program, receipt_arm, "performance")
            if layout is None or timing_counts[timed] != 1 or timing["result"].get("score") != scores[case["id"]]:
                return None
            verified = _observed(receipts.get(parity), case, source, profile, program, receipt_arm, "parity")
            repeat = _observed(receipts.get(native_repeat), case, source, profile, None, "native", "parity")
            if (layout != verified or repeat is None or repeat["hot_slots"] != layout["hot_slots"]
                    or repeat["backing_mode"] != "tiered" or any(rank["gpu_uuid"] != other["gpu_uuid"]
                       for rank, other in zip(layout["ranks"], repeat["ranks"], strict=True))):
                return None
            configurations[case["id"]].append(layout)
    return configurations


def compare_heldout_configurations(pair, contract, program_hashes, receipts):
    """Return a descriptive comparison only when all 12 blocks bind for both arms."""
    blocks, arms = pair["input"]["block_ids"], pair["result"]
    cases = contract.get("cases", {}).get("test")
    source = contract.get("source_manifest_sha256")
    profile = contract.get("metadata_profile_sha256")
    comparisons = {label: {"status": "unverified"} for label in arms if label != "adaptive"}
    if (len(blocks) != 12 or not isinstance(cases, list) or not cases or not _sha(source) or not _sha(profile)
            or any(not isinstance(case, dict) or case.keys() != _CASE_KEYS or not isinstance(case["id"], str)
                   or any(not _sha(case[key]) for key in _CASE_KEYS - {"id"}) for case in cases)
            or len({case["id"] for case in cases}) != len(cases)):
        return comparisons
    references = [arm.get("metrics", {}).get("receipts") for arm in arms.values()]
    timing_counts = Counter(ref for refs in references if isinstance(refs, list) for ref in refs[2::3] if isinstance(ref, str))
    layouts = {label: _arm_configurations(arm, label, program_hashes[label], blocks, cases, receipts, timing_counts, source, profile)
               for label, arm in arms.items()}
    candidate = layouts["adaptive"]
    if candidate is None:
        return comparisons
    for label in comparisons:
        reference = layouts[label]
        if reference is None:
            continue
        observations = []
        for case in cases:
            key = case["id"]
            matches = [a == b for a, b in zip(candidate[key], reference[key], strict=True)]
            observations.append({"id": key, "matching_blocks": sum(matches),
                                 "adaptive": candidate[key][0] if all(row == candidate[key][0] for row in candidate[key]) else None,
                                 "reference": reference[key][0] if all(row == reference[key][0] for row in reference[key]) else None})
        count = len(blocks) * len(cases)
        matched = sum(row["matching_blocks"] for row in observations)
        comparisons[label] = {"status": "same" if matched == count else "different",
                              "matched": matched, "observations": count, "cases": observations}
    return comparisons
