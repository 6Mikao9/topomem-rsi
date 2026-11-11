"""Offline, prespecified paired-block summaries of settled CUDA evaluations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from random import Random
from statistics import median

from .topology import _fields, fingerprint


_AGGREGATE = "aggregate"


def _labels(values, label):
    if (type(values) is not list or not values or any(type(value) is not str or not value for value in values)
            or len(values) != len(set(values))):
        raise ValueError(f"{label} must be a nonempty list of distinct identifiers")
    return values


def _number(value, label, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"invalid {label}")
    return value


def _quantile(ordered, fraction):
    position = (len(ordered)-1)*fraction
    index = math.floor(position)
    return ordered[index] + (ordered[math.ceil(position)]-ordered[index])*(position-index)


def _samples(record, blocks, cases):
    if record is None:
        return "missing", {}
    if type(record.get("feasible")) is not bool or type(record.get("samples")) is not list:
        raise ValueError("expected settled paired-arm result with feasibility and samples")
    by_block = {}
    for sample in record["samples"]:
        block = sample["block_id"]
        if block not in blocks or block in by_block:
            raise ValueError("unexpected or duplicate paired block")
        times = sample["case_scores_ms"]
        if type(times) is not dict or set(times) != set(cases):
            raise ValueError("sample must report every preregistered case")
        score = _number(sample["score"], "aggregate score", positive=True)
        for value in times.values():
            _number(value, "case latency", positive=True)
        if not math.isclose(score, sum(times.values()), rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("aggregate score differs from the sum of fixed case scores")
        by_block[block] = {**times, _AGGREGATE: score}
    return "feasible" if record["feasible"] else "infeasible", by_block


def summarize_paired(plan, result, *, expected_plan_sha256):
    """Report every prespecified contrast without changing the controller's gate."""
    _fields(plan, ("schema", "kind", "block_ids", "case_names", "candidate", "comparators", "threshold_ms", "bootstrap"))
    if type(plan["schema"]) is not int or plan["schema"] != 1 or plan["kind"] != "paired_latency_preregistration":
        raise ValueError("unsupported paired-analysis plan")
    plan_sha = fingerprint(plan)
    if plan_sha != expected_plan_sha256:
        raise ValueError("analysis plan differs from its preregistered fingerprint")
    blocks = _labels(plan["block_ids"], "block IDs")
    cases = _labels(plan["case_names"], "case names")
    references = _labels(plan["comparators"], "comparators")
    candidate = plan["candidate"]
    if len(blocks) != 12 or _AGGREGATE in cases or type(candidate) is not str or not candidate or candidate in references:
        raise ValueError("expected 12 preregistered paired blocks and distinct candidate, comparator and case labels")
    scopes = [*cases, _AGGREGATE]
    _fields(plan["threshold_ms"], scopes)
    for value in plan["threshold_ms"].values():
        _number(value, "minimum practical gain")
    bootstrap = plan["bootstrap"]
    _fields(bootstrap, ("seed", "draws", "family_alpha"))
    seed, draws, alpha = bootstrap["seed"], bootstrap["draws"], bootstrap["family_alpha"]
    if (type(seed) is not int or not 0 <= seed < 2**64 or type(draws) is not int or not 2000 <= draws <= 100000
            or not 0 < _number(alpha, "bootstrap family alpha") < 1):
        raise ValueError("invalid fixed paired-bootstrap settings")
    arms = {candidate, *references}
    if type(result) is not dict or result.keys() - arms:
        raise ValueError("unregistered comparison arm")
    statuses = {arm: _samples(result.get(arm), blocks, cases) for arm in (candidate, *references)}
    count = len(scopes)*len(references)
    tail = alpha/(2*count)
    rng = Random(seed)
    resamples = [tuple(rng.randrange(len(blocks)) for _ in blocks) for _ in range(draws)]
    contrasts = {}
    candidate_status, candidate_values = statuses[candidate]
    for reference in references:
        reference_status, reference_values = statuses[reference]
        missing = [block for block in blocks if block not in candidate_values or block not in reference_values]
        complete = candidate_status == reference_status == "feasible" and not missing
        contrast = {"candidate_status": candidate_status, "comparator_status": reference_status,
                    "missing_block_ids": missing, "complete": complete, "scopes": {}}
        for scope in scopes:
            deltas = [{"block_id": block, "candidate_ms": candidate_values[block][scope],
                       "reference_ms": reference_values[block][scope],
                       "gain_ms": reference_values[block][scope] - candidate_values[block][scope]}
                      for block in blocks if block in candidate_values and block in reference_values]
            threshold = plan["threshold_ms"][scope]
            row = {"blocks": deltas, "preregistered_minimum_gain_ms": threshold,
                   "median_gain_ms": None, "median_relative_gain_pct": None,
                   "paired_bootstrap_interval_ms": None, "all_blocks_positive": None,
                   "median_above_minimum": None, "interval_above_minimum": None}
            if complete:
                values = [block["gain_ms"] for block in deltas]
                center = median(values)
                estimates = sorted(median(values[index] for index in sample) for sample in resamples)
                interval = [_quantile(estimates, tail), _quantile(estimates, 1-tail)]
                row.update(median_gain_ms=center,
                           median_relative_gain_pct=median(100*block["gain_ms"]/block["reference_ms"] for block in deltas),
                           paired_bootstrap_interval_ms=interval, all_blocks_positive=all(value > 0 for value in values),
                           median_above_minimum=center > threshold, interval_above_minimum=interval[0] > threshold)
            contrast["scopes"][scope] = row
        contrasts[reference] = contrast
    return {"schema": 1, "kind": "offline_paired_latency_analysis", "plan_sha256": plan_sha,
            "paired_result_sha256": fingerprint(result), "candidate": candidate, "block_ids": blocks,
            "bootstrap": {**bootstrap, "method": "whole_block_percentile_bootstrap_paired_median_gain",
                          "multiplicity": "bonferroni_for_all_prespecified_case_and_aggregate_contrasts",
                          "contrast_count": count, "marginal_interval_confidence": 1-2*tail,
                          "approximate": True}, "comparators": contrasts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--paired-result", type=Path, required=True, help="settled paired result, or its result mapping")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    paired = json.loads(args.paired_result.read_text(encoding="utf-8"))
    report = summarize_paired(plan, paired.get("result", paired), expected_plan_sha256=args.expected_plan_sha256)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
