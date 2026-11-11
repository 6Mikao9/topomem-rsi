"""CPU diagnostic for the allocation-replica decision, including timeline append."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
from statistics import median
import sys
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))
from test_producer_retain import LRU_RETAIN, NEW, NEW_RAW, OLD, OLD_RAW, runtime, state


def sample(engine, block, iterations):
    engine.timeline.clear()
    start = perf_counter_ns()
    for _ in range(iterations):
        engine._optional("allocate", block, retain_source=True)
    elapsed = perf_counter_ns() - start
    return elapsed / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20000)
    args = parser.parse_args()
    if args.iterations < 1000:
        parser.error("at least 1000 iterations required")
    engines = [runtime(module, LRU_RETAIN) for module in (OLD, NEW)]
    block = state()
    for engine in engines:
        sample(engine, block, 2000)
    rows = []
    for index in range(12):
        order = (0, 1) if index % 2 == 0 else (1, 0)
        times = [None, None]
        for arm in order:
            times[arm] = sample(engines[arm], block, args.iterations)
        assert engines[0].timeline == engines[1].timeline and len(engines[0].timeline) == args.iterations
        rows.append({"block": index, "order": ["v11" if arm == 0 else "v13" for arm in order],
                     "v11_ns_per_optional": times[0], "v13_ns_per_optional": times[1],
                     "old_minus_new_ns_per_optional": times[0]-times[1]})
    result = {"schema": 1, "scope": "CPU-only _optional on modeled producer-owned copies with timeline append",
              "python": platform.python_version(), "host": platform.system(), "iterations": args.iterations, "blocks": rows,
              "source_sha256": {"v11": sha256(OLD_RAW).hexdigest(), "v13": sha256(NEW_RAW).hexdigest()},
              "median_v11_ns": median(row["v11_ns_per_optional"] for row in rows),
              "median_v13_ns": median(row["v13_ns_per_optional"] for row in rows),
              "paired_median_gain_ns": median(row["old_minus_new_ns_per_optional"] for row in rows)}
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print({key: result[key] for key in ("median_v11_ns", "median_v13_ns", "paired_median_gain_ns")})


if __name__ == "__main__":
    main()
