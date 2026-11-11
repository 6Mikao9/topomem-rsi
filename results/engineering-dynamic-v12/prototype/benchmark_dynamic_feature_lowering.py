"""CPU-only paired benchmark of real runtime route selection on Python metadata."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import platform
from statistics import median
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tests"))
from test_dynamic_feature_lowering import REPO, block, guarded_program, new, old, runtime, thin_program


def sample(engine, state, iterations):
    start = perf_counter_ns()
    for _ in range(iterations):
        selected = engine._choose_place(state)
    elapsed = perf_counter_ns() - start
    return elapsed / iterations, selected


def run(iterations, blocks):
    configurations = (
        ("one_dynamic_feature", thin_program, 3),
        ("multiple_rules_hot_first_rule", guarded_program, 3),
        ("multiple_rules_cold_fallback", guarded_program, 8),
    )
    result = {}
    for name, make_program, valid in configurations:
        program = make_program()
        engines = [(runtime(version, program, (True,True,True)), block(version,valid=valid))
                   for version in (old, new)]
        expected = engines[0][0]._choose_place(engines[0][1])
        assert expected is not None
        for engine, state in engines:
            _, observed = sample(engine, state, min(1000, iterations))
            assert observed == expected
        trials = []
        for number in range(blocks):
            order = (0,1) if number % 2 == 0 else (1,0)
            times = [None,None]
            for index in order:
                times[index], observed = sample(*engines[index], iterations)
                assert observed == expected
            trials.append({"block":number+1,"order":["v11" if index==0 else "v12" for index in order],
                           "v11_ns_per_choice":times[0],"v12_ns_per_choice":times[1]})
        before = median(row["v11_ns_per_choice"] for row in trials)
        after = median(row["v12_ns_per_choice"] for row in trials)
        result[name] = {"trials":trials, "median_v11_ns_per_choice":before, "median_v12_ns_per_choice":after,
                        "paired_median_gain_ns_per_choice":median(row["v11_ns_per_choice"]-row["v12_ns_per_choice"] for row in trials)}
    paths = (REPO / "integrations/topology_memory/torch_runtime.py", ROOT / "integrations/topology_memory/torch_runtime.py",
             ROOT / "integrations/topology_memory/torch_dynamic_features.py")
    hashes = {name:sha256(path.read_bytes()).hexdigest() for name,path in zip(("v11_runtime","v12_runtime","v12_feature_view"),paths)}
    return {"schema":1,"scope":"CPU route choice only, no Torch tensor ops, CUDA, serving, or RSI",
            "python":platform.python_version(),"system":platform.platform(),"iterations_per_arm_block":iterations,
            "paired_blocks":blocks,"source_sha256":hashes,"cases":result}


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--iterations",type=int,default=10000)
    parser.add_argument("--blocks",type=int,default=12)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.iterations < 1 or args.blocks < 2 or args.blocks % 2:
        parser.error("positive iterations and an even number of paired blocks required")
    result=run(args.iterations,args.blocks)
    with args.output.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")
    for name,row in result["cases"].items():
        print(name,row["median_v11_ns_per_choice"],row["median_v12_ns_per_choice"],row["paired_median_gain_ns_per_choice"])


if __name__=="__main__":
    main()
