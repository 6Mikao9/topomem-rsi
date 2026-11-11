import json
from pathlib import Path
import shutil
import sys
import tempfile

scratch = Path(r"D:\ws\.scratch")
repo = Path(r"D:\ws\01_研究\实验环境\topomem-rsi")
release = scratch / "topomem-sglang-contract-release-v3/contract-v3-rc.source.zip"
source = scratch / "topomem-sglang-contract-v3-draft"
public_abort = repo / "results/sglang-contract-v2-abort"
sys.path[:0] = [str(scratch), str(source)]
from export_sglang_controller_results import export, digest, fingerprint
from rsi_memory.memory_program import MemoryProgram
from rsi_memory.topology_rsi import Domain, Config
from rsi_memory.topology_rsi_contract import ContractTopologyRSI
from topomem_sglang_rsi import contract_v3

def read(path):
    return json.loads(path.read_bytes())

def save(path,value):
    path.write_text(json.dumps(value,sort_keys=True,ensure_ascii=False),encoding="utf-8")

archive_config = read(public_abort/"schema3.template.json")
contract = read(public_abort/"adapter_contract.json")
archive_config["experiment_id"] = contract["experiment_id"] = "synthetic-export-check"
archive_config["receipt_directory"] = "receipts"
archive_config["executor"].update(host="synthetic.invalid", worker="/fake-x/worker", python="/fake-x/python", root="/fake-x/root", gpu_uuids=["synthetic-device-a","synthetic-device-b"])
archive_config["local_cases"] = "synthetic-case-root"
archive_config["manual_selection"].update(sweep_dir="synthetic-hand-root",source_config="synthetic-schema2.json")
original_program = MemoryProgram.from_dict(archive_config["initial_program"])
new_program = MemoryProgram.from_dict(archive_config["manual_program"])
assert original_program.fingerprint() != new_program.fingerprint()
context = {"cases":[{"case":case["id"]} for case in contract["cases"]["train"]]}
tests = [case["id"] for case in contract["cases"]["test"]]

def score(program):
    if program is None:
        return 1000
    return 980 if program.fingerprint() == new_program.fingerprint() else 990

def evaluate(split, program, key):
    return {"feasible":True,"score":score(program),"metrics":{},"feedback":{}}

def evaluate_pair(split, programs, key):
    cases = [case["id"] for case in contract["cases"][split]]
    return {label:{"feasible":True,"metrics":{},"samples":[
        {"block_id":block,"score":score(program)*len(cases),"case_scores_ms":{case:score(program) for case in cases}}
        for block in contract["paired_block_ids"][split]]} for label,program in programs.items()}

def fake_provider(payload):
    envelope = json.loads(payload["messages"][-1]["content"])
    if "trusted_revision_instruction" in envelope:
        value = {"source":envelope["harness_source"],"plan":{
            "observed_problem":"Synthetic test only.","mechanism_change":"Preserve seed.",
            "expected_effect":"Exercise both branches.","falsification":"Reject source unchanged."}}
    elif "rule" in envelope["response_shape"]:
        parent = next(iter(envelope["selected_parents"].values()))["program"]
        value = {"rule":parent["rules"][envelope["generator_decision"]["rule"]]}
    else:
        value = new_program.to_dict()
    return {"choices":[{"message":{"content":json.dumps(value)},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}

with tempfile.TemporaryDirectory(prefix="topomem-export-check-") as temp:
    base = Path(temp)
    study = base/"study"
    study.mkdir()
    (study/"receipts").mkdir()
    private=study/"schema3.json"
    save(private,archive_config)
    controls=contract_v3.contract_domain(Domain(contract,context,(("allocate","place"),("configure","hot_capacity"),("configure","backing_mode")),
        evaluate, original_program, evaluate_pair, new_program, lambda p:p.fingerprint()))
    ctl=ContractTopologyRSI(study/"controller",Config(rounds=2,slots=2,min_improvement=11.49,min_paired_blocks=6),
                           controls,fake_provider,backend_source=Path(contract_v3.__file__))
    try:
        assert ctl.run()["status"] == "rounds_completed"
        assert ctl.freeze()["rounds"] == 2
        assert ctl.test()["status"] == "test_settled"
    finally:
        ctl.close()
    report=study/"controller/report.json"
    report_bytes=report.read_bytes()
    try:
        export(study,private,release,None,base/"should-not-exist","0"*64,False)
    except ValueError as error:
        assert "terminal marker" in str(error)
        assert not (base/"should-not-exist").exists()
    else:
        raise AssertionError("wrong authorization digest was accepted")
    export(study,private,release,None,base/"export",digest(report_bytes),False)
    output=base/"export"
    assert (output/"controller/journal.sqlite").read_bytes()==(study/"controller/journal.sqlite").read_bytes()
    exported=read(output/"export_audit.json")
    assert exported["receipt_count"]==0 and exported["verified_controller_source_bindings"]==15
    assert exported["heldout_comparisons"]["adaptive_vs_native"]["passed"]
    assert not exported["heldout_comparisons"]["adaptive_vs_manual"]["passed"]
    assert len(exported["heldout_comparisons"]["adaptive_vs_native"]["deltas"])==12
    choices=read(output/"selected/selection.json")["finalists"]
    for arm, meta in choices.items():
        assert fingerprint(read(output/f"selected/{arm}.program.json"))==meta["program_sha256"]
        assert digest((output/"selected"/meta["generator_filename"]).read_bytes())==meta["harness_sha256"]
    assert not (output/"schema3.json").exists()
    for name,h in read(output/"artifact_manifest.json")["files_sha256"].items():
        assert digest((output/name).read_bytes())==h
print("Two rounds, full freeze, 12 synthetic heldout blocks, finalists/provenance, and terminal SHA refusal passed.")

