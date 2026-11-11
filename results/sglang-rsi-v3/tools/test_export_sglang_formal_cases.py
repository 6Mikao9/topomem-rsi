from pathlib import Path
import json
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import export_sglang_formal_cases as tool

BASE=Path(r"D:\ws\.scratch")
SOURCE=BASE/"topomem-sglang-contract-v3-draft"
RELEASE=BASE/"topomem-sglang-contract-release-v3/contract-v3-rc.source.zip"
TRAIN=BASE/"topomem-sglang-resident-release-v2/local-formal-cases/archive-train-reproduction"
sys.path.insert(0,str(SOURCE))
from integrations.topology_memory import sglang_ingress_gate as ingress,sglang_serving as serving
from rsi_memory.memory_program import MemoryScenario
from topomem_sglang_rsi.contract import CASE_HASHES
from topomem_sglang_rsi.rebind_cases import ORIGINAL_CASE_SET,INDICES,FORMAL_KIND

class FourCaseExportTest(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.formal=self.root/"formal"
        (self.formal/"controller").mkdir(parents=True)
        self.archive=self.root/"mock-original-six-cases.zip"
        self.unused={"archive-train-mock-omitted","archive-validation-mock-omitted"}
        self.output=self.root/"publish"

    def write_case_archive(self,private_locator=False):
        data={}
        tokens={}
        splits={split:list(ids) for split,ids in tool.CASE_IDS.items()}
        splits["train"].append("archive-train-mock-omitted")
        splits["validation"].append("archive-validation-mock-omitted")
        desc={split:[] for split in splits}
        target=json.loads((TRAIN/"target.json").read_bytes())
        prior_scenario=json.loads((TRAIN/"scenario.json").read_bytes())
        prior_case=json.loads((TRAIN/"manifest.json").read_bytes())
        for split,ids in splits.items():
            for case in ids:
                fixture={"schema":1,"name":case,**{group:[{"rid":f"{case}-{group}-r{rank}",
                        "arrival_ms":0,"max_new_tokens":64,"routed_dp_rank":rank,
                        "prompt":f"Mock authored independent text {case} {group} {rank}."}
                        for rank in range(2)] for group in ("warmup","requests")}}
                if private_locator and case==tool.CASE_IDS["test"][0]:
                    fixture["requests"][0]["prompt"]="Mock heldout prompt includes private mailbox example@example.com"
                tokens[case]={r["rid"]:{"input_tokens":5000,"input_ids_sha256":tool.canonical(r["prompt"])}
                    for group in ("warmup","requests") for r in fixture[group]}
                schedule=serving.schedule_inputs(fixture,2,prior_scenario["attention"]["selector"],attention_dp_size=2,
                                                hot_slots=target["pool"]["hot_slots"])
                gated=ingress.bind_schedule(schedule,tokens[case])
                scenario=MemoryScenario.from_dict({**prior_scenario,"workload_sha256":tool.canonical(fixture),
                                                   "schedule_sha256":tool.canonical(gated)})
                manifest={**prior_case,"workload_sha256":scenario.workload_sha256,"schedule_sha256":scenario.schedule_sha256,
                          "scenario_sha256":scenario.fingerprint()}
                new={"external_schedule.json":tool.json_bytes(gated),"fixture.json":tool.json_bytes(fixture),
                     "scenario.json":tool.json_bytes(scenario.to_dict()),"manifest.json":tool.json_bytes(manifest)}
                for name in (tool.CASE_FILES-set(new))|set(manifest["link_evidence"].values()):
                    new[name]=(TRAIN/name).read_bytes()
                desc[split].append({"id":case,"manifest_sha256":tool.digest(new["manifest.json"]),**{k:manifest[k] for k in CASE_HASHES}})
                data.update({case+"/"+name:raw for name,raw in new.items()})
        # Preserve noncanonical JSON byte formatting in an original member to catch reserialization.
        parent=tool.json_bytes({"prior_case_set_manifest_sha256":ORIGINAL_CASE_SET,"cases":desc})
        formal={split:[desc[split][i] for i in indices] for split,indices in INDICES.items()}
        formal_bytes=tool.json_bytes({"kind":FORMAL_KIND,"cases":formal,"preregistered_zero_based_indices":INDICES,
                  "prior_case_set_manifest_sha256":ORIGINAL_CASE_SET,"parent_case_set_manifest_sha256":tool.digest(parent)})
        selected={case:tokens[case] for ids in tool.CASE_IDS.values() for case in ids}
        receipt=tool.json_bytes({"status":"tokenized","case_set_manifest_sha256":tool.digest(formal_bytes),"tokens_by_case":selected})
        data.update(dict(zip(tool.ROOTS,(parent,formal_bytes,receipt),strict=True)))
        data["tokenization_receipt.json"]=tool.json_bytes({"tokens_by_case":tokens})
        with ZipFile(RELEASE) as package:
            source=json.loads(package.read("LOCAL_SOURCE_MANIFEST.json"))
        worker=source["worker_source_manifest_sha256"]
        manifest_raw=tool.json_bytes({"worker_source_manifest_sha256":worker,"formal_case_set_sha256":tool.digest(formal_bytes),
            "formal_tokenization_receipt_sha256":tool.digest(receipt),"files":{k:tool.digest(v) for k,v in data.items()}})
        self.members=data
        self.upstream=manifest_raw
        self.write_zip()
        self.fixed={**tool.FIXED,"original_case_archive_sha256":tool.digest(self.archive.read_bytes()),
                   "original_case_transfer_manifest_sha256":tool.digest(manifest_raw),"formal_case_set_manifest_sha256":tool.digest(formal_bytes),
                   "parent_case_set_manifest_sha256":tool.digest(parent),"formal_tokenization_receipt_sha256":tool.digest(receipt)}
        sources=source["files"]
        manifest={"config":{"rounds":2},"source_sha256":{"topology_rsi.py":sources["rsi_memory/topology_rsi.py"]},
                "contract":{"cases":formal,"source_manifest_sha256":worker,"tokenization_sha256":tool.canonical(selected),
                            "adapter_sources_sha256":{"topomem_sglang_rsi/domain.py":sources["topomem_sglang_rsi/domain.py"]},
                            "generator_program_protocol":{"kind":"backend_program_contract_v3",
                              "source_sha256":{"topomem_sglang_rsi/contract_v3.py":sources["topomem_sglang_rsi/contract_v3.py"]}}}}
        rounds=[{"round":0},{"round":1}]
        frozen={"partial":False,"rounds":2,"manifest_sha256":tool.canonical(manifest),"rounds_sha256":tool.canonical(rounds)}
        report={"status":"test_settled","manifest":manifest,"frozen":frozen,"rounds":rounds,"paired_evaluations":{
            "pair/frozen/test":{"status":"completed","input":{"split":"test"}}}}
        self.formal_files={"controller/report.json":tool.json_bytes(report),"controller/frozen.json":tool.json_bytes(frozen),
                           "pipeline-status.json":tool.json_bytes([{"phase":s,"returncode":0} for s in ("run","freeze","test")]),
                           "frozen-source.zip":RELEASE.read_bytes()}
        self.write_formal()

    def write_zip(self):
        with ZipFile(self.archive,"w") as package:
            package.writestr("CASE_TRANSFER_MANIFEST.json",self.upstream)
            for name,raw in self.members.items():
                package.writestr(name,raw)

    def write_formal(self):
        for name,raw in self.formal_files.items():
            (self.formal/name).write_bytes(raw)
        (self.formal/"artifact_manifest.json").write_bytes(tool.json_bytes({
            "files_sha256":{name:tool.digest(raw) for name,raw in self.formal_files.items()}}))

    def args(self):
        return SimpleNamespace(formal_export=self.formal,original_case_archive=self.archive,output=self.output,local_tokenizer_model=None,
            expected_formal_report_sha256=tool.digest(self.formal_files["controller/report.json"]),
            expected_pipeline_status_sha256=tool.digest(self.formal_files["pipeline-status.json"]))

    def test_exact_four_subset_loaders_and_no_omitted_body_reads(self):
        self.write_case_archive()
        orig=ZipFile.read
        def tracked(package,name,*args,**kwargs):
            if isinstance(name,str):
                self.assertNotIn(name.split("/")[0],self.unused)
            return orig(package,name,*args,**kwargs)
        with patch.object(ZipFile,"read",tracked):
            result=tool.export(self.args(),self.fixed)
        self.assertTrue(result["verification"]["frozen_case_and_schedule_loaders_verified"])
        self.assertFalse(result["verification"]["retokenized_locally"])
        self.assertEqual(result["frozen_source_file_count"],131)
        self.assertEqual(len(result["published_zip_member_sha256"]),43)
        self.assertNotEqual(result["new_public_four_case_zip_sha256"],result["original_case_archive_sha256"])
        self.assertEqual((self.output/"provenance/CASE_TRANSFER_MANIFEST.json").read_bytes(),self.upstream)
        with ZipFile(self.output/"formal-four-cases.zip") as package:
            for name in package.namelist():
                self.assertEqual(package.read(name),self.members[name])
                self.assertEqual((self.output/"cases"/name).read_bytes(),self.members[name])
        self.assertFalse((self.output/"cases/tokenization_receipt.json").exists())

    def test_terminal_seal_or_nonzero_phase_fail_before_original_archive_read(self):
        self.write_case_archive()
        with patch.object(tool,"original_cases",side_effect=AssertionError("original archive opened before terminal")):
            args=self.args()
            args.expected_formal_report_sha256="0"*64
            with self.assertRaisesRegex(ValueError,"SHA-256"):
                tool.export(args,self.fixed)
            self.formal_files["pipeline-status.json"]=tool.json_bytes([{"phase":"run","returncode":1}])
            self.write_formal()
            with self.assertRaisesRegex(ValueError,"all exit"):
                tool.export(self.args(),self.fixed)
        self.assertFalse(self.output.exists())

    def test_changed_member_refused_even_if_outer_archive_is_repinned(self):
        self.write_case_archive()
        name=tool.CASE_IDS["test"][0]+"/fixture.json"
        self.members[name]+=b" "
        self.write_zip()
        self.fixed["original_case_archive_sha256"]=tool.digest(self.archive.read_bytes())
        with self.assertRaisesRegex(ValueError,"case member hash differs"):
            tool.export(self.args(),self.fixed)
        self.assertFalse(self.output.exists())

    def test_private_locator_match_stops_and_preserves_original(self):
        self.write_case_archive(private_locator=True)
        archive_before=self.archive.read_bytes()
        with self.assertRaisesRegex(ValueError,"private-locator review"):
            tool.export(self.args(),self.fixed)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.archive.read_bytes(),archive_before)

if __name__=="__main__":
    unittest.main()
