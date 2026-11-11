"""Exercise v4.1 controller replay and audit on synthetic external effects."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

BASE=Path(r"D:\ws\.scratch")
SOURCE=BASE/"topomem-v41-controller-source"
PUBLIC=Path(r"D:\ws\01_研究\实验环境\topomem-rsi\results\prompt-v4-abort")
sys.path[:0]=[str(BASE),str(SOURCE),str(SOURCE/"tests")]
import export_topomem_v41_gpu3 as audit
from rsi_memory import cuda_v4
from rsi_memory.cuda_domain import CudaDomain
from rsi_memory.topology_rsi_v4 import TopologyRSIV4
from test_topology_rsi_v4 import PLAN,program,rule,source,value


def request(payload):
    envelope=json.loads(payload["messages"][1]["content"])
    decision=envelope["generator_decision"]
    if "mode" in decision:
        number=envelope["round"]
        content={"source":source(7-2*number,("repair","rethink","revise")[number]),"plan":PLAN}
    else:
        guide=decision["guidance"]
        item=int(guide) if guide.isdigit() else 10
        content=program(item) if decision["operator"]=="construct" else {"rule":rule(item)}
    return {"choices":[{"message":{"content":json.dumps(content)}}],
            "usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}


class ExportReplayTests(unittest.TestCase):
    def test_real_frozen_engineering_qualification_binding(self):
        publication=PUBLIC.with_name("engineering-producer-v13")
        root=publication/"run"
        plan=json.loads((publication/"inputs/preregistration.json").read_bytes())
        config={"initial_program":plan["arms"]["lru-equivalent-v13"]["program"],
                "cases":{"train":plan["payload"]["cases"]},
                "executor":{"gpu_indices":[3],"gpu_uuids":plan["payload"]["expected_gpu_uuids"]}}
        copied={}
        with tempfile.TemporaryDirectory(prefix="topomem-v41-qualification-test-") as directory:
            run=Path(directory)
            (run/"engineering-qualification.json").write_bytes((root/"analysis.json").read_bytes())
            ids=audit.verify_engineering_qualification(run,publication,config,{
                "qualified_engineering_plan_sha256":audit.ENGINEERING_PLAN,
                "qualified_engineering_result_sha256":audit.ENGINEERING_RESULT},copied)
        self.assertEqual(len(set(ids)),24)
        self.assertEqual(audit.digest(copied["qualification/engineering-analysis.json"]),audit.ENGINEERING_RESULT)
        self.assertIn("qualification/engineering-jobs.jsonl",copied)

    def test_v41_offline_full_ledger_roles_training_and_test(self):
        config=json.loads((PUBLIC/"inputs/adapter-config.template.json").read_bytes())
        config.update(experiment_id="mock-v41-export",initial_program=program(12),manual_program=program(8),
                      output_directory="mock-adapter")
        pre=config["controller_preregistration"]
        pre["rounds"]=2
        raw_jobs={}
        paired={}
        def evaluate(split,candidate,key):
            job="eval-"+audit.canon(key)[:48]
            score=value(candidate)+0.1
            raw_jobs[job]={"split":split,"effect":key,"score_ms":score,
                           "cases":{case["name"]:{"median_ns":round(score*1e6/4)} for case in config["cases"][split]}}
            return {"feasible":True,"score":score,"metrics":{"receipts":[job]},"feedback":{}}

        def paired_eval(split,variants,key):
            blocks=config["paired_block_ids"][split]
            order=audit.paired_orders(variants,key,blocks)
            observations=[]
            paired[key]=observations
            roles={}
            for label,candidate in variants.items():
                total=[]
                for i,block in enumerate(blocks):
                    total.append({"block_id":block,"score":value(candidate)+i/10})
                roles[label]={"feasible":True,"samples":total,"metrics":{}}
            for i,block in enumerate(blocks):
                observations.append({"block_id":block,"order":order[block],"arms":{
                    label:{"feasible":True,"cases":{case["name"]:{
                        "median_ns":round(roles[label]["samples"][i]["score"]*1e6/4),
                        "managed_peak_bytes_by_node":{"gpu0":64,"gpu1":128,"host0":512},
                        "cuda_peak_allocated_delta_bytes_by_node":{"gpu0":1024,"gpu1":2048},
                        "link_bytes":{"fake-h2d":64},"physical_sha256":"a"*64}
                        for case in config["cases"][split]}} for label in variants}})
            return roles

        with tempfile.TemporaryDirectory(prefix="topomem-v41-export-test-") as tmp:
            root=Path(tmp)
            adapter=CudaDomain(config,root)
            domain=cuda_v4.contract_domain(adapter.domain(),pre)
            domain=replace(domain,evaluate=evaluate,evaluate_pair=paired_eval)
            controller=TopologyRSIV4(root/"original",cuda_v4.checked_preregistration(pre,config["paired_block_ids"]),
                                     domain,request,backend_source=Path(cuda_v4.__file__))
            try:
                self.assertEqual(controller.run()["status"],"rounds_completed")
                frozen=controller.freeze()
                self.assertTrue(all(role in frozen for role in audit.ARMS))
                report=controller.test()
                records={k:json.loads(v) for k,v in controller.db.execute("SELECT key,data FROM records")}
            finally:
                controller.close()
                adapter.close()
            originals={f"controller/{p.relative_to(root/'original').as_posix()}":p.read_bytes()
                       for sub in ("harnesses","rounds") for p in (root/"original"/sub).iterdir()}
            audit.audit_provider_envelopes(report,config)
            _,effects=audit.replay_controller(SOURCE,config,records,originals,report,root)
            self.assertGreater(effects,20)
            rounds=audit.audit_training(report,records,raw_jobs)
            self.assertEqual(len(rounds),2)
            score=audit.test_summary(report,records,paired,config)
            self.assertEqual(len(score["block_ids"]),6)
            self.assertEqual(set(score["scenarios"]),set(audit.SCENARIOS))
            self.assertIn("managed_peak_bytes_by_node_max",score["scenarios"]["agent_tools"]["arms"]["manager_best"])

if __name__=="__main__":
    unittest.main()
