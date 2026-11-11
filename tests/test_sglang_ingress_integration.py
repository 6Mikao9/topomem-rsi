from pathlib import Path
import json
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from integrations.topology_memory import sglang_serving as serving
from integrations.topology_memory.sglang_ingress_gate import bind_schedule
from integrations.topology_memory.sglang_observe import _install_global_observer


class IntegrationTest(unittest.TestCase):
    def test_live_tokenizer_commitments_are_rebuilt_and_reused(self):
        fixture = {"schema": 1, "name": "mock-two-groups"}
        for phase in ("warmup", "requests"):
            fixture[phase] = [{"rid": f"{phase}-{rank}", "routed_dp_rank": rank, "arrival_ms": 0,
                               "max_new_tokens": 4, "prompt": str(rank)} for rank in (0, 1)]
        base = serving.schedule_inputs(serving.validate_fixture(fixture), 2, "torch-unfused", attention_dp_size=2, hot_slots=6144)
        class Tokenizer:
            def encode(self, prompt, add_special_tokens):
                if add_special_tokens:
                    raise ValueError("unexpected special tokens")
                return [int(prompt), 10]
        ids = serving._tokenize(fixture, Tokenizer())
        expected = bind_schedule(base, serving._token_commitments(ids))
        self.assertEqual(serving._bound_external(base, expected, ids), expected)
        ids["requests-0"][1] = 11
        with self.assertRaisesRegex(ValueError, "live tokenizer"):
            serving._bound_external(base, expected, ids)

    def test_performance_requires_same_global_schedule_but_parity_keeps_full_kv(self):
        values = {"effective_schedule_sha256": "sched", "dispatch_sha256": "dispatch", "output_ids_sha256": "output",
                  "global_ticks_sha256": "global", "packed_kv_sha256_by_rank": ["rank0", "rank1"]}
        serving._compare(values, {**values, "packed_kv_sha256_by_rank": []}, parity=False)
        with self.assertRaisesRegex(ValueError, "actual dispatch"):
            serving._compare(values, {**values, "global_ticks_sha256": "different"}, parity=False)
        with self.assertRaisesRegex(ValueError, "selected KV"):
            serving._compare(values, {**values, "packed_kv_sha256_by_rank": ["bad", "rank1"]}, parity=True)

    def test_proof_checks_gated_identity_before_launch(self):
        proof = {key: "bound" for key in ("target_sha256", "scenario_sha256", "topology_sha256", "runtime",
                 "schedule_inputs_sha256", "ingress_contract_sha256", "input_tokenization_sha256", "compiled_plan_sha256")}
        proof.update(status="completed", phase="parity", arm="native", global_ticks_sha256="a"*64)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "proof.json"
            path.write_text(json.dumps(proof))
            self.assertEqual(serving._load_proof(path, proof, same_arm=True), proof)
            with self.assertRaisesRegex(ValueError, "input_tokenization_sha256"):
                serving._load_proof(path, {**proof, "input_tokenization_sha256": "different"}, same_arm=True)
            del proof["global_ticks_sha256"]
            path.write_text(json.dumps(proof))
            with self.assertRaisesRegex(ValueError, "global forward signature"):
                serving._load_proof(path, proof, same_arm=True)

    def test_cpu_observer_keeps_rank_idle_until_both_dispatches_finish(self):
        extend = SimpleNamespace(name="EXTEND")
        idle = SimpleNamespace(name="IDLE")

        class Forward:
            def __init__(self, batch):
                self.forward_mode = batch.forward_mode
                self.original_global_num_tokens_cpu = batch.global_num_tokens
                self.global_num_tokens_for_logprob_cpu = batch.global_num_tokens_for_logprob

            def prepare_mlp_sync_batch(self):
                self.dp_padding_mode = SimpleNamespace(name="SUM_LEN")
                self.global_num_tokens_cpu = self.original_global_num_tokens_cpu
                self.global_dp_buffer_len = sum(self.global_num_tokens_cpu)

        class Scheduler:
            forward_ct = 0

            def run_batch(self, batch):
                self.forward_ct += 1
                batch.forward_iter = self.forward_ct
                Forward(batch).prepare_mlp_sync_batch()
                return "original"

            def get_next_batch_to_run(self, next_batch):
                return SimpleNamespace(batch_to_run=next_batch)

        real = SimpleNamespace(forward_mode=extend, global_forward_mode=extend, global_num_tokens=[5, 5],
                               global_num_tokens_for_logprob=[1, 1], reqs=[SimpleNamespace(rid="warm-0")],
                               seq_lens_cpu=SimpleNamespace(device=SimpleNamespace(type="cpu"), tolist=lambda: [5]),
                               extend_lens=[5], prefix_lens=[0])
        peer_only = SimpleNamespace(forward_mode=idle, global_forward_mode=extend, global_num_tokens=[0, 1],
                                    global_num_tokens_for_logprob=[0, 1], reqs=[])
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _install_global_observer(directory, 0, Scheduler, Forward, lambda: True)
            scheduler = Scheduler()
            self.assertEqual(scheduler.run_batch(real), "original")
            (directory / "dispatch-rank-0.json").touch()
            scheduler.get_next_batch_to_run(None)
            output = directory / "global-ticks-rank-0.json"
            self.assertFalse(output.exists())
            scheduler.run_batch(peer_only)
            (directory / "dispatch-rank-1.json").touch()
            scheduler.get_next_batch_to_run(None)
            records = json.loads(output.read_bytes())["ticks"]
            self.assertEqual([row["mode"] for row in records], ["EXTEND", "IDLE"])
            self.assertEqual([row["forward_iter"] for row in records], [1, 2])


if __name__ == "__main__":
    unittest.main()
