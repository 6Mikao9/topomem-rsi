from copy import deepcopy
from types import SimpleNamespace
import unittest

from integrations.topology_memory.sglang_global_ticks import prepared_tick, scheduler_tick, verify_global_ticks
from integrations.topology_memory.sglang_ingress_gate import EXTERNAL_FIELD, bind_schedule
from test_sglang_ingress_gate import inputs


def row(rank, iteration, phase, counts, *, extend=False, global_mode=None):
    count = counts[rank]
    mode = ("EXTEND" if extend else "DECODE") if count else "IDLE"
    padding = "SUM_LEN" if extend else "MAX_LEN"
    padded = counts if extend else [max(counts)] * 2
    return {"rank": rank, "forward_iter": iteration, "mode": mode, "forward_mode": mode,
            "global_forward_mode": global_mode, "global_num_tokens": counts,
            "global_num_tokens_for_logprob": [int(x != 0) for x in counts], "dp_padding_mode": padding,
            "padded_global_num_tokens": padded, "global_dp_buffer_len": sum(padded),
            "rids": [f"g{phase}r{rank}"] if count else [], "seq_lens": [count] if count else [],
            "extend_lens": [count] if count and extend else [], "prefix_lens": [0] if count and extend else []}


def no_forward(rank):
    return {"rank": rank, "forward_iter": None, "mode": "NO_FORWARD", "forward_mode": None,
            "global_forward_mode": None, "global_num_tokens": [0, 0], "global_num_tokens_for_logprob": [0, 0],
            "dp_padding_mode": None, "padded_global_num_tokens": [0, 0], "global_dp_buffer_len": 0,
            "rids": [], "seq_lens": [], "extend_lens": [], "prefix_lens": []}


class GlobalTicksTest(unittest.TestCase):
    def setUp(self):
        self.contract = bind_schedule(*inputs())[EXTERNAL_FIELD]
        self.rows = [[row(rank, 1, 0, [2, 2], extend=True, global_mode="EXTEND"),
                      row(rank, 2, 0, [1, 0], global_mode="DECODE"),
                      row(rank, 3, 1, [2, 3], extend=True, global_mode="EXTEND"),
                      row(rank, 4, 1, [1, 1], global_mode="DECODE")] for rank in (0, 1)]

    def test_only_globally_empty_poll_is_normalized(self):
        initial = verify_global_ticks(self.rows, self.contract)
        self.assertEqual(initial["active_forward_count"], 4)
        self.assertEqual(initial["rank_idle_forward_count"], 1)
        noise = [[no_forward(rank), *rows[:2], no_forward(rank), *rows[2:]] for rank, rows in enumerate(self.rows)]
        self.assertEqual(verify_global_ticks(noise, self.contract, expected_sha256=initial["global_ticks_sha256"]), initial)
        noise[1][2] = no_forward(1)
        with self.assertRaisesRegex(ValueError, "iteration"):
            verify_global_ticks(noise, self.contract)

    def test_schedule_shape_changes_fail_even_when_rids_are_unchanged(self):
        reference = verify_global_ticks(self.rows, self.contract)
        changed = deepcopy(self.rows)
        for rank in (0, 1):
            changed[rank][2] = row(rank, 3, 1, [3, 3], extend=True, global_mode="EXTEND")
        with self.assertRaisesRegex(ValueError, "frozen native reference"):
            verify_global_ticks(changed, self.contract, expected_sha256=reference["global_ticks_sha256"])

    def test_staggered_first_prefill_is_rejected(self):
        for rank in (0, 1):
            self.rows[rank][2] = row(rank, 3, 1, [0, 3], extend=True, global_mode="EXTEND")
        with self.assertRaisesRegex(ValueError, "same global prefill tick"):
            verify_global_ticks(self.rows, self.contract)

    def test_lost_idle_rank_or_missing_actual_padding_is_rejected(self):
        self.rows[1][1]["dp_padding_mode"] = None
        with self.assertRaisesRegex(ValueError, "actual DP padding mode"):
            verify_global_ticks(self.rows, self.contract)
        self.rows[1].pop(1)
        with self.assertRaisesRegex(ValueError, "iteration"):
            verify_global_ticks(self.rows, self.contract)

    def test_projection_joins_original_and_actual_cpu_padding_without_touching_cuda(self):
        mode = SimpleNamespace(name="EXTEND")
        rows = SimpleNamespace(device=SimpleNamespace(type="cpu"), tolist=lambda: [5])
        batch = SimpleNamespace(forward_mode=mode, global_forward_mode=mode, seq_lens_cpu=rows,
                                reqs=[SimpleNamespace(rid="g0r0")], extend_lens=[2], prefix_lens=[3],
                                global_num_tokens=[2, 3], global_num_tokens_for_logprob=[1, 1])
        before = scheduler_tick(0, 1, batch)
        live = SimpleNamespace(original_global_num_tokens_cpu=[2, 3], global_num_tokens_for_logprob_cpu=[1, 1],
                               forward_mode=mode, dp_padding_mode=SimpleNamespace(name="MAX_LEN"),
                               global_num_tokens_cpu=[3, 3], global_dp_buffer_len=6)
        proof = prepared_tick(before, live)
        self.assertEqual(proof["padded_global_num_tokens"], [3, 3])
        self.assertEqual(proof["prefix_lens"], [3])
        live.original_global_num_tokens_cpu = [0, 3]
        with self.assertRaisesRegex(ValueError, "original global token vectors"):
            prepared_tick(before, live)


if __name__ == "__main__":
    unittest.main()
