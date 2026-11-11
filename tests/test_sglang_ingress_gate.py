from array import array
from dataclasses import dataclass
from types import SimpleNamespace
import unittest

from integrations.topology_memory.sglang_ingress_gate import (
    EXTERNAL_FIELD, GateFailure, IngressGate, REASONS, _verify_live, _wrap_processor,
    bind_schedule, fingerprint, verify_ingress_events,
)


@dataclass
class Request:
    rid: str
    routed_dp_rank: int
    input_ids: array
    time_stats: object = None


class ReadOnly:
    pass


class Shutdown:
    pass


def inputs():
    engine = {"tp_size": 2, "pp_size": 1, "dp_size": 2, "enable_dp_attention": True,
              "disable_overlap_schedule": True, "disable_cuda_graph": True, "disable_radix_cache": True,
              "enable_hisparse": True}
    groups, commitments = {}, {}
    for phase_index, phase in enumerate(("warmup", "requests")):
        groups[phase] = [{"rid": f"g{phase_index}r{rank}", "routed_dp_rank": rank, "arrival_ms": 0} for rank in (0, 1)]
        for rank in (0, 1):
            commitments[f"g{phase_index}r{rank}"] = {"input_tokens": 2, "input_ids_sha256": fingerprint([phase_index, rank])}
    return {"engine": engine, **groups}, commitments


class GateTest(unittest.TestCase):
    def setUp(self):
        schedule, committed = inputs()
        self.contract = bind_schedule(schedule, committed, gate_timeout_ms=10, collective_timeout_ms=5)[EXTERNAL_FIELD]
        self.events = [[], []]
        self.time = 0
        self.gates = [IngressGate(self.contract, rank, work_type=Request, controls=(ReadOnly,), teardown_controls=(Shutdown,),
                                  now=lambda: self.time, report=self.events[rank].append) for rank in (0, 1)]
        self.requests = [[Request(f"g{phase}r{rank}", rank, array("I", [phase, rank]), object()) for rank in (0, 1)]
                         for phase in (0, 1)]

    def step(self, rank, messages, votes, idle=True):
        def collect(row):
            self.assertEqual(row, votes[rank])
            return votes
        return self.gates[rank].step(messages, lambda: idle, collect)

    def test_schedule_commits_payload_order_and_rejects_extra_tokenization(self):
        schedule, committed = inputs()
        bound = bind_schedule(schedule, committed)
        self.assertNotIn(EXTERNAL_FIELD, schedule)
        self.assertEqual(bound[EXTERNAL_FIELD]["groups"][1]["members"][1]["input_ids_sha256"], fingerprint([1, 1]))
        schedule["requests"][1]["arrival_ms"] = 1
        with self.assertRaisesRegex(ValueError, "simultaneous"):
            bind_schedule(schedule, committed)
        schedule["requests"][1]["arrival_ms"] = 0
        with self.assertRaisesRegex(ValueError, "commitments"):
            bind_schedule(schedule, {**committed, "extra": committed["g1r1"]})

    def test_release_together_and_wait_for_previous_phase_drain(self):
        warm0, warm1 = self.requests[0]
        meas0, meas1 = self.requests[1]
        read_only = ReadOnly()
        vote = ((0, 0, 0, 0), (0, 1, 1, 0))
        self.assertEqual(self.step(0, [], vote), [])
        self.assertEqual(self.step(1, [warm1, read_only], vote), [read_only])
        self.assertEqual([gate.phase for gate in self.gates], [0, 0])
        self.time = 1_000_000
        vote = ((0, 1, 1, 0), (0, 1, 1, 0))
        self.assertIs(self.step(0, [warm0], vote)[0], warm0)
        self.assertIs(self.step(1, [], vote)[0], warm1)
        self.assertEqual(self.events[1][0]["wait_ns"], 1_000_000)
        vote = ((1, 1, 0, 0), (1, 1, 1, 0))
        self.assertEqual(self.step(0, [meas0], vote, idle=False), [])
        self.assertEqual(self.step(1, [meas1], vote), [])
        vote = ((1, 1, 1, 0), (1, 1, 1, 0))
        self.assertIs(self.step(0, [], vote)[0], meas0)
        self.assertIs(self.step(1, [], vote)[0], meas1)
        self.assertEqual([gate.phase for gate in self.gates], [2, 2])
        end = [Shutdown()]
        self.assertIs(self.gates[0].step(end, lambda: self.fail("no post-release idle probe"),
                                        lambda _: self.fail("no post-release collective")), end)
        with self.assertRaisesRegex(GateFailure, "after_final_release"):
            self.gates[0].step([meas0], None, None)

    def test_peer_failure_is_coherent_before_release(self):
        good, wrong = self.requests[0]
        wrong.input_ids = array("I", [77, 77])
        vote = ((0, 1, 1, 0), (0, 0, 0, REASONS.index("wrong_input_tokens")))
        for rank, message in enumerate((good, wrong)):
            with self.assertRaisesRegex(GateFailure, "wrong_input_tokens"):
                self.step(rank, [message], vote)
        self.assertEqual([gate.phase for gate in self.gates], [0, 0])

    def test_timeout_is_coherent_before_release(self):
        vote = ((0, 1, 1, 0), (0, 0, 0, 0))
        self.assertEqual(self.step(0, [self.requests[0][0]], vote), [])
        self.assertEqual(self.step(1, [], vote), [])
        self.time = 12_000_000
        vote = ((0, 1, 1, REASONS.index("group_timeout")), (0, 0, 0, 0))
        for rank in (0, 1):
            with self.assertRaisesRegex(GateFailure, "group_timeout"):
                self.step(rank, [], vote)

    def test_malformed_or_interrupted_vote_fails_without_native_work(self):
        gate = self.gates[0]
        def broken(_):
            raise TimeoutError("mock missing participant")
        with self.assertRaisesRegex(GateFailure, "ingress_collective_failed"):
            gate.step([self.requests[0][0]], lambda: True, broken)
        with self.assertRaisesRegex(GateFailure, "ingress_collective_failed"):
            gate.step([], None, lambda _: self.fail("must not retry failed vote"))
        gate = self.gates[1]
        with self.assertRaisesRegex(GateFailure, "state_mismatch"):
            gate.step([self.requests[0][1]], lambda: True, lambda _: ((0, 1, 1, 0), (1, 1, 1, 0)))
        self.assertEqual(gate.phase, 0)

    def test_processor_calls_original_once_per_native_poll(self):
        calls = []
        scheduler = SimpleNamespace(is_fully_idle=lambda: True)
        init = []
        remote = [(0, 0, 0, 0), (0, 1, 1, 0)]
        def initialize(value):
            init.append(value)
            return lambda row: (row, remote.pop(0))
        def native(value, batch):
            self.assertIs(value, scheduler)
            calls.append(batch)
            return "native-result"
        wrapped = _wrap_processor(native, self.gates[0], initialize)
        request = self.requests[0][0]
        time_stats = request.time_stats
        self.assertEqual(wrapped(scheduler, [request]), "native-result")
        self.assertEqual(wrapped(scheduler, []), "native-result")
        self.assertEqual(init, [scheduler])
        self.assertEqual(calls, [[], [request]])
        self.assertIs(calls[-1][0].time_stats, time_stats)

    def test_verify_gate_receipt_requires_both_fresh_worker_identities_and_exact_ids(self):
        rows = []
        for rank, gate in enumerate(self.gates):
            vote = ((0, 1, 1, 0), (0, 1, 1, 0))
            gate._event("ingress_gate_installed")
            self.step(rank, [self.requests[0][rank]], vote)
            vote = ((1, 1, 1, 0), (1, 1, 1, 0))
            self.step(rank, [self.requests[1][rank]], vote)
            rows.append([{**event, "pid": rank+110} for event in self.events[rank]])
        self.assertEqual(verify_ingress_events(rows, self.contract)["completed_groups"], ["warmup", "requests"])
        rows[0][2]["vote_index"] += 1
        with self.assertRaisesRegex(ValueError, "same ingress votes"):
            verify_ingress_events(rows, self.contract)
        rows[0][2]["vote_index"] -= 1
        rows[0][2]["input_ids_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "tokenized input"):
            verify_ingress_events(rows, self.contract)

    def test_live_binding_rejects_local_group_and_stock_collective_bypass(self):
        ps = SimpleNamespace(tp_size=2, pp_size=1, pp_rank=0, attn_tp_size=1, attn_cp_size=1,
                             attn_dp_size=2, tp_rank=0, attn_dp_rank=0)
        scheduler = SimpleNamespace(ps=ps, tp_cpu_group=object(), enable_dp_attention=True, enable_hisparse=True,
                                    require_mlp_sync=True, enable_overlap=False, _engine_paused=False,
                                    idle_sleeper=None, recv_skipper=None, scripted_scheduler_hook=None, dllm_config=None,
                                    disaggregation_mode="none", spec_algorithm=SimpleNamespace(is_none=lambda: True))
        dist = SimpleNamespace(get_backend=lambda _: "gloo", get_world_size=lambda _: 2, get_rank=lambda _: 0)
        envs = SimpleNamespace(SGLANG_SCHEDULER_SKIP_ALL_GATHER=SimpleNamespace(get=lambda: False))
        self.assertIs(_verify_live(scheduler, 0, dist, envs, "none"), scheduler.tp_cpu_group)
        envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get = lambda: True
        with self.assertRaisesRegex(GateFailure, "live scheduler"):
            _verify_live(scheduler, 0, dist, envs, "none")


if __name__ == "__main__":
    unittest.main()
