from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as Obj
import unittest

from integrations.topology_memory import sglang_observe as observe, sglang_resident as resident, sglang_serving as serving
from integrations.topology_memory.sglang_resident_oracle import PackedProducerOracle


def inputs(prompt=4200, decode=64):
    engine = {**resident.ENGINE_FIELDS, "max_total_tokens": 16384,
              "hisparse_config": json.dumps({"top_k": 2048, "device_buffer_size": 6144, "host_to_device_ratio": 2})}
    fixture = {phase: [{"rid": f"{phase}-{rank}", "routed_dp_rank": rank, "max_new_tokens": decode} for rank in (0, 1)]
               for phase in resident.PHASES}
    tokens = {row["rid"]: list(range(prompt)) for phase in resident.PHASES for row in fixture[phase]}
    commitments = serving._token_commitments(tokens)
    return engine, fixture, commitments


def binding(engine, fixture, commitments):
    return {"proof": resident.proof_from_commitments(engine, fixture, commitments, device_rows=16384),
            "allocator_charge_bytes_by_rank": [resident.RESIDENT_ALLOCATOR_CHARGE_BYTES] * 2}


class Vec(list):
    def tolist(self):
        return list(self)

    def long(self):
        return self


class TestResidentServing(unittest.TestCase):
    def test_scheduler_snapshot_uses_actual_wheel_attribute(self):
        args = {key: None for key in observe.SCHEDULER_FIELDS}
        args["max_total_tokens"] = 16640
        effective = Obj(server_args=Obj(**args), ps=Obj(tp_size=2, attn_tp_size=1, attn_dp_size=2, attn_dp_rank=1),
                        max_running_requests=1, max_total_num_tokens=16384, max_prefill_tokens=16384,
                        chunked_prefill_size=4096, max_req_len=8192, max_req_input_len=8191)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "schedule.json"
            observe._save_schedule(path, 1, effective)
            result = json.loads(path.read_bytes())
        self.assertEqual(result["effective"]["max_total_tokens"], 16640)
        self.assertEqual(result["effective"]["worker_limits"]["max_total_num_tokens"], 16384)
        self.assertNotIn("max_total_tokens", result["effective"]["worker_limits"])

    def test_proof_rebuilds_from_live_commitments_and_charges_pages(self):
        engine, fixture, commitments = inputs(6080, 64)
        sealed = binding(engine, fixture, commitments)
        p = resident.checked_binding(engine, fixture, commitments, sealed)
        self.assertEqual(p.reserved_rows_by_rank, (12416, 12416))
        self.assertEqual(p.fingerprint(), serving.fingerprint(sealed["proof"]))
        changed = dict(commitments)
        changed["requests-0"] = {**changed["requests-0"], "input_ids_sha256": "a" * 64}
        with self.assertRaisesRegex(ValueError, "differs from actual"):
            resident.checked_binding(engine, fixture, changed, sealed)
        with self.assertRaisesRegex(ValueError, "allocator charge"):
            resident.checked_binding(engine, fixture, commitments, {**sealed, "allocator_charge_bytes_by_rank": [0, 0]})
        with self.assertRaisesRegex(ValueError, "hot range"):
            binding(*inputs(6081, 64))

    def test_producer_bytes_have_the_same_canonical_stock_event_format(self):
        owner = Obj(size=16384, page_size=64, translate_loc_from_full_to_hisparse_device=lambda ids: Vec(
            [{100: 2, 101: 3, 102: 4}[item] for item in ids]))

        class Logical:
            def __getitem__(self, key):
                self.assert_key = key
                return Vec([100, 101])

        class Consumer:
            def __getitem__(self, key):
                return 4

        store = {2: bytes([2]) * 656, 3: bytes([3]) * 656, 4: bytes([4]) * 656, 17: bytes([3]) * 656}
        coordinator = Obj(mem_pool_device=owner, num_real_reqs=Obj(item=lambda: 1),
                          req_to_token_pool=Obj(req_to_token=Logical()), req_device_buffer_token_locs=Consumer())
        request = Obj(rid="r", kv=Obj(req_pool_idx=0), extend_range=Obj(end=2))
        oracle = PackedProducerOracle(lambda _pool, _layer, locs: [store[location] for location in locs])
        oracle.capture_prefill(coordinator, request)
        oracle.plan_decode(coordinator, Vec([102]), Vec([3]), Vec([0]))
        pad = [-1] * (resident.TOP_K - 3)
        selections = Vec([[1, 2, 0] + pad])
        with self.assertRaisesRegex(RuntimeError, "independently observed"):
            oracle.check_selected(coordinator, Vec([0]), Vec([3]), selections, Vec([[3, 4, 2] + pad]), 0)
        oracle.capture_producer_store(owner, 0, Vec([102]))
        with self.assertRaisesRegex(RuntimeError, "producer address"):
            oracle.check_selected(coordinator, Vec([0]), Vec([3]), selections, Vec([[17, 4, 2] + pad]), 0)
        store[3] = bytes([8]) * 656
        with self.assertRaisesRegex(RuntimeError, "independently captured"):
            oracle.check_selected(coordinator, Vec([0]), Vec([3]), selections, Vec([[3, 4, 2] + pad]), 0)
        store[3] = bytes([3]) * 656
        observed = oracle.check_selected(coordinator, Vec([0]), Vec([3]), selections, Vec([[3, 4, 2] + pad]), 0)
        row = observe._packed_row(*observed[0], 0)
        self.assertEqual(row, {"request_slot": 0, "seq_len": 3, "selected_ids": [1, 2, 0],
                         "history_bytes": 1312, "history_sha256": sha256(store[3] + store[2]).hexdigest(),
                         "newest_sha256": sha256(store[4]).hexdigest(), "miss_rows": 0})
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.jsonl"
            observe._packed_event(path, 0, 0, [row])
            self.assertEqual(json.loads(path.read_bytes()), {"event": "kv_selected_exact", "rank": 0, "layer": 0, "rows": [row]})

    def test_no_cold_receipt_and_diagnostic_coverage_are_required(self):
        sealed = binding(*inputs(prompt=4200, decode=3))
        bound = {"backing_mode": "resident", "resident_proof_sha256": serving.fingerprint(sealed["proof"]),
                 "resident_sentinel_logical_bytes": 3936, "resident_sentinel_allocator_charge_bytes": resident.RESIDENT_ALLOCATOR_CHARGE_BYTES,
                 "resident_sentinel_allocated_delta_bytes": 6144, "resident_sentinel_reserved_delta_bytes": 0,
                 "pinned_host_bytes": 0, "local_hbm_bytes": 0, "peer_hbm_bytes": 0,
                 "relay_workspace_bytes": {"compute": 0, "host": 0, "peer": 0}}
        common = {"backup_payload_bytes_by_tier": {"resident": 0}, "relay": None}
        diagnostic = {**common, "selected_miss_payload_bytes_by_tier": {"resident": 0},
                      "resident_producer_oracle": {"decode_steps_by_rid": {"warmup-0": 2, "requests-0": 2},
                          "verified_swaps": 24, "verified_newest_rows": 24, "captured_prefill_rows": 50400,
                          "verified_selected_history_rows": 500, "producer_and_selected_sha256": "a" * 64}}
        serving._verify_resident_rank(diagnostic, bound, 0, sealed, diagnostic=True)
        serving._verify_resident_rank(common, bound, 0, sealed, diagnostic=False)
        with self.assertRaisesRegex(ValueError, "diagnostic shadow"):
            serving._verify_resident_rank(diagnostic, bound, 0, sealed, diagnostic=False)
        with self.assertRaisesRegex(ValueError, "independent packed producer"):
            serving._verify_resident_rank({**diagnostic, "resident_producer_oracle": {}}, bound, 0, sealed, diagnostic=True)
        with self.assertRaisesRegex(ValueError, "no-cold allocation"):
            serving._verify_resident_rank(common, {**bound, "pinned_host_bytes": 64}, 0, sealed, diagnostic=False)

    def test_worker_uses_measured_sentinels_and_preserves_cleanup(self):
        engine, fixture, commitments = inputs(prompt=80, decode=5)
        p = resident.checked_binding(engine, fixture, commitments, binding(engine, fixture, commitments))
        events, reports, buffers = [], [], []
        device = Obj(type="cuda", index=0)
        layer = Obj(device=device, shape=(16448, 1, 656), nbytes=16448 * 656)
        pool = Obj(size=16384, page_size=64, layer_num=6, layer_shard_enabled=False,
                   kv_cache_dim=656, store_dtype=Obj(itemsize=1), kv_buffer=[layer] * 6)
        cuda = Obj(get_allocator_backend=lambda: "native", memory_allocated=lambda _: len(buffers) * 1024,
                   memory_reserved=lambda _: 2097152 if buffers else 0)

        def empty(shape, **kwargs):
            buffers.append(shape)
            return Obj(shape=shape, device=kwargs["device"])

        class Event:
            def record(self):
                events.append("event")

            def wait(self, _stream):
                events.append("wait")

        @contextmanager
        def stream(_value):
            yield

        class Host:
            def __init__(self, *, is_dummy, pin_memory, **_kwargs):
                self._is_dummy, self.pin_memory, self.size = is_dummy, pin_memory, 32832

        class Coord:
            def __init__(self):
                self.mem_pool_host = module.MLATokenToKVPoolHost(device_pool=pool, layout="layer_first", page_size=64,
                                                               host_to_device_ratio=2, host_size=0)
                self.mem_pool_device = pool
                self.req_to_token_pool = Obj(req_to_token=Obj(shape=(2, 8192)))
                for key in ("req_to_device_buffer", "req_device_buffer_tokens", "req_device_buffer_token_locs",
                            "_lru_init", "lru_slots", "_device_buffer_arange_i32"):
                    setattr(self, key, Obj(nbytes=128))
                self.is_dsv4_hisparse = self.enable_prefetch = self.skip_io = self._has_pending_backup = False
                self.compress_ratio, self.page_size, self.item_size_bytes = 1, 64, 656
                self.device_buffer_size, self.top_k, self.tp_world_size = 6144, 2048, 1
                self.ack_staging_queue, self._skip_first_backup, self.req_to_host_pool_allocated_len = [], [True], [0]
                self.write_staging_stream = "staging"

            def request_finished(self, _req):
                events.append("native-finish")

            def abort_staging_request(self, _req):
                events.append("native-abort")

        class Scheduler:
            def _add_request_to_queue(self, _req, is_retracted=False):
                events.append("queue")

        class Batch:
            def prepare_for_extend(self):
                events.append("prefill")

            def prepare_for_decode(self):
                events.append("decode")

        module = Obj(HiSparseCoordinator=Coord, MLATokenToKVPoolHost=Host,
                     device_module=Obj(Event=Event, stream=stream), HiSparseAct=lambda *args: args)
        resident._patched_classes(p, 0, Obj(cuda=cuda, empty=empty), module, Scheduler, Batch, reports.append)
        coord = Coord()
        self.assertEqual(buffers, [(1, 1, 656)] * 6)
        self.assertEqual(reports[0]["resident_sentinel_allocated_delta_bytes"], 6144)
        self.assertEqual(reports[0]["resident_sentinel_reserved_delta_bytes"], 2097152)
        self.assertEqual(reports[0]["packed_device_payload_bytes"], 16448 * 656 * 6)
        self.assertEqual(reports[0]["hot_metadata_bytes"], 768)
        self.assertEqual(reports[0]["pinned_host_bytes"], 0)
        with self.assertRaisesRegex(RuntimeError, "cannot allocate"):
            coord.mem_pool_host.backup_from_device_all_layer()
        req = Obj(rid="warmup-0", origin_input_ids=list(range(80)), sampling_params=Obj(max_new_tokens=5),
                  kv=Obj(req_pool_idx=0, kv_allocated_len=80), extend_range=Obj(end=80), hisparse_staging=False)
        scheduler = Scheduler()
        scheduler.ps, scheduler.enable_overlap, scheduler.hisparse_coordinator = Obj(tp_rank=0), False, coord
        scheduler._add_request_to_queue(req)
        batch = Batch()
        batch.reqs, batch.seq_lens_cpu = [req], Vec([84])
        batch.prepare_for_decode()
        batch.seq_lens_cpu = Vec([85])
        with self.assertRaisesRegex(ValueError, "resident-only proof"):
            batch.prepare_for_decode()
        coord.admit_request_into_staging(req)
        coord._eager_backup_previous_token(None, None, Vec([81]), Vec([0]))
        coord.request_finished(req)
        self.assertEqual(events, ["queue", "decode", "event", "wait", "event", "native-finish"])
        self.assertFalse(coord._resident_bounds_by_slot)


if __name__ == "__main__":
    unittest.main()
