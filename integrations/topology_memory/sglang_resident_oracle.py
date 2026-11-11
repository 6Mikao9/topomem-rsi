"""Diagnostic packed-row oracle for the resident-only backend."""

from functools import wraps
from hashlib import sha256
import json

from .sglang_resident import HOT_SLOTS, LAYERS, ROW_BYTES, TOP_K


def torch_row_reader(torch):
    def read(pool, layer, locations):
        slab = pool.kv_buffer[layer]
        index = torch.tensor(locations, dtype=torch.int64, device=slab.device)
        raw = slab.index_select(0, index).contiguous().view(torch.uint8).cpu().numpy().tobytes()
        if len(raw) != len(locations) * ROW_BYTES:
            raise RuntimeError("packed KV producer row geometry changed")
        return [raw[offset:offset + ROW_BYTES] for offset in range(0, len(raw), ROW_BYTES)]

    return read


class PackedProducerOracle:
    """Observe actual producer rows, then compare stock kernel-selected hot rows."""

    def __init__(self, read_rows):
        self.read_rows = read_rows
        self.pool = None
        self.requests = {}
        self.live_locations = set()
        self.pending = {}
        self.checked = set()
        self.completed = {}
        self.digest = sha256()
        self.swaps = self.history_rows = self.newest_rows = self.prefill_rows = 0

    def _reserve(self, locations):
        if (len(set(locations)) != len(locations) or self.live_locations.intersection(locations)
                or any(not 0 < location < self.pool.size + self.pool.page_size for location in locations)):
            raise RuntimeError("packed producer aliases or escapes live resident KV ownership")
        self.live_locations.update(locations)

    def _read(self, layer, locations):
        if any(not 0 < location < self.pool.size + self.pool.page_size for location in locations):
            raise RuntimeError("packed KV physical row outside the real hot pool")
        rows = self.read_rows(self.pool, layer, locations)
        if len(rows) != len(locations) or any(type(row) is not bytes or len(row) != ROW_BYTES for row in rows):
            raise RuntimeError("packed KV row reader violated its byte contract")
        return rows

    def capture_prefill(self, coordinator, req):
        pool = coordinator.mem_pool_device
        if self.pool is not None and self.pool is not pool:
            raise RuntimeError("packed KV producer pool changed")
        self.pool = pool
        slot, length = req.kv.req_pool_idx, req.extend_range.end
        if slot in self.requests or req.rid in self.completed:
            raise RuntimeError("packed KV oracle saw a repeated request epoch")
        logical = coordinator.req_to_token_pool.req_to_token[slot, :length].long()
        locations = pool.translate_loc_from_full_to_hisparse_device(logical).tolist()
        if len(locations) != length:
            raise RuntimeError("prefill producer lacks a logical row")
        self._reserve(locations)
        rows = [self._read(layer, locations) for layer in range(LAYERS)]
        self.requests[slot] = {"rid": req.rid, "locations": locations, "rows": rows, "decode_steps": 0}
        self.prefill_rows += length * LAYERS

    def _require_complete(self, slots):
        for slot, position, _ in self.pending.values():
            if slot in slots and any((slot, layer, position) not in self.checked for layer in range(LAYERS)):
                raise RuntimeError("missing packed KV producer-to-consumer layer evidence")

    def plan_decode(self, coordinator, logical, lengths_cpu, slots_cpu):
        self._require_complete(self.requests)
        slots, lengths = slots_cpu.tolist(), lengths_cpu.tolist()
        logical_ids = logical.tolist()
        locations = self.pool.translate_loc_from_full_to_hisparse_device(logical).tolist()
        if (coordinator.mem_pool_device is not self.pool or not len(slots) == len(lengths) == len(locations) == len(logical_ids)
                or len(set(slots)) != len(slots) or len(set(logical_ids)) != len(logical_ids)):
            raise RuntimeError("packed decode producer coordinates changed")
        self._reserve(locations)
        self.pending.clear()
        self.checked.clear()
        for logical_id, slot, length, location in zip(logical_ids, slots, lengths, locations):
            request = self.requests[slot]
            if any(len(rows) != length - 1 for rows in request["rows"]):
                raise RuntimeError("decode producer skipped or repeated a logical position")
            self.pending[logical_id] = (slot, length - 1, location)
            request["locations"].append(location)
            request["decode_steps"] += 1

    def capture_producer_store(self, pool, layer, logical):
        if pool is not self.pool or not self.pending:
            return
        coordinates = [self.pending[token] for token in logical.tolist() if token in self.pending]
        if not coordinates:
            return
        if not 0 <= layer < LAYERS:
            raise RuntimeError("unknown packed producer layer")
        rows = self._read(layer, [location for _, _, location in coordinates])
        for (slot, position, _), row in zip(coordinates, rows):
            history = self.requests[slot]["rows"][layer]
            if len(history) != position:
                raise RuntimeError("decode producer wrote the same logical position twice")
            history.append(row)

    def check_selected(self, coordinator, indices, lengths, topk, locations, layer):
        slots, lengths = indices.tolist(), lengths.tolist()
        selections, mapped = topk.tolist(), locations.tolist()
        real = int(coordinator.num_real_reqs.item())
        if (not 0 <= layer < LAYERS or not len(slots) == len(lengths) == len(selections) == len(mapped)
                or not 0 <= real <= len(slots) or len(set(slots[:real])) != real
                or any(len(row) != TOP_K for row in (*selections, *mapped))):
            raise RuntimeError("packed KV consumer geometry changed")
        if any(location != -1 for row in mapped[real:] for location in row):
            raise RuntimeError("padded request has a live KV location")
        if real and coordinator.mem_pool_device is not self.pool:
            raise RuntimeError("packed KV consumer pool changed")
        records = []
        for slot, length, positions, physical in zip(slots[:real], lengths[:real], selections[:real], mapped[:real], strict=True):
            request = self.requests.get(slot)
            position = length - 1
            key = (slot, layer, position)
            if (request is None or not 1 <= length <= HOT_SLOTS or length != len(request["locations"])
                    or key in self.checked or len(request["rows"][layer]) != length):
                raise RuntimeError("packed KV consumer lacks an independently observed producer row")
            pairs = []
            selected_tokens = set()
            for lane, (token, location) in enumerate(zip(positions, physical)):
                if lane >= min(length, TOP_K) or token < 0:
                    if token != -1 or location != -1:
                        raise RuntimeError("padded KV selection has a live token or location")
                elif token >= length:
                    raise RuntimeError("KV selection escapes its request length")
                else:
                    if token in selected_tokens:
                        raise RuntimeError("duplicate logical token in packed KV selection")
                    selected_tokens.add(token)
                    # This check is specific to the stock short path. Staging
                    # preserves the producer prefix, and decode grows it in place.
                    if location != request["locations"][token]:
                        raise RuntimeError("selected KV location differs from its resident producer address")
                    pairs.append((token, location))
            newest = int(coordinator.req_device_buffer_token_locs[layer, slot, position])
            if newest != request["locations"][position]:
                raise RuntimeError("newest KV location differs from its resident producer address")
            selected_rows = self._read(layer, [location for _, location in pairs] + [newest])
            expected = [request["rows"][layer][token] for token, _ in pairs] + [request["rows"][layer][position]]
            if selected_rows != expected:
                raise RuntimeError("resident hot bytes differ from the independently captured producer bytes")
            binding = [request["rid"], layer, position, positions]
            self.digest.update(json.dumps(binding, separators=(",", ":")).encode())
            for row in selected_rows:
                self.digest.update(row)
            self.swaps += 1
            self.history_rows += sum(token != position for token, _ in pairs)
            self.newest_rows += 1
            self.checked.add(key)
            history_bytes = b"".join(row for (token, _), row in zip(pairs, expected[:-1], strict=True) if token != position)
            records.append((slot, length, [token for token, _ in pairs], history_bytes, expected[-1]))
        return records

    def release(self, coordinator, req):
        slot = req.kv.req_pool_idx
        if slot not in self.requests:
            return
        self._require_complete({slot})
        request = self.requests.pop(slot)
        if coordinator.mem_pool_device is not self.pool or request["rid"] != req.rid:
            raise RuntimeError("packed KV release changed its request epoch")
        self.live_locations.difference_update(request["locations"])
        self.pending = {logical: row for logical, row in self.pending.items() if row[0] != slot}
        self.checked = {key for key in self.checked if key[0] != slot}
        self.completed[req.rid] = request["decode_steps"]

    def result(self, expected_decode_steps):
        if self.requests or self.live_locations or self.completed != expected_decode_steps:
            raise RuntimeError("packed KV oracle lacks the sealed request/decode coverage")
        expected_swaps = LAYERS * sum(expected_decode_steps.values())
        if self.swaps != expected_swaps or self.newest_rows != expected_swaps or not self.history_rows:
            raise RuntimeError("packed KV oracle did not verify every scheduled layer and decode step")
        return {"producer_and_selected_sha256": self.digest.hexdigest(), "verified_swaps": self.swaps,
                "verified_selected_history_rows": self.history_rows, "verified_newest_rows": self.newest_rows,
                "captured_prefill_rows": self.prefill_rows, "decode_steps_by_rid": dict(self.completed)}


def attach_oracle(coordinator, pool, oracle, *, emit, on_mismatch):
    """Install the same diagnostic around either resident or native backing."""
    if type(oracle) is not PackedProducerOracle or getattr(coordinator, "_resident_oracle_installed", False):
        raise RuntimeError("packed KV oracle must attach once in a fresh worker")
    original_stage = coordinator.admit_request_into_staging
    original_map = coordinator.map_last_loc_to_buffer
    original_store = pool.set_mla_kv_buffer
    original_swap = coordinator.swap_in_selected_pages

    @wraps(original_stage)
    def stage(self, req):
        oracle.capture_prefill(self, req)
        return original_stage(self, req)

    @wraps(original_map)
    def map_decode(self, seq_lens, out_cache_loc, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu):
        result = original_map(self, seq_lens, out_cache_loc, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu)
        oracle.plan_decode(self, out_cache_loc, seq_lens_cpu, req_pool_indices_cpu)
        return result

    @wraps(original_store)
    def store(self, layer, loc, *args, **kwargs):
        result = original_store(self, layer, loc, *args, **kwargs)
        oracle.capture_producer_store(self, layer.layer_id - self.start_layer, loc)
        return result

    @wraps(original_swap)
    def swap(self, req_pool_indices, compressed_seq_lens, top_k_result, layer_id):
        result = original_swap(self, req_pool_indices, compressed_seq_lens, top_k_result, layer_id)
        try:
            records = oracle.check_selected(self, req_pool_indices, compressed_seq_lens, top_k_result, result, layer_id)
        except RuntimeError as exc:
            on_mismatch(layer_id, str(exc))
            raise
        if records:
            emit(layer_id, records)
        return result

    for method in ("request_finished", "abort_staging_request"):
        def wrap_release(native):
            @wraps(native)
            def release(self, req):
                result = native(self, req)
                oracle.release(self, req)
                return result
            return release

        setattr(coordinator, method, wrap_release(getattr(coordinator, method)))

    coordinator.admit_request_into_staging = stage
    coordinator.map_last_loc_to_buffer = map_decode
    coordinator.swap_in_selected_pages = swap
    coordinator._resident_oracle_installed = True
    pool.set_mla_kv_buffer = store
