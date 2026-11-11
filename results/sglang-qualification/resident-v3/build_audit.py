"""Reconstruct the engineering probe summary from immutable archived receipts."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

SOURCE = '0f64a120aafa242f65efcef4cdf2cfaa9caa268bf8a10c5362d639ad28a66fff'
GATE = 'a8c385514f2ebe7732f5bffcf8e4dd4f2c66171e7a1462dea30bfec413648012'
COMMON = ('target_sha256', 'scenario_sha256', 'topology_sha256', 'schedule_inputs_sha256',
          'resident_proof_sha256', 'input_tokenization_sha256', 'ingress_contract_sha256',
          'runtime', 'global_ticks_sha256', 'effective_schedule_sha256', 'dispatch_sha256', 'output_ids_sha256')
ORDER = ('stock', 'resident', 'resident', 'stock')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def audit(path):
    result = {'schema': 1, 'kind': 'human_engineering_resident_backing_probe',
              'source_manifest_sha256': SOURCE, 'evidence_archive_sha256': sha(path.read_bytes()),
              'generated_program': False, 'recursive_improvement': False, 'heldout': False,
              'scope': 'One public authored DP2 architecture-checkpoint fixture, 6144 hot slots; two paired timing observations.',
              'phases': {}}
    all_rows = []
    with ZipFile(path) as z:
        inventory = {name: {'sha256': sha(z.read(name)), 'bytes': len(z.read(name))} for name in z.namelist()}
        for phase in ('parity', 'performance'):
            prefix = f'resident-{phase}-v3'
            proof = json.loads(z.read(prefix + '/qualification.json'))
            assert proof['qualified'] and proof['mismatches'] == []
            assert proof['source_manifest_sha256'] == SOURCE and proof['gated_qualification_sha256'] == GATE
            rows, records = [], []
            for index, mode in enumerate(ORDER, 1):
                name = f'{prefix}/{index:02d}-{mode}'
                raw = z.read(name + '/result.json')
                row = json.loads(raw)
                receipt = json.loads(z.read(name + '.probe.json'))
                assert sha(raw) == proof['result_sha256s'][index - 1] == receipt['result_sha256']
                assert row['status'] == receipt['status'] == 'completed'
                assert row['mode'] == mode and row['phase'] == phase
                assert row['source_manifest_sha256'] == SOURCE
                assert receipt['returncode'] == 0 and not receipt['process_group_cleanup_required']
                assert not receipt['process_group_remaining'] and not receipt['remaining_compute_pids']
                assert len(receipt['nccl_workers']) == 2
                for worker in receipt['nccl_workers']:
                    assert worker['maps_samples_after_decode'] > 0 and len(worker['libraries']) == 1
                    assert worker['libraries'][0]['qualified']
                    assert worker['libraries'][0]['nccl_get_version'] == 23203
                memory = []
                for rank in row['ranks']:
                    worker = rank['worker']
                    assert worker['hot_geometry']['slots'] == 6144
                    assert worker['hot_geometry']['packed_device_payload_bytes'] == 64739328
                    assert worker['cold_payload_bytes'] == (129226752 if mode == 'stock' else 0)
                    assert worker['cuda_pointer_sentinel_logical_bytes'] == (0 if mode == 'stock' else 3936)
                    assert (worker['oracle'] is not None) == (phase == 'parity')
                    if phase == 'parity':
                        assert worker['oracle']['verified_swaps'] == worker['oracle']['verified_newest_rows'] == 756
                        assert sorted(worker['oracle']['decode_steps_by_rid'].values()) == [63, 63]
                    memory.append({key: value for key, value in worker.items() if key not in ('pid',)})
                record = {'trial': index, 'mode': mode, 'raw_result_sha256': sha(raw), 'ranks': memory}
                if phase == 'performance':
                    measured = row['requests']
                    window_ms = (max(v['completed_ns'] for v in measured) - min(v['submitted_ns'] for v in measured)) / 1e6
                    assert abs(window_ms - row['metrics']['window_seconds'] * 1000) < 1e-8
                    record['window_ms'] = window_ms
                records.append(record)
                rows.append(row)
            assert all(all(row[key] == rows[0][key] for key in COMMON) for row in rows)
            if phase == 'parity':
                assert all(row['producer_and_selected_sha256_by_rank'] == rows[0]['producer_and_selected_sha256_by_rank'] for row in rows)
            else:
                assert all(row['producer_and_selected_sha256_by_rank'] is None for row in rows)
                assert all(all(row[key] == old[key] for key in COMMON) for row, old in zip(rows, all_rows[0]))
            result['phases'][phase] = records
            all_rows.append(rows)
        result['members'] = inventory
    values = [row['window_ms'] for row in result['phases']['performance']]
    result['paired_gain_ms'] = [values[0] - values[1], values[3] - values[2]]
    result['paired_relative_gain_pct'] = [100 * (values[0] - values[1]) / values[0], 100 * (values[3] - values[2]) / values[3]]
    result['posthoc_stability_claim'] = False
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.evidence)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')
    print(json.dumps({'paired_gain_ms': result['paired_gain_ms'], 'output_sha256': sha(args.output.read_bytes())}))
