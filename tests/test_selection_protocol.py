from hashlib import sha256
import unittest

from rsi_memory.selection_protocol import Measurement, Proposal, Sample, TrainingArchive, TrainingPlan


SCOPE = "1" * 64
BLOCKS = ("train-1", "train-2", "train-3")
SEED, REJECTED, ADMITTED = "a" * 64, "b" * 64, "c" * 64
RESIDENT, TIERED = "2" * 64, "3" * 64


def raw(name):
    return sha256(name.encode()).hexdigest()


def proposal(name, source, effect, parents=(), lane="adaptive", program=None):
    return Proposal(name, lane, source, program or raw(name), effect, parents)


def measure(effect, values, status="complete", scope=SCOPE):
    samples = tuple(Sample(block, value, raw(effect + block)) for block, value in zip(BLOCKS, values))
    return Measurement(scope, effect, raw("qualification" + effect), status, samples)


class SelectionProtocolTests(unittest.TestCase):
    def setUp(self):
        self.plan = TrainingPlan(SCOPE, BLOCKS, 1.0)

    def test_submillisecond_difference_does_not_displace_incumbent(self):
        resident = proposal("resident", SEED, RESIDENT)
        tiered = proposal("tiered", SEED, TIERED)
        archive = TrainingArchive(self.plan, (resident, tiered), (
            measure(RESIDENT, [1082.374225, 1082.1, 1083.0]),
            measure(TIERED, [1082.367042, 1082.0, 1082.9]),
        ))
        selected = archive.choose("adaptive", incumbent_id="resident")
        self.assertEqual(selected.frontier_effects, (RESIDENT, TIERED))
        self.assertEqual(selected.proposal, resident)
        independent = archive.choose("adaptive", proposal_ids=("tiered",), incumbent_id="resident")
        self.assertEqual(independent.proposal, tiered)

    def test_rejected_provenance_does_not_swallow_real_seed_emission(self):
        old = proposal("rejected-resident", REJECTED, RESIDENT)
        seed = proposal("seed-resident", SEED, RESIDENT, program=old.program_sha256)
        assisted = proposal("assisted", SEED, RESIDENT, parents=(old.proposal_id,))
        baseline = proposal("seed-tiered", SEED, TIERED, parents=("bootstrap",))
        archive = TrainingArchive(self.plan, (old, seed, assisted, baseline), (
            measure(RESIDENT, [100, 101, 100]), measure(TIERED, [120, 122, 123]),
        ), trusted_anchor_ids=("bootstrap",))
        self.assertEqual(archive.provenance(RESIDENT), (old, seed, assisted))
        self.assertEqual(archive.direct_emission_ids((SEED,)), (seed.proposal_id, assisted.proposal_id, baseline.proposal_id))
        self.assertEqual(archive.fully_admitted_ancestry_ids((SEED,)), (seed.proposal_id, baseline.proposal_id))
        unrestricted = archive.choose("adaptive", incumbent_id=old.proposal_id)
        self.assertEqual(unrestricted.proposal, old)
        admitted_only = archive.choose("adaptive", proposal_ids=(old.proposal_id, seed.proposal_id),
                                       admitted_sources=(SEED,), incumbent_id=old.proposal_id)
        self.assertEqual(admitted_only.proposal, seed)
        self.assertEqual(admitted_only.proposal.program_sha256, old.program_sha256)
        self.assertEqual(admitted_only.proposal.source_sha256, SEED)

    def test_seed_edit_of_rejected_parent_is_a_real_emission(self):
        rejected = proposal("rejected", REJECTED, RESIDENT)
        seed_edit = proposal("seed-edit", SEED, RESIDENT, parents=(rejected.proposal_id,))
        later_edit = proposal("later-edit", ADMITTED, TIERED, parents=(seed_edit.proposal_id,))
        archive = TrainingArchive(self.plan, (rejected, seed_edit, later_edit), (
            measure(RESIDENT, [100, 100, 100]), measure(TIERED, [110, 110, 110]),
        ))
        self.assertEqual(archive.direct_emission_ids((SEED, ADMITTED)), ("seed-edit", "later-edit"))
        self.assertEqual(archive.fully_admitted_ancestry_ids((SEED, ADMITTED)), ())
        selected = archive.choose("adaptive", admitted_sources=(SEED, ADMITTED), incumbent_id=rejected.proposal_id)
        self.assertEqual(selected.proposal, seed_edit)
        self.assertEqual(selected.proposal.parent_proposal_ids, (rejected.proposal_id,))
        alias_only = archive.choose("adaptive", proposal_ids=(rejected.proposal_id,), admitted_sources=(SEED,))
        self.assertEqual(alias_only.status, "no_eligible_proposal")
        self.assertIsNone(alias_only.proposal)

    def test_discovery_and_admitted_output_can_pick_different_effects(self):
        resident = proposal("child-resident", REJECTED, RESIDENT)
        tiered = proposal("seed-tiered", SEED, TIERED)
        fixed = proposal("fixed-resident", SEED, RESIDENT, lane="fixed")
        archive = TrainingArchive(self.plan, (resident, tiered, fixed), (
            measure(RESIDENT, [100, 100, 100]), measure(TIERED, [115, 116, 117]),
        ))
        self.assertEqual(archive.choose("adaptive").proposal, resident)
        self.assertEqual(archive.choose("adaptive", admitted_sources=(SEED, ADMITTED)).proposal, tiered)
        self.assertEqual(archive.choose("fixed", admitted_sources=(SEED,)).proposal, fixed)

    def test_missing_training_effect_blocks_selection_and_correctness_failure_does_not(self):
        resident = proposal("resident", SEED, RESIDENT)
        tiered = proposal("tiered", SEED, TIERED)
        invalid = proposal("bad", SEED, None)
        measured = measure(RESIDENT, [110, 111, 112])
        incomplete = TrainingArchive(self.plan, (resident, tiered, invalid), (measured, measure(TIERED, [100], "incomplete")))
        pending = incomplete.choose("adaptive")
        self.assertEqual((pending.status, pending.missing_effects), ("incomplete_training", (TIERED,)))
        self.assertIsNone(pending.proposal)
        failed = TrainingArchive(self.plan, (resident, tiered, invalid), (measured, measure(TIERED, [], "correctness_failed")))
        result = failed.choose("adaptive")
        self.assertEqual(result.proposal, resident)
        self.assertEqual(result.excluded_infeasible_proposals, ("bad", "tiered"))

    def test_rejects_mismatched_scope_order_and_reused_receipts(self):
        candidate = proposal("resident", SEED, RESIDENT)
        valid = measure(RESIDENT, [101, 100, 102])
        wrong_scope = measure(RESIDENT, [101, 100, 102], scope="f" * 64)
        wrong_order = Measurement(SCOPE, RESIDENT, valid.qualification_sha256, "complete", valid.samples[::-1])
        reused = Measurement(SCOPE, TIERED, raw("other proof"), "complete", valid.samples)
        for records in ((wrong_scope,), (wrong_order,), (valid, reused)):
            with self.assertRaises(ValueError):
                TrainingArchive(self.plan, (candidate,), records)

    def test_late_alias_cannot_change_raw_effect_or_adopt_cross_lane_parent(self):
        old = proposal("adaptive", REJECTED, RESIDENT)
        inconsistent = proposal("same-raw", SEED, TIERED, program=old.program_sha256)
        fixed = proposal("fixed", SEED, RESIDENT, parents=(old.proposal_id,), lane="fixed")
        for candidate in (inconsistent, fixed):
            with self.assertRaises(ValueError):
                TrainingArchive(self.plan, (old, candidate), ())
