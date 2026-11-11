"""Training-only selection over immutable proposals and matched physical measurements."""

from dataclasses import dataclass
import math
import re
from statistics import median


_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha(value, name):
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise ValueError(f"invalid {name} SHA-256")


def _identifier(value):
    if type(value) is not str or not value or len(value) > 160:
        raise ValueError("invalid selection identifier")


def _unique_ids(values):
    if type(values) is not tuple:
        raise ValueError("expected an immutable sequence of identifiers")
    for value in values:
        _identifier(value)
    if len(values) != len(set(values)):
        raise ValueError("duplicate selection identifier")


def _valid_ms(value, *, positive):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError("invalid finite timing value")


@dataclass(frozen=True)
class TrainingPlan:
    scope_sha256: str
    block_ids: tuple[str, ...]
    noise_margin_ms: float

    def __post_init__(self):
        _sha(self.scope_sha256, "training scope")
        _unique_ids(self.block_ids)
        if not 3 <= len(self.block_ids) <= 100:
            raise ValueError("require 3–100 predeclared training blocks")
        _valid_ms(self.noise_margin_ms, positive=True)


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    lane: str
    source_sha256: str
    program_sha256: str
    effect_sha256: str | None
    parent_proposal_ids: tuple[str, ...] = ()

    def __post_init__(self):
        _identifier(self.proposal_id)
        if self.lane not in ("adaptive", "fixed"):
            raise ValueError("invalid proposal lane")
        _sha(self.source_sha256, "generator source")
        _sha(self.program_sha256, "raw program")
        if self.effect_sha256 is not None:
            _sha(self.effect_sha256, "training effect")
        _unique_ids(self.parent_proposal_ids)


@dataclass(frozen=True)
class Sample:
    block_id: str
    window_ms: float
    receipt_sha256: str

    def __post_init__(self):
        _identifier(self.block_id)
        _valid_ms(self.window_ms, positive=True)
        _sha(self.receipt_sha256, "timing receipt")


@dataclass(frozen=True)
class Measurement:
    scope_sha256: str
    effect_sha256: str
    qualification_sha256: str
    status: str
    samples: tuple[Sample, ...] = ()

    def __post_init__(self):
        _sha(self.scope_sha256, "training scope")
        _sha(self.effect_sha256, "training effect")
        _sha(self.qualification_sha256, "qualification")
        if self.status not in ("complete", "correctness_failed", "incomplete"):
            raise ValueError("unknown measurement status")
        if type(self.samples) is not tuple or any(type(item) is not Sample for item in self.samples):
            raise ValueError("invalid timing sample sequence")
        if self.status == "correctness_failed" and self.samples:
            raise ValueError("a correctness failure cannot supply timing samples")


@dataclass(frozen=True)
class Selection:
    status: str
    proposal: Proposal | None
    frontier_effects: tuple[str, ...] = ()
    missing_effects: tuple[str, ...] = ()
    excluded_infeasible_proposals: tuple[str, ...] = ()


class TrainingArchive:
    """Index trial provenance independently of deduplicated training observations."""

    def __init__(self, plan, proposals, measurements, *, trusted_anchor_ids=()):
        self.plan = plan
        _unique_ids(trusted_anchor_ids)
        self._anchors = frozenset(trusted_anchor_ids)
        self._proposals, self._measurements = {}, {}
        raw_effects, receipt_ids = {}, set()
        for trial in proposals:
            if type(trial) is not Proposal or trial.proposal_id in self._proposals or trial.proposal_id in self._anchors:
                raise ValueError("invalid or repeated proposal")
            for parent in trial.parent_proposal_ids:
                if parent not in self._anchors and (parent not in self._proposals or self._proposals[parent].lane != trial.lane):
                    raise ValueError("parent must be a previously recorded proposal in the same lane or a trusted anchor")
            if raw_effects.setdefault(trial.program_sha256, trial.effect_sha256) != trial.effect_sha256:
                raise ValueError("one raw program has conflicting effects in one training scope")
            self._proposals[trial.proposal_id] = trial

        for row in measurements:
            if type(row) is not Measurement or row.scope_sha256 != plan.scope_sha256 or row.effect_sha256 in self._measurements:
                raise ValueError("invalid, out-of-scope or repeated physical measurement")
            blocks = tuple(sample.block_id for sample in row.samples)
            if blocks != plan.block_ids[:len(blocks)] or (row.status == "complete" and blocks != plan.block_ids):
                raise ValueError("timing blocks differ from the predeclared order")
            for sample in row.samples:
                if sample.receipt_sha256 in receipt_ids:
                    raise ValueError("one timing receipt cannot count as multiple measurements")
                receipt_ids.add(sample.receipt_sha256)
            self._measurements[row.effect_sha256] = row

    def provenance(self, effect_sha256):
        """Return every proposal with this training effect, including rejected sources."""
        return tuple(trial for trial in self._proposals.values() if trial.effect_sha256 == effect_sha256)

    def direct_emission_ids(self, admitted_sources):
        """Identify proposals actually emitted by one of the admitted sources."""
        admitted = frozenset(admitted_sources)
        for digest in admitted:
            _sha(digest, "admitted source")
        return tuple(trial.proposal_id for trial in self._proposals.values() if trial.source_sha256 in admitted)

    def fully_admitted_ancestry_ids(self, admitted_sources):
        """Diagnose which admitted emissions also have admitted editing ancestry."""
        allowed = set(self._anchors)
        result = []
        for trial_id in self.direct_emission_ids(admitted_sources):
            if set(self._proposals[trial_id].parent_proposal_ids) <= allowed:
                allowed.add(trial_id)
                result.append(trial_id)
        return tuple(result)

    def _beats(self, faster, slower):
        a, b = (self._measurements[digest].samples for digest in (faster, slower))
        gains = [right.window_ms - left.window_ms for left, right in zip(a, b, strict=True)]
        return min(gains) > 0 and median(gains) > self.plan.noise_margin_ms

    def choose(self, lane, *, proposal_ids=None, admitted_sources=None, incumbent_id=None):
        """Pick a measured, undominated effect and an exact eligible proposal for it.

        A fresh parent/child comparison supplies its own proposal IDs and no
        incumbent. An eligible incumbent can settle an unresolved training tie.
        """
        if lane not in ("adaptive", "fixed"):
            raise ValueError("invalid proposal lane")
        ids = tuple(self._proposals) if proposal_ids is None else proposal_ids
        _unique_ids(ids)
        if any(trial_id not in self._proposals for trial_id in ids):
            raise ValueError("unknown proposal ID")
        eligible = None if admitted_sources is None else set(self.direct_emission_ids(admitted_sources))
        members, excluded, missing = {}, [], set()
        for trial_id in ids:
            trial = self._proposals[trial_id]
            if trial.lane != lane or (eligible is not None and trial_id not in eligible):
                continue
            effect = trial.effect_sha256
            observation = self._measurements.get(effect)
            if effect is None or (observation is not None and observation.status == "correctness_failed"):
                excluded.append(trial_id)
                continue
            if observation is None or observation.status != "complete":
                missing.add(effect)
            members.setdefault(effect, []).append(trial)
        if missing:
            return Selection("incomplete_training", None, missing_effects=tuple(sorted(missing)),
                             excluded_infeasible_proposals=tuple(sorted(excluded)))
        if not members:
            return Selection("no_eligible_proposal", None, excluded_infeasible_proposals=tuple(sorted(excluded)))

        frontier = tuple(effect for effect in sorted(members) if not any(
            other != effect and self._beats(other, effect) for other in members))
        incumbent = self._proposals.get(incumbent_id)
        if incumbent is not None and incumbent.effect_sha256 in frontier and incumbent in members[incumbent.effect_sha256]:
            selected = incumbent
        else:
            selected = min(members[frontier[0]], key=lambda trial: trial.proposal_id)
        return Selection("selected", selected, frontier, excluded_infeasible_proposals=tuple(sorted(excluded)))
