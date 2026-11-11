"""Fixed backup event before a native HiSparse newest slot can be overwritten."""

from functools import wraps
import os

from .sglang_observe import append
from .sglang_sources import COORDINATOR, SOURCES, check_source


def _wrap_map(original, current_stream, on_wait):
    @wraps(original)
    def ordered(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if self._has_pending_backup:
            self._backup_done_event.wait(current_stream())
            on_wait()
        return result

    return ordered


def install(marker):
    from sglang.srt.managers import hisparse_coordinator

    source_sha = SOURCES[COORDINATOR]
    check_source(hisparse_coordinator.__file__, source_sha)
    cls = hisparse_coordinator.HiSparseCoordinator
    if getattr(cls, "_rsi_backup_dependency_installed", False):
        return
    reported = False

    def report():
        nonlocal reported
        if not reported:
            append(marker, {"event": "backup_dependency_exercised", "pid": os.getpid()})
            reported = True

    cls.map_last_loc_to_buffer = _wrap_map(
        cls.map_last_loc_to_buffer, hisparse_coordinator.device_module.current_stream, report)
    cls._rsi_backup_dependency_installed = True
    append(marker, {"event": "backup_dependency_installed", "pid": os.getpid(), "coordinator_sha256": source_sha,
                    "invariant": "backup event precedes reuse of the newest-token slot"})
