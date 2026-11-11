"""Bind the existing lease supervisor to the isolated SGLang evaluator only."""

import sys

sys.dont_write_bytecode = True

from rsi_memory import execution_worker


if __name__ == "__main__":
    execution_worker.MODULES = {"topomem_sglang_rsi.sglang_eval"}
    execution_worker.main()
