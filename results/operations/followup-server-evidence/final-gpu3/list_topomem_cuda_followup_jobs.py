"""List completed owned jobs from published CUDA audits, without consulting a live run."""
from argparse import ArgumentParser
import json
from pathlib import Path
import re

FOLLOWUPS = {
    "v12": ("engineering-dynamic-v12", "run/jobs.jsonl", 24),
    "v13": ("engineering-producer-v13", "run/jobs.jsonl", 24),
    "prompt-v4-abort": ("prompt-v4-abort", "server_job_index.json", 11),
    "v41-reboot-interruption": ("cuda-v41-reboot-interruption", "server_job_index.json", 26),
}
FINAL = ("cuda-rsi-v41-gpu3", "server_job_index.json", None)
JOB = re.compile(
    r"eval-[a-f0-9]{48}"
    r"|qualification-v12-59d19cb8a0ca-b[0-7]-(?:native|writeback-v11|writeback-v12)"
    r"|qualification-v13-19789ad0cd6c-b[0-7]-(?:native|lru-equivalent-v11|lru-equivalent-v13)"
)


def completed_group(results, directory, filename, count):
    path = results / directory / filename
    if filename.endswith(".jsonl"):
        jobs = [json.loads(line)["job_id"] for line in path.read_bytes().splitlines()]
    else:
        jobs = list(json.loads(path.read_bytes())["jobs"])
    if (count is not None and len(jobs) != count) or not jobs or len(jobs) != len(set(jobs)):
        raise ValueError(f"Unexpected job roster in {directory}")
    if not all(JOB.fullmatch(job) for job in jobs):
        raise ValueError(f"Unexpected job ID in {directory}")
    return jobs


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-final", action="store_true")
    args = parser.parse_args()
    definitions = {**FOLLOWUPS, "final-gpu3": FINAL} if args.include_final else FOLLOWUPS
    groups = {label: completed_group(args.results, *definition) for label, definition in definitions.items()}
    all_jobs = [job for group in groups.values() for job in group]
    if len(all_jobs) != len(set(all_jobs)):
        raise ValueError("Job is included in more than one group")
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(groups, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({label: len(jobs) for label, jobs in groups.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
