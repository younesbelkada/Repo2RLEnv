"""Compare seeds and their harder variants from Harbor job results.

  uv run python scripts/harder_pilot/score.py JOB_DIR [JOB_DIR ...] [--json out.json]

Reads every <job>/<trial>/result.json (reward at verifier_result.rewards.reward). A task's mean
reward averages the attempts that reached the verifier; "solved" is the share with reward 1.0,
which is how the model card counts it. Attempts that crashed before grading count as errors.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

from repo2rlenv.ui import console

VARIANTS = ("re-traces", "re-blind", "debug")


def read_rewards(job_dirs: list[Path]) -> dict[str, list[float | None]]:
    rewards = collections.defaultdict(list)
    for job_dir in job_dirs:
        for result_path in sorted(job_dir.glob("*/result.json")):
            trial = json.loads(result_path.read_text())
            # Trial names truncate long task names (both "-re-*" variants become "-re"), so use the
            # task folder, then Harbor's task_name, and the trial name only as a last resort.
            task = (
                Path(((trial.get("config") or {}).get("task") or {}).get("path") or "").name
                or str(trial.get("task_name") or "").rsplit("/", 1)[-1]
                or trial.get("trial_name", result_path.parent.name).split("__")[0]
            )
            reward = ((trial.get("verifier_result") or {}).get("rewards") or {}).get("reward")
            rewards[task].append(None if reward is None else float(reward))
    return rewards


def split(task: str) -> tuple[str, str]:
    for variant in VARIANTS:
        if task.endswith(f"-{variant}"):
            return task.removesuffix(f"-{variant}"), variant
    return task, "seed"


def summarize(values: list[float | None]) -> dict:
    scored = [v for v in values if v is not None]
    return {
        "attempts": len(values),
        "errors": len(values) - len(scored),
        "mean_reward": round(statistics.mean(scored), 3) if scored else None,
        "solved": round(sum(v == 1.0 for v in scored) / len(scored), 3) if scored else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("job_dirs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, help="also write the table as JSON")
    args = parser.parse_args()

    by_seed = collections.defaultdict(dict)
    for task, values in read_rewards(args.job_dirs).items():
        seed, variant = split(task)
        by_seed[seed][variant] = summarize(values)

    columns = ("seed", *VARIANTS)
    console.print(f"{'seed task':<34}" + "".join(f"{c:>20}" for c in columns))
    console.print(f"{'':<34}" + "".join(f"{'solved / mean':>20}" for _ in columns))
    for seed, row in sorted(by_seed.items()):
        cells = []
        for column in columns:
            s = row.get(column)
            cells.append(
                f"{s['solved']} / {s['mean_reward']}" if s and s["solved"] is not None else "-"
            )
        console.print(f"{seed:<34}" + "".join(f"{c:>20}" for c in cells))

    console.print()
    for column in columns:
        rows = [
            row[column]
            for row in by_seed.values()
            if column in row and row[column]["solved"] is not None
        ]
        if rows:
            solved = statistics.mean(r["solved"] for r in rows)
            mean = statistics.mean(r["mean_reward"] for r in rows)
            errors = sum(r["errors"] for r in rows)
            console.print(
                f"{column:<10} tasks={len(rows):<3} solved={solved:.3f} mean_reward={mean:.3f} errors={errors}"
            )
    if args.json:
        args.json.write_text(json.dumps(by_seed, indent=1))


if __name__ == "__main__":
    main()
