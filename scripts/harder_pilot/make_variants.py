"""Derive harder variants from published traj2envs-rl tasks (pilot, not a pipeline).

For each seed task (reference in solution/files/, calibrated tests/test_hidden.py) it writes:

  re-traces  reimplement the module from its interface and a log of real calls (~50)
  re-blind   the same with only module and function names and a shorter log (~30)
  debug      the reference with 3 injected, independent, test-detected bugs plus a bug report

All three are graded by the seed's hidden tests plus behavior checks: the scenarios that produced
the call log are rerun on unseen seeds and compared with the original's recorded behavior.

  uv run python scripts/harder_pilot/make_variants.py --seeds ID [ID ...] --out workspace/harder-pilot

Every variant is checked in Docker before it is kept: the reference scores 1.0, doing nothing
scores 0 (re-*) or below 1 (debug). `report.json` in --out records what was kept and why not.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import random
import re
import shutil
import tempfile
import tomllib
from pathlib import Path

import tomli_w
from huggingface_hub import snapshot_download
from pydantic import BaseModel

from repo2rlenv.pipelines.trajectory import docker, llm
from repo2rlenv.pipelines.trajectory.mutants import TEST_FILE, mutate
from repo2rlenv.pipelines.trajectory.task import Trial, run_trials
from repo2rlenv.spec.input import LLMSpec
from repo2rlenv.ui import console

ASSETS = Path(__file__).parent / "assets"
DATASET = "harness-hub/traj2envs-rl"
SHOWN_SEEDS = list(range(10))
HIDDEN_SEEDS = list(range(1000, 1020))
TRACE_BUDGET = {"re-traces": 50, "re-blind": 30}
SINGLE_MUTANTS = 40
COMBO_ATTEMPTS = 8

SCENARIO_PROMPT = """You write behavior-probing scenarios for a Python project. They are used to (1) show
an engineer a log of real calls so they can reimplement the project without seeing its source, and
(2) check a reimplementation against the original on new random inputs.

Write 5 to 8 scenarios that together cover the whole public interface: normal cases, edge cases
and error cases. Each scenario is a complete Python function:

    def scenario_<short_name>(api, rng):
        m = api.mod("<module path>")
        ...

Rules:
- Get project modules only with api.mod("<module path>") (module paths are listed below) and call
  public functions, classes and methods only through the objects it returns. Every such call is
  logged automatically with its result or the exception type it raised.
- Draw every input from rng (a random.Random). The same seed must give the same calls.
- Make 3 to 15 calls per run. When a call may raise, wrap it in try/except Exception: pass.
- On returned objects only call methods and read attributes. Log anything you compute yourself
  with api.note(value).
- No time, no network, no absolute or temporary paths in results. Standard library only; import
  inside the function.
- Never compute expected values or reimplement the project: the log records the original's outputs.
"""

BUG_REPORT_PROMPT = """You are a user of a Python project filing a bug report. Below are the project's
specification and the checks that now fail, with their error messages. Write a concise report
(5 to 12 sentences or bullet points) describing the wrong behavior you observe, with concrete
inputs and the outputs you got versus what the specification says. Do not mention tests, test
names, file names, line numbers or how to fix it. Do not reveal how many bugs there are.
"""

RE_INSTRUCTION = """Reimplement a Python project from its observed behavior, in `/workspace` (it is currently empty).

The original source is not available. Below are {what} and a log of real calls to the original,
each followed by the value it returned or the exception it raised. Your implementation must behave
identically, also on inputs not shown here: hidden checks call it with new inputs and compare the
results with the original.
{title}
## Files

{files}

## Interface

{interface}

## Observed calls

```
{trace}
```

## Environment

Python 3.12, standard library only (pytest is installed). No internet. Each hidden check passed earns partial credit.
"""

DEBUG_INSTRUCTION = """The project in `/workspace` should implement the specification below, but users report
bugs. Find and fix them. Keep the public interface unchanged.

## Bug report

{report}

## Specification

{request}

{contract}

## Environment

Python 3.12, standard library only (pytest is installed). No internet. Each hidden check passed earns partial credit.
"""

TEST_SH = """#!/usr/bin/env bash
# Hidden tests plus behavior checks against the original; reward = fraction passed.
mkdir -p /logs/verifier
cd /tmp && python -m pytest -q -p no:cacheprovider --timeout=60 --junitxml=/logs/verifier/junit.xml \\
  /tests/test_hidden.py > /logs/verifier/pytest.log 2>&1
if [ -f /tests/behavior/expected.json ]; then
  python /tests/behavior/behavior.py check --root "${WORKSPACE:-/workspace}" \\
    --scenarios /tests/behavior/scenarios.py --expected /tests/behavior/expected.json \\
    --junit /logs/verifier/behavior.xml > /logs/verifier/behavior.log 2>&1
fi
python - <<'PY'
import xml.etree.ElementTree as ET
passed = 0
for report in ("/logs/verifier/junit.xml", "/logs/verifier/behavior.xml"):
    try:
        for case in ET.parse(report).getroot().iter("testcase"):
            passed += not any(child.tag in ("failure", "error", "skipped") for child in case)
    except Exception:
        pass
open("/logs/verifier/reward.txt", "w").write(f"{min(passed, TOTAL) / TOTAL:.4f}\\n")
PY
"""

SOLVE_SH = "#!/usr/bin/env bash\nset -e\n# The original implementation.\ncp -a /solution/files/. /workspace/\n"


class Scenario(BaseModel):
    name: str
    code: str


class Scenarios(BaseModel):
    scenarios: list[Scenario]


class BugReport(BaseModel):
    report: str


class Skip(Exception):
    pass


# ---------------------------------------------------------------- seed task


class Seed:
    def __init__(self, task_id: str, task_dir: Path):
        self.task_id, self.dir = task_id, task_dir
        self.toml = tomllib.loads((task_dir / "task.toml").read_text())
        instruction = (task_dir / "instruction.md").read_text()
        self.request = section(instruction, "## Request\n", "\n## Interface contract")
        self.contract = section(
            instruction, "## Interface contract\n", "\n## Environment", last=True
        )
        self.contract = self.contract.replace(
            "Hidden tests use only this interface, so follow it exactly. Everything else is up to you.",
            "",
        ).strip()
        files_dir = task_dir / "solution" / "files"
        self.reference = {
            str(p.relative_to(files_dir)): p.read_text()
            for p in sorted(files_dir.rglob("*"))
            if p.is_file() and not TEST_FILE.search(str(p.relative_to(files_dir)))
        }
        self.hidden_tests = int(self.toml["metadata"]["hidden_tests"])
        self.image = f"harder-pilot-seed-{task_id.lower()}"

    def modules(self) -> list[str]:
        return sorted(
            path.removesuffix(".py").removesuffix("/__init__").replace("/", ".")
            for path in self.reference
            if path.endswith(".py")
        )


def section(text: str, start: str, end: str, last: bool = False) -> str:
    begin = text.find(start)
    if begin < 0:
        return ""
    begin += len(start)
    stop = text.rfind(end) if last else text.find(end, begin)
    return text[begin : stop if stop > begin else len(text)].strip()


def download(task_id: str, seeds_dir: Path) -> Path:
    snapshot_download(
        DATASET, repo_type="dataset", allow_patterns=[f"tasks/{task_id}/*"], local_dir=seeds_dir
    )
    task_dir = seeds_dir / "tasks" / task_id
    if not (task_dir / "task.toml").exists():
        raise Skip(f"{task_id} not found in {DATASET}")
    return task_dir


def write_files(root: Path, files: dict[str, str]) -> None:
    for path, content in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(content)


# ---------------------------------------------------------------- behavior scenarios


def make_scenarios(seed: Seed) -> str:
    reference = "\n\n".join(f"### {p}\n```\n{c}\n```" for p, c in seed.reference.items())
    reply = llm.ask(
        SCENARIO_PROMPT,
        f"<module_paths>\n{', '.join(seed.modules())}\n</module_paths>\n\n"
        f"<request>\n{seed.request}\n</request>\n\n<contract>\n{seed.contract}\n</contract>\n\n"
        f"<reference>\n{reference}\n</reference>",
        Scenarios,
    )
    if reply is None:
        raise Skip("llm_declined (scenarios)")
    functions = []
    for scenario in reply.scenarios:
        try:
            tree = ast.parse(scenario.code)
        except SyntaxError:
            continue
        if any(
            isinstance(n, ast.FunctionDef) and n.name.startswith("scenario_") for n in tree.body
        ):
            functions.append(scenario.code.strip())
    if not functions:
        raise Skip("no valid scenarios")
    return "import random\n\n\n" + "\n\n\n".join(functions) + "\n"


def record_behavior(seed: Seed, scenarios: str, seeds: list[int], work: Path, tag: str) -> dict:
    pilot = work / f"behavior-{tag}"
    pilot.mkdir(parents=True)
    shutil.copy(ASSETS / "behavior.py", pilot / "behavior.py")
    (pilot / "scenarios.py").write_text(scenarios)
    write_files(work / "reference", seed.reference)
    command = (
        "python /pilot/behavior.py record --root /reference --scenarios /pilot/scenarios.py "
        f"--seeds {','.join(map(str, seeds))} --out /pilot/log.json"
    )
    _, output = docker.run_in_container(
        seed.image,
        command,
        {pilot: "/pilot", work / "reference": "/reference:ro"},
        timeout_seconds=1800,
    )
    if not (pilot / "log.json").exists():
        raise Skip(f"behavior recording failed: {output[-300:]}")
    return json.loads((pilot / "log.json").read_text())


def usable(runs: dict | None) -> bool:
    if not runs:
        return False
    for log in runs.values():
        if not log or any(
            isinstance(e.get("result"), dict) and "__error__" in e["result"] for e in log
        ):
            return False
    return True


def behavior(seed: Seed, work: Path) -> tuple[str, dict, dict]:
    """(scenarios source, hidden-seed expectations, shown-seed logs) for deterministic scenarios."""
    scenarios = make_scenarios(seed)
    seeds = SHOWN_SEEDS + HIDDEN_SEEDS
    first = record_behavior(seed, scenarios, seeds, work, "a")
    second = record_behavior(seed, scenarios, seeds, work, "b")
    kept = [name for name, runs in first.items() if usable(runs) and runs == second.get(name)]
    if len(kept) < 2:
        raise Skip(f"only {len(kept)} deterministic scenarios")
    expected = {n: {str(s): first[n][str(s)] for s in HIDDEN_SEEDS} for n in kept}
    shown = {n: {str(s): first[n][str(s)] for s in SHOWN_SEEDS} for n in kept}
    return scenarios, expected, shown


def render_trace(shown: dict, budget: int) -> str:
    blocks, total = [], 0
    for seed in map(str, SHOWN_SEEDS):
        for name in sorted(shown):
            log = shown[name][seed]
            if total and total + len(log) > budget:
                continue
            lines = []
            for entry in log:
                if entry["call"] == "note":
                    lines.append(f"# value: {entry['shown']}")
                else:
                    lines += [f">>> {entry['call']}", entry.get("shown", "")]
            blocks.append("\n".join(lines))
            total += len(log)
    return "\n\n".join(blocks)


def public_names(seed: Seed) -> str:
    lines = []
    for path, source in seed.reference.items():
        if not path.endswith(".py"):
            continue
        tree = ast.parse(source)
        names = [
            f"{n.name} (class)" if isinstance(n, ast.ClassDef) else n.name
            for n in tree.body
            if isinstance(n, ast.FunctionDef | ast.ClassDef) and not n.name.startswith("_")
        ]
        if names:
            module = path.removesuffix(".py").replace("/", ".")
            lines.append(f"- `{module}`: {', '.join(names)}")
    return "\n".join(lines)


def interface_code(seed: Seed) -> str:
    blocks = re.findall(r"```python\n(.*?)```", seed.contract, re.S)
    return (
        "```python\n" + "\n".join(b.strip() for b in blocks) + "\n```"
        if blocks
        else public_names(seed)
    )


# ---------------------------------------------------------------- debug: injected bugs


def enclosing_functions(tree: ast.AST) -> dict[int, str]:
    """id(node) -> name of the innermost function containing it."""
    owner = {}

    def visit(node, function):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            function = node.name
        owner[id(node)] = function
        for child in ast.iter_child_nodes(node):
            visit(child, function)

    visit(tree, "<module>")
    return owner


def mutation_sites(files: dict[str, str]) -> list[tuple[str, int, str]]:
    sites = []
    for path, source in files.items():
        if not path.endswith(".py"):
            continue
        tree = ast.parse(source)
        owner = enclosing_functions(tree)
        for index, node in enumerate(ast.walk(tree)):
            if owner[id(node)] != "<module>" and mutate(copy.deepcopy(node)):
                sites.append((path, index, owner[id(node)]))
    return sites


def apply_bugs(files: dict[str, str], bugs: list[tuple[str, int, str]]) -> dict[str, str]:
    result = dict(files)
    for path in {b[0] for b in bugs}:
        tree = ast.parse(files[path])
        nodes = list(ast.walk(tree))  # references stay valid while mutating
        for _, index, _ in (b for b in bugs if b[0] == path):
            mutate(nodes[index])
        result[path] = ast.unparse(ast.fix_missing_locations(tree))
    return result


def run_variants(seed: Seed, variants: dict[str, dict[str, str]], work: Path) -> dict:
    # A fresh folder per run: Docker Desktop can serve a stale view of a deleted, recreated mount.
    pilot = Path(tempfile.mkdtemp(prefix="variants-run-", dir=work))
    for name, files in variants.items():
        (pilot / "variants" / name).mkdir(parents=True)
        write_files(pilot / "variants" / name, files)
    shutil.copy(seed.dir / "tests" / "test_hidden.py", pilot / "test_hidden.py")
    shutil.copy(ASSETS / "run_variants.py", pilot / "run_variants.py")
    _, output = docker.run_in_container(
        seed.image, "python /pilot/run_variants.py", {pilot: "/pilot"}, timeout_seconds=3600
    )
    if not (pilot / "results.json").exists():
        raise Skip(f"debug calibration failed: {output[-300:]}")
    return json.loads((pilot / "results.json").read_text())


def failing(outcomes: dict) -> dict:
    return {test: o for test, o in outcomes.items() if o["outcome"] != "passed"}


def choose_bugs(
    seed: Seed, normalized: dict[str, str], count: int, work: Path
) -> tuple[list, dict]:
    sites = mutation_sites(normalized)
    random.Random(0).shuffle(sites)
    singles = {f"single_{i:02d}": [site] for i, site in enumerate(sites[:SINGLE_MUTANTS])}
    results = run_variants(
        seed,
        {"reference": normalized} | {n: apply_bugs(normalized, b) for n, b in singles.items()},
        work,
    )
    if failing(results.get("reference", {})) or not results.get("reference"):
        raise Skip("reference fails its hidden tests after normalization")
    # A bug that hangs or exhausts memory makes a poor task: verification would just time out.
    killed = [
        singles[n][0]
        for n in singles
        if failing(results.get(n, {})) and "<run>" not in results.get(n, {})
    ]
    rng = random.Random(1)
    combos = []
    for _ in range(COMBO_ATTEMPTS * 10):
        pick = rng.sample(killed, min(count, len(killed)))
        if len(pick) == count and len({(p, f) for p, _, f in pick}) == count and pick not in combos:
            combos.append(pick)
        if len(combos) == COMBO_ATTEMPTS:
            break
    if not combos:
        raise Skip(f"not enough distinct killable bugs ({len(killed)} killed)")
    variants = {}
    for c, combo in enumerate(combos):
        variants[f"combo{c}_all"] = apply_bugs(normalized, combo)
        for i in range(count):
            variants[f"combo{c}_drop{i}"] = apply_bugs(normalized, combo[:i] + combo[i + 1 :])
    results = run_variants(seed, variants, work)
    valid = []
    for c, combo in enumerate(combos):
        outcomes = results.get(f"combo{c}_all", {})
        all_fail = failing(outcomes)
        independent = all(failing(results.get(f"combo{c}_drop{i}", {})) for i in range(count))
        visible = count <= len(all_fail) < len(outcomes)  # some signal, but not everything broken
        if independent and visible and "<run>" not in outcomes:
            valid.append((len(all_fail), c, combo, all_fail))
    if not valid:
        raise Skip("no independent bug combination")
    # Fewest failing tests: subtle bugs make a harder task than one that breaks everything.
    _, _, combo, all_fail = min(valid, key=lambda v: (v[0], v[1]))
    return combo, all_fail


# ---------------------------------------------------------------- writing and checking variants


def task_toml(seed: Seed, variant: str, description: str, extra: dict, pin_image: bool) -> str:
    data = copy.deepcopy(seed.toml)
    data.pop("difficulty", None)
    data["task"]["name"] = f"harder-pilot-{variant}/{seed.task_id}-{variant}"
    data["task"]["description"] = description[:200]
    data["task"]["keywords"] = ["harder-pilot", variant]
    metadata = {
        k: v
        for k, v in data["metadata"].items()
        if not k.startswith("solve_rate") and k != "family"
    }
    data["metadata"] = {
        "family": variant,
        "seed_task": seed.task_id,
        "seed_solve_rate_lfm": seed.toml["metadata"].get("solve_rate"),
        **metadata,
        **extra,
        "generated_by": llm.MODEL,
    }
    data["metadata"]["repo2env"] = {  # what `repo2rlenv validate` and push read
        "spec_version": "0.2.0",
        "pipeline": "trajectory_to_env",
        "recipe": "harder_pilot",  # solve.sh restores files; there is no patch.diff
        "reward_kinds": ["test_execution"],
        "reproducibility": {"mode": "inline_dockerfile"},
        "family": variant,
        "source_trace": str(seed.toml["metadata"].get("source_trace", "")),
    }
    if not pin_image:
        data["environment"].pop("docker_image", None)
    return tomli_w.dumps({"version": "1.0", **data})


def write_task(
    task_dir: Path,
    seed: Seed,
    instruction: str,
    toml: str,
    behavior_files: dict | None,
    total: int,
    workspace: dict[str, str] | None,
) -> None:
    shutil.rmtree(task_dir, ignore_errors=True)
    dockerfile = (seed.dir / "environment" / "Dockerfile").read_text().rstrip() + "\n"
    files = {
        "instruction.md": instruction,
        "task.toml": toml,
        "tests/test_hidden.py": (seed.dir / "tests" / "test_hidden.py").read_text(),
        "tests/test.sh": TEST_SH.replace("TOTAL", str(total)),
        "solution/solve.sh": SOLVE_SH,
        **{f"solution/files/{p}": c for p, c in seed.reference.items()},
    }
    if behavior_files:
        files |= {f"tests/behavior/{n}": c for n, c in behavior_files.items()}
    if workspace is not None:
        dockerfile += "COPY workspace/ /workspace/\n"
        files |= {f"environment/workspace/{p}": c for p, c in workspace.items()}
    files["environment/Dockerfile"] = dockerfile
    write_files(task_dir, files)


def check_task(task_dir: Path, image: str, do_nothing_must_be_zero: bool) -> dict:
    rewards = run_trials(
        task_dir,
        image,
        [Trial("reference", "bash /solution/solve.sh", "1.0000"), Trial("nothing", "true", "")],
        {},
        {},
    )
    ok_reference = rewards["reference"] == "1.0000"
    try:
        nothing = float(rewards["nothing"])
    except ValueError:
        nothing = None
    ok_nothing = nothing is not None and (
        nothing == 0.0 if do_nothing_must_be_zero else nothing < 1.0
    )
    if not (ok_reference and ok_nothing):
        raise Skip(f"trials failed: {rewards}")
    return rewards


def make_variants_for(seed: Seed, out: Path, bugs: int, work: Path) -> list[dict]:
    records = []
    if error := docker.build_image(seed.image, seed.dir / "environment"):
        raise Skip(f"seed image build failed: {error}")

    behavior_files, behavior_checks = None, 0
    try:
        scenarios, expected, shown = behavior(seed, work)
        behavior_files = {
            "behavior.py": (ASSETS / "behavior.py").read_text(),
            "scenarios.py": scenarios,
            "expected.json": json.dumps(expected),
        }
        behavior_checks = len(expected)
    except Skip as skip:
        records.append({"variant": "re-*", "kept": False, "reason": str(skip)})
        shown = None

    total = seed.hidden_tests + behavior_checks
    files = "\n".join(f"- `{p}`" for p in seed.reference)
    title = seed.request.split(". ")[0].strip().rstrip(".")
    for variant in ("re-traces", "re-blind") if shown else ():
        blind = variant == "re-blind"
        instruction = RE_INSTRUCTION.format(
            what="its file layout, public names" if blind else "its file layout, interface",
            title="" if blind else f"\n## What it is\n\n{title}.\n",
            files=files,
            interface=public_names(seed) if blind else interface_code(seed),
            trace=render_trace(shown, TRACE_BUDGET[variant]),
        )
        task_dir = out / "tasks" / f"{seed.task_id}-{variant}"
        extra = {"hidden_tests": seed.hidden_tests, "behavior_checks": behavior_checks}
        write_task(
            task_dir,
            seed,
            instruction,
            task_toml(seed, variant, f"Reimplement from behavior: {title}", extra, True),
            behavior_files,
            total,
            None,
        )
        try:
            rewards = check_task(task_dir, seed.image, do_nothing_must_be_zero=True)
            records.append(
                {"variant": variant, "kept": True, "task": task_dir.name, "trials": rewards}
            )
        except Skip as skip:
            shutil.rmtree(task_dir, ignore_errors=True)
            records.append({"variant": variant, "kept": False, "reason": str(skip)})

    task_dir = out / "tasks" / f"{seed.task_id}-debug"
    try:
        normalized = {
            p: ast.unparse(ast.parse(c)) if p.endswith(".py") else c
            for p, c in seed.reference.items()
        }
        combo, symptoms = choose_bugs(seed, normalized, bugs, work)
        evidence = "\n".join(f"- {test}: {o['message']}" for test, o in symptoms.items())
        reply = llm.ask(
            BUG_REPORT_PROMPT,
            f"<specification>\n{seed.request}\n\n{seed.contract}\n</specification>\n\n<failing_checks>\n{evidence}\n</failing_checks>",
            BugReport,
        )
        if reply is None:
            raise Skip("llm_declined (bug report)")
        instruction = DEBUG_INSTRUCTION.format(
            report=reply.report.strip(), request=seed.request, contract=seed.contract
        )
        extra = {
            "hidden_tests": seed.hidden_tests,
            "behavior_checks": behavior_checks,
            "bugs": [{"file": p, "function": f} for p, _, f in combo],
        }
        write_task(
            task_dir,
            seed,
            instruction,
            task_toml(seed, "debug", f"Fix the bugs: {title}", extra, False),
            behavior_files,
            total,
            apply_bugs(normalized, combo),
        )
        image = f"harder-pilot-{seed.task_id.lower()}-debug"
        if error := docker.build_image(image, task_dir / "environment"):
            raise Skip(f"debug image build failed: {error}")
        rewards = check_task(task_dir, image, do_nothing_must_be_zero=False)
        records.append(
            {
                "variant": "debug",
                "kept": True,
                "task": task_dir.name,
                "trials": rewards,
                "failing_hidden_tests": len(symptoms),
            }
        )
    except Skip as skip:
        shutil.rmtree(task_dir, ignore_errors=True)
        records.append({"variant": "debug", "kept": False, "reason": str(skip)})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", nargs="+", required=True, help="traj2envs-rl task ids")
    parser.add_argument("--out", type=Path, default=Path("workspace/harder-pilot"))
    parser.add_argument("--llm", default="huggingface/mistralai/Mistral-Large-4-1T-A52B:openrouter")
    parser.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--max-spend-usd", type=float, default=2.0, help="0 = unlimited")
    parser.add_argument("--bugs", type=int, default=3, help="bugs injected per debug task")
    parser.add_argument("--platform", default="linux/amd64", help="'native' builds for the host")
    args = parser.parse_args()

    provider, model = args.llm.split("/", 1)
    llm.configure(
        LLMSpec(provider=provider, model=model),
        reasoning_effort=args.reasoning_effort,
        cache_dir=args.out / "llm-cache",
        budget_usd=args.max_spend_usd or None,
    )
    docker.PLATFORM = args.platform
    report = []
    for task_id in args.seeds:
        console.info(f"seed {task_id}")
        with tempfile.TemporaryDirectory(prefix="harder-pilot-") as work:
            try:
                seed = Seed(task_id, download(task_id, args.out / "seeds"))
                records = make_variants_for(seed, args.out, args.bugs, Path(work))
            except (Skip, llm.BudgetExceeded) as skip:
                records = [{"variant": "*", "kept": False, "reason": str(skip)}]
        for record in records:
            status = "kept" if record["kept"] else "drop"
            console.dim(
                f"{status}  {record['variant']:<10} {record.get('reason', record.get('trials', ''))}"
            )
            report.append({"seed": task_id, **record})
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    kept = sum(r["kept"] for r in report)
    console.success(
        f"{kept} variants kept; LLM spend ${llm.spent_usd():.2f}; report: {args.out / 'report.json'}"
    )


if __name__ == "__main__":
    main()
