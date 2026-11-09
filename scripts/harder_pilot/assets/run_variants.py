"""Run the hidden tests against code variants inside a task image (network off). Stdlib only.

/pilot/variants/<name>/   one directory of project files per variant
/pilot/test_hidden.py     the seed task's hidden tests
Writes /pilot/results.json: {variant: {test: {"outcome": ..., "message": ...}}}

A mutant can loop forever or allocate without bound. Each variant's pytest gets a memory cap and
a timeout, so only that variant fails (an empty result), not the whole run; results are written
after every variant.
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PILOT = Path("/pilot")
MEMORY_BYTES = 1 << 30
VARIANT_TIMEOUT_SECONDS = 300
workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
results = {}


def limit_memory() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))


for variant in sorted((PILOT / "variants").iterdir()):
    shutil.rmtree(workspace, ignore_errors=True)
    shutil.copytree(variant, workspace)
    report = Path(f"/tmp/{variant.name}.xml")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--timeout=30",
                f"--junitxml={report}",
                str(PILOT / "test_hidden.py"),
            ],
            cwd="/tmp",
            capture_output=True,
            timeout=VARIANT_TIMEOUT_SECONDS,
            preexec_fn=limit_memory,
        )
    except subprocess.TimeoutExpired:
        pass
    outcomes = {}
    if report.exists():
        for case in ET.parse(report).getroot().iter("testcase"):
            problem = next((c for c in case if c.tag in ("failure", "error", "skipped")), None)
            outcomes[case.get("name")] = {
                "outcome": problem.tag if problem is not None else "passed",
                "message": (
                    (problem.get("message") or problem.text or "")[:400]
                    if problem is not None
                    else ""
                ),
            }
    results[variant.name] = outcomes or {
        "<run>": {"outcome": "error", "message": "crashed or timed out"}
    }
    (PILOT / "results.json").write_text(json.dumps(results, indent=1))
