"""Record and compare a module's observable behavior. Standard library only; runs inside task images.

A scenario is `def scenario_<name>(api, rng)`: it gets modules through `api.mod("pkg.module")`
and calls them with inputs drawn from `rng`. Every call made through `api` is logged with the
value it returned or the exception type it raised. `api.note(value)` logs anything else.

  record  python behavior.py record --root DIR --scenarios FILE --seeds 0,1,2 --out log.json
  check   python behavior.py check --root DIR --scenarios FILE --expected log.json --junit out.xml

`check` runs each scenario on the implementation under --root with the seeds in the expected
log, and passes a scenario only if every seed's log matches the reference's. Each scenario runs
in its own subprocess with a timeout, so module state and hangs can't leak between scenarios.
"""

from __future__ import annotations

import argparse
import collections.abc
import importlib
import importlib.util
import json
import math
import os
import random
import resource
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SCENARIO_TIMEOUT_SECONDS = 60
MEMORY_BYTES = (
    2 << 30
)  # per scenario process, so a runaway implementation can't take the verifier down
MAX_DISPLAY = 300
MAX_DEPTH = 6


def normalize(value, tracked: tuple[str, ...], depth: int = 0):
    """A JSON value that two equivalent implementations produce identically."""
    if depth > MAX_DEPTH:
        return "<deep>"
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return "nan" if math.isnan(value) else round(value, 9)
    if isinstance(value, bytes | bytearray):
        return {"__bytes__": bytes(value).hex()}
    if isinstance(value, list | tuple):
        return [normalize(item, tracked, depth + 1) for item in value]
    if isinstance(value, dict):
        items = [
            [normalize(k, tracked, depth + 1), normalize(v, tracked, depth + 1)]
            for k, v in value.items()
        ]
        return {"__dict__": sorted(items, key=repr)}
    if isinstance(value, set | frozenset):
        return {
            "__set__": sorted((normalize(item, tracked, depth + 1) for item in value), key=repr)
        }
    if isinstance(value, BaseException):
        return {"__raise__": type(value).__name__, "attrs": public_attrs(value, tracked, depth)}
    if type(value).__module__.split(".")[0] in tracked:
        return {"__obj__": type(value).__name__, "attrs": public_attrs(value, tracked, depth)}
    return {"__repr__": type(value).__name__}


def public_attrs(value, tracked, depth) -> dict:
    attrs = getattr(value, "__dict__", None) or {}
    return {
        k: normalize(v, tracked, depth + 1)
        for k, v in sorted(attrs.items())
        if not k.startswith("_") and k != "args"
    }


def matches(reference, candidate) -> bool:
    """Equality, except objects and exceptions only need the reference's public attributes."""
    if isinstance(reference, dict) and ("__obj__" in reference or "__raise__" in reference):
        tag = "__obj__" if "__obj__" in reference else "__raise__"
        return (
            isinstance(candidate, dict)
            and candidate.get(tag) == reference[tag]
            and all(
                k in candidate.get("attrs", {}) and matches(v, candidate["attrs"][k])
                for k, v in reference["attrs"].items()
            )
        )
    if isinstance(reference, list):
        return (
            isinstance(candidate, list)
            and len(candidate) == len(reference)
            and all(matches(r, c) for r, c in zip(reference, candidate, strict=True))
        )
    if isinstance(reference, dict):
        return (
            isinstance(candidate, dict)
            and reference.keys() == candidate.keys()
            and all(matches(reference[k], candidate[k]) for k in reference)
        )
    return reference == candidate


def display(value) -> str:
    text = value._handle if isinstance(value, ObjectProxy | IteratorProxy) else repr(value)
    return text if len(text) <= MAX_DISPLAY else text[:MAX_DISPLAY] + "…"


class Api:
    def __init__(self, tracked: tuple[str, ...]):
        self.log: list[dict] = []
        self.tracked = tracked
        self.handles = 0

    def mod(self, name: str) -> ModuleProxy:
        return ModuleProxy(self, name, importlib.import_module(name))

    def note(self, value) -> None:
        self.log.append(
            {"call": "note", "shown": display(value), "result": normalize(value, self.tracked)}
        )

    def wrap_result(self, value):
        # A returned iterator (e.g. a generator) is logged item by item as it is consumed: its repr
        # holds a memory address, and laziness is behavior too (mutating while iterating).
        if isinstance(value, collections.abc.Iterator) and not isinstance(value, BaseException):
            self.handles += 1
            return IteratorProxy(self, f"iterator{self.handles}", value)
        if (
            type(value).__module__.split(".")[0] in self.tracked
            and not isinstance(value, BaseException | type)
            and not callable(value)
        ):
            self.handles += 1
            return ObjectProxy(self, f"{type(value).__name__.lower()}{self.handles}", value)
        return value

    def call(self, label: str, function, args, kwargs):
        proxies = ObjectProxy | IteratorProxy
        raw_args = [a._target if isinstance(a, proxies) else a for a in args]
        raw_kwargs = {k: v._target if isinstance(v, proxies) else v for k, v in kwargs.items()}
        shown_args = ", ".join(
            [display(a) for a in args] + [f"{k}={display(v)}" for k, v in kwargs.items()]
        )
        entry = {"call": f"{label}({shown_args})"}
        self.log.append(entry)
        try:
            result = function(*raw_args, **raw_kwargs)
        except Exception as error:
            entry["result"] = normalize(error, self.tracked)
            attrs = entry["result"]["attrs"]
            detail = ", ".join(f"{k}={v!r}" for k, v in attrs.items())
            entry["shown"] = f"raises {type(error).__name__}" + (f" ({detail})" if detail else "")
            raise
        wrapped = self.wrap_result(result)
        if isinstance(wrapped, IteratorProxy):
            entry["result"] = {"__iterator__": True}
            entry["shown"] = f"-> {wrapped._handle} (an iterator)"
        elif isinstance(wrapped, ObjectProxy):
            entry["result"] = {"__handle__": type(result).__name__}
            entry["shown"] = f"-> {wrapped._handle} (a {type(result).__name__})"
        else:
            entry["result"] = normalize(result, self.tracked)
            entry["shown"] = display(result)
        return wrapped


class CallableProxy:
    def __init__(self, api: Api, label: str, target):
        self._api, self._label, self._target = api, label, target

    def __call__(self, *args, **kwargs):
        return self._api.call(self._label, self._target, args, kwargs)


class ModuleProxy:
    def __init__(self, api: Api, name: str, module):
        self._api, self._name, self._module = api, name, module

    def __getattr__(self, attr):
        value = getattr(self._module, attr)
        if callable(value):
            return CallableProxy(self._api, f"{self._name}.{attr}", value)
        self._api.log.append(
            {
                "call": f"{self._name}.{attr}",
                "shown": display(value),
                "result": normalize(value, self._api.tracked),
            }
        )
        return value


class ObjectProxy:
    def __init__(self, api: Api, handle: str, target):
        self._api, self._handle, self._target = api, handle, target

    def __getattr__(self, attr):
        value = getattr(self._target, attr)
        if callable(value):
            return CallableProxy(self._api, f"{self._handle}.{attr}", value)
        self._api.log.append(
            {
                "call": f"{self._handle}.{attr}",
                "shown": display(value),
                "result": normalize(value, self._api.tracked),
            }
        )
        return self._api.wrap_result(value)

    def __len__(self):
        return self._api.call(f"len({self._handle})", lambda: len(self._target), (), {})

    def __iter__(self):
        return iter(self._api.call(f"list({self._handle})", lambda: list(self._target), (), {}))

    def __getitem__(self, key):
        return self._api.call(f"{self._handle}.__getitem__", self._target.__getitem__, (key,), {})

    def __contains__(self, item):
        return self._api.call(
            f"{self._handle}.__contains__", self._target.__contains__, (item,), {}
        )


class IteratorProxy:
    def __init__(self, api: Api, handle: str, target):
        self._api, self._handle, self._target = api, handle, target

    def __iter__(self):
        return self

    def __next__(self):
        # StopIteration is logged like any exception ("raises StopIteration"), then re-raised.
        return self._api.call(f"{self._handle}.__next__", lambda: next(self._target), (), {})


def load_scenarios(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("scenarios", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        name.removeprefix("scenario_"): function
        for name, function in vars(module).items()
        if name.startswith("scenario_") and callable(function)
    }


def run_one(root: Path, scenarios: Path, name: str, seeds: list[int]) -> dict:
    """In this process: run one scenario for each seed and return {seed: log}."""
    sys.path.insert(0, str(root))
    tracked = tuple(
        p.stem if p.is_file() else p.name
        for p in root.iterdir()
        if (p.suffix == ".py" and p.is_file()) or (p.is_dir() and (p / "__init__.py").exists())
    )
    function = load_scenarios(scenarios)[name]
    logs = {}
    for seed in seeds:
        # Fresh module state per seed (counters, registries), so a seed's log doesn't depend on
        # which seeds ran before it: recording runs shown + hidden seeds, checking only hidden ones.
        for module in [m for m in sys.modules if m.split(".")[0] in tracked]:
            del sys.modules[module]
        api = Api(tracked)
        try:
            function(api, random.Random(seed))
        except Exception as error:
            last = api.log[-1] if api.log else None
            if not (
                last and isinstance(last.get("result"), dict) and "__raise__" in last["result"]
            ):
                api.log.append(
                    {
                        "call": "scenario",
                        "shown": f"scenario error: {type(error).__name__}: {error}"[:MAX_DISPLAY],
                        "result": {"__error__": type(error).__name__},
                    }
                )
        logs[str(seed)] = api.log
    return logs


def run_isolated(root: Path, scenarios: Path, name: str, seeds: list[int]) -> dict | None:
    """Run one scenario in a fresh subprocess; None if it crashed or timed out."""
    command = [
        sys.executable,
        __file__,
        "one",
        "--root",
        str(root),
        "--scenarios",
        str(scenarios),
        "--name",
        name,
        "--seeds",
        ",".join(map(str, seeds)),
    ]
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=SCENARIO_TIMEOUT_SECONDS,
            # String hashing is randomized per process; fix it so set/dict order is reproducible.
            env={**os.environ, "PYTHONHASHSEED": "0"},
            preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES)),
        )
    except subprocess.TimeoutExpired:
        return None
    try:
        return json.loads(process.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return None


def record(args) -> None:
    seeds = [int(s) for s in args.seeds.split(",")]
    names = sorted(load_scenarios(args.scenarios))
    result = {name: run_isolated(args.root, args.scenarios, name, seeds) for name in names}
    args.out.write_text(json.dumps(result))


def first_difference(reference: list[dict], candidate: list[dict]) -> str:
    """The first call whose result differs, with what the original and the candidate returned."""
    for i, entry in enumerate(reference):
        if i >= len(candidate):
            return f"stopped before {entry.get('call', '?')} (expected {entry.get('shown', '?')})"
        if not matches(entry.get("result"), candidate[i].get("result")):
            return (
                f"{entry.get('call', '?')}: expected {entry.get('shown', '?')}, "
                f"got {candidate[i].get('shown', '?')}"
            )[: MAX_DISPLAY * 2]
    return f"made {len(candidate) - len(reference)} extra calls"


def check(args) -> None:
    expected = json.loads(args.expected.read_text())
    suite = ET.Element("testsuite", name="behavior")
    for name, by_seed in sorted(expected.items()):
        case = ET.SubElement(suite, "testcase", classname="behavior", name=name)
        got = run_isolated(args.root, args.scenarios, name, [int(s) for s in by_seed])
        mismatch = None
        if got is None:
            mismatch = "crashed or timed out"
        else:
            for seed, log in by_seed.items():
                reference = [e.get("result") for e in log]
                theirs = got.get(seed, [])
                candidate = [e.get("result") for e in theirs]
                if not matches(reference, candidate):
                    mismatch = f"seed {seed}: " + first_difference(log, theirs)
                    break
        if mismatch:
            ET.SubElement(case, "failure", message=mismatch)
    ET.ElementTree(suite).write(args.junit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["record", "check", "one"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--name")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--junit", type=Path)
    args = parser.parse_args()
    if args.mode == "one":
        logs = run_one(
            args.root, args.scenarios, args.name, [int(s) for s in args.seeds.split(",")]
        )
        print(json.dumps(logs))
    elif args.mode == "record":
        record(args)
    else:
        check(args)


if __name__ == "__main__":
    main()
