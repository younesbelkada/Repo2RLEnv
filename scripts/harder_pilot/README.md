# Harder variants of traj2envs-rl tasks (pilot)

Derives harder tasks from published [traj2envs-rl](https://huggingface.co/datasets/harness-hub/traj2envs-rl)
tasks and compares a model's results on the seeds and on their variants. A pilot script, not a
pipeline yet.

For each seed task it writes up to three variants, graded by the seed's hidden tests plus
behavior checks: scenarios rerun on unseen inputs and compared with the original's recorded
behavior.

| Variant | The agent gets | It must |
|---|---|---|
| `re-traces` | file layout, interface, title, ~50 logged calls with results | reimplement the module from scratch |
| `re-blind` | file layout, public names only, ~30 logged calls | the same, with less to go on |
| `debug` | the reference with 3 injected bugs, the spec, a user bug report | find and fix all bugs |

`debug` bugs are AST mutations (`pipelines/trajectory/mutants.py`). Each one is caught by at least
one hidden test, and they're independent: fixing any two still leaves failures. Among valid
combinations, the one with the fewest failing tests (at least 3) is chosen.

Every variant is checked in Docker before it is kept: the reference scores 1.0; doing nothing
scores 0 (`re-*`) or below 1 (`debug`). `report.json` records what was kept and why not.

## On the VM

The seed images copy `opencode` from an `linux/amd64`-only image, so run on x86 (or pass
`--platform linux/amd64` under emulation elsewhere).

```bash
uv sync && uv tool install harbor
export HF_TOKEN=hf_...   # Inference Providers permission: teacher LLM and Qwen
```

### 1. Baseline: Qwen on the seeds and 5 more tasks

```bash
SEEDS="01a0338e-5d82-73f4-bdb4-build 01a03450-30ea-7347-ac9b-build 01a031bf-cac4-74a0-926b-build \
       01a0320b-3ff4-71be-8d15-build 01a032ad-0592-71df-8f09-build"
EXTRA="01a03246-1db3-72f4-a7f0-build 01a034c0-feb5-76b6-b20a-build 01a03335-ea6e-74c6-8f71-build \
       01a03424-3604-704c-bb4a-build 01a034ae-614d-77ad-8d4a-build"
for id in $SEEDS $EXTRA; do
  uv run hf download harness-hub/traj2envs-rl --repo-type dataset \
    --include "tasks/$id/*" --local-dir workspace/harder-pilot/baseline
done

harbor run -p workspace/harder-pilot/baseline/tasks -a <agent> \
  -m openai/<qwen-model-id> --base-url https://router.huggingface.co/v1 \
  --api-key-name HF_TOKEN --api-format openai_chat_completions \
  --env docker -k 4 -n 8 -o workspace/harder-pilot/jobs --job-name baseline -y
```

Use the same agent and sampling as the model card (pi, temperature 0.8). For `terminus-2`, add
`--ak record_terminal_session=false` (tasks are offline).

### 2. Generate the variants

```bash
uv run python scripts/harder_pilot/make_variants.py --seeds $SEEDS \
  --out workspace/harder-pilot --max-spend-usd 2
```

About two teacher calls per seed (scenarios, bug report), cached in
`workspace/harder-pilot/llm-cache`, so a rerun is free. Tasks land in
`workspace/harder-pilot/tasks/`.

### 3. Check them with Harbor

```bash
uv run repo2rlenv validate workspace/harder-pilot/tasks --deep --oracle
harbor run -p workspace/harder-pilot/tasks -a oracle --env docker -n 8 \
  -o workspace/harder-pilot/jobs --job-name oracle -y      # every task should score 1.0
```

### 4. Qwen on the variants, then compare

```bash
harbor run -p workspace/harder-pilot/tasks -a <agent> <same model flags as step 1> \
  --env docker -k 4 -n 8 -o workspace/harder-pilot/jobs --job-name variants -y

uv run python scripts/harder_pilot/score.py \
  workspace/harder-pilot/jobs/baseline workspace/harder-pilot/jobs/variants
```

`score.py` prints, per seed, the solved rate (reward 1.0, as in the model card) and the mean
reward for the seed and each variant, then the averages per variant type.
