# RL training: GRPO for p5.brush watercolour code

RL-trains a Qwen model (on the Tinker service) to write p5.brush JavaScript
sketches that render into good watercolour paintings of the prompted animal/scene.
Recreates the ["RLing Qwen to paint with code"](https://surya.website/rling-qwen-to-paint-with-code)
experiment over the 10-animal × 3-scene dataset in `datagen/prompts.json`.

## Pipeline

```
prompt -> Qwen (policy on Tinker) --group of G rollouts--> G p5.brush sketches
       -> render/worker.js (headless Chrome) -> PNG each
       -> reward: compile gate + length penalty      (Env.step)
                + holistic aesthetic judge + pairwise judge vs references
                                                       (compute_group_rewards)
       -> GRPO: center rewards within each group -> forward_backward -> optim_step
```

## Reward

| term | where | signal |
|---|---|---|
| compile gate | `Env.step` | render fail / no-code / no stop-seq → reward = `r_fail` (−0.3), no judge spend |
| blank gate | `Env.step` | flat/near-uniform render (`reward/imgcheck.py`, painted fraction < 0.02) → `r_blank` (−0.1), excluded from all judging |
| length penalty | `Env.step` | 0 up to `len_free_tokens` (6000), linear to `−w_len` (−0.30) at `max_tokens` |
| aesthetic | `compute_group_rewards` | one holistic VLM judge call/image → 0–10 → [0,1], weight `w_aesth` (0.35); prompt anchors the low end (blank=0, crude shapes 1–2, …) |
| pairwise | `compute_group_rewards` | winrate vs K refpool opponents, order-swapped, weight `w_pair` (0.15) |
| tournament | `compute_group_rewards` | optional (`tournament=True`): winrate vs `tournament_k` non-blank group-mates, weight `w_tour` (0.30, independent of `w_pair`) |

Pairwise votes are **forced choice** (A or B, no ties — run 2's judge tied ~every
intra-group pair, flattening the tournament to a constant 0.5) with an explicit
dominance rule: an image that attempts the subject always beats a blank/near-empty
canvas, however crude.

`total = step_reward + group_reward`. All knobs: `reward/config.py` (`RewardConfig`),
surfaced as `train_rl.py` / `PaintRLDatasetBuilder` fields.

Run 3 (blank gate worked, best aesthetic yet, then a NEW collapse: the policy
locked onto one buggy code pattern — `Brush "color" not found` in 48/64 rollouts —
and compile crashed) added: thinking back ON (`renderer_name=qwen3_5`; run 1 with
thinking had ~2× the aesthetic of any thinking-off run), default LR cut to 1.5e-4
(~1/3 of get_lr; lock-in happened within ~3 steps at 4.7e-4), and a small KL
anchor to the base model (`kl_penalty_coef=0.01`). Token budget deliberately
unchanged (8k cap, 6000 free) — watch `env/all/format_ok` for thinking-driven
truncation.

Run 2 (same collapse, slower) added: the blank gate (validated against run 2's own
rollouts: 100% of the collapsed tail flagged, 0 false positives on references),
no-tie judging with the dominance rule, low-end holistic anchors, w_pair 0.45→0.15
(refpool winrate was 0.000 the whole run), w_tour 0.20→0.30, and a trimmed system
prompt toolbox (wash/hatch/mass/beginStroke/clip/add removed — smaller error
surface; fill family kept).

Shaping notes (post-mortem of the first 9B run, which collapsed to blank canvases):
`r_fail=-0.3` equals the worst possible success score so failing never beats
succeeding but no longer dwarfs the judge signal; thinking is disabled by default
(`renderer_name=qwen3_5_disable_thinking`) so an 8k `max_tokens` fits whole
sketches; judge calls retry (`judge_retries`) and a still-failed call scores **0**,
not a neutral 0.5 (which out-scored honest attempts); default judge is
`qwen/qwen3-vl-32b-instruct` (~20× cheaper and ~2× faster than gemini-3.7-flash,
sane scores); the tournament gives gradient while refpool winrate sits at 0.

## Files

- `train/system_prompt.txt` — barebones prompt (harness contract + method list only).
- `train/env.py` — `PaintEnv`, `PaintEnvGroupBuilder`, `PaintRLDataset`, `PaintRLDatasetBuilder`.
- `train/train_rl.py` — CLI driver → `tinker_cookbook.rl.train.main`.
- `train/eval_grid.py` — offline before/after eval (5-criterion GEPA judge, eval-only).
- `reward/` — render bridge, judges, scorer, refpool, reward math, cost tracker (importable without tinker_cookbook).
- `render/worker.js` — persistent JSON-lines render worker over `renderSketch`.
- `refpool/combos/<id>/*.png`, `refpool/animals/<slug>/*.png` — reference images (fill from datagen).
- `<log_path>/rollouts/iteration_%06d/{code,renders}/` — per-iteration rollout artifacts
  (matching stems: `<combo>_<hash>.js` ↔ `.png`; code saved even when the render fails);
  held-out eval rollouts land in `<log_path>/rollouts/eval/{code,renders}/`.

## Run

```bash
# smoke: 3 steps, tiny groups, NO judge spend (NullScorer + pairwise off)
.venv/Scripts/python.exe -m train.train_rl smoke=True log_path=runs/smoke1 behavior_if_log_dir_exists=raise

# MVP: holistic + pairwise (needs refpool/ filled, else pairwise_mode error)
.venv/Scripts/python.exe -m train.train_rl log_path=runs/mvp1 \
    group_size=8 groups_per_batch=8 max_tokens=8192 max_steps=50

# with the intra-group tournament term (adds w_tour=0.20 on top of 0.35 aesth + 0.45 refpool):
.venv/Scripts/python.exe -m train.train_rl log_path=runs/mvp1 tournament=True

# start without references yet (aesthetic-only reward):
.venv/Scripts/python.exe -m train.train_rl log_path=runs/mvp0 pairwise_mode=off

# resume: re-run with the SAME log_path (checkpoints + tracker.json continue)

# eval a checkpoint vs the base model:
.venv/Scripts/python.exe -m train.eval_grid model_name=Qwen/Qwen3.5-4B out=runs/eval/base
```

Keys: OpenRouter `OPENROUTER_KEY` and Tinker `TINKER_KEY` (bridged to `TINKER_API_KEY`)
are read from the project `.env`. Cost/time is tracked to `<log_path>/tracker.json`.

## Tests (no Tinker)

```bash
.venv/Scripts/python.exe reward/tests/test_render_bridge.py    # Phase 1: render worker
.venv/Scripts/python.exe train/tests/test_env_offline.py       # Phase 3: env gate/reward
```
