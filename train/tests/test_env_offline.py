"""Phase 3 offline test for train/env.py (no Tinker service).

Exercises the full Env.step gate/render path and compute_group_rewards with a
synthetic action (a known-good sketch encoded as raw completion tokens + the stop
token). Uses NullScorer + pairwise 'off' so it costs nothing.

Run:  .venv/Scripts/python.exe train/tests/test_env_offline.py
"""

import asyncio
import glob
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tinker_cookbook.renderers import get_renderer  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

from reward.config import RewardConfig  # noqa: E402
from reward.render_bridge import close_render_bridge  # noqa: E402
from train.env import PaintEnv, PaintEnvGroupBuilder  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"
RENDERER = "qwen3_5"
COMBO = {"id": "koi_pond_sunny", "animal": "koi fish",
         "prompt": "koi fish in a small pond on a sunny day"}
SYS = (ROOT / "train" / "system_prompt.txt").read_text(encoding="utf-8")
OUT = ROOT / "render" / "out" / "env_test"
OUT.mkdir(parents=True, exist_ok=True)


async def main():
    tk = get_tokenizer(MODEL)
    renderer = get_renderer(RENDERER, tk)
    stop_id = renderer.get_stop_sequences()[0]

    good_path = sorted(glob.glob(str(ROOT / "datagen" / "out" / "*" / "sketches" / "*.js")))[0]
    good_code = Path(good_path).read_text(encoding="utf-8")
    good_action = tk.encode(good_code, add_special_tokens=False) + [stop_id]
    # a "bad" action: text that yields no code and no stop token (format gate fail)
    bad_action = tk.encode("I cannot help with that.", add_special_tokens=False)

    cfg = RewardConfig(scorer="null", pairwise_mode="off", max_tokens=16384)
    ok_all = True

    # initial observation shape
    env = PaintEnv(renderer, COMBO, SYS, cfg, str(OUT))
    obs, stop = await env.initial_observation()
    print("[initial_observation] prompt tokens=%d  stop=%s" % (obs.length, stop))

    # good rollout
    res = await env.step(good_action)
    print("[good step] compiled=%s reward=%.3f render_ms=%d png=%s"
          % (env.compiled, res.reward, env.render_ms, env.png_path))
    if not env.compiled:
        print("   !! expected compiled=True; render_error=%s" % env.render_error)
        ok_all = False

    # bad rollout (no stop sequence -> gate fail -> r_fail)
    env_bad = PaintEnv(renderer, COMBO, SYS, cfg, str(OUT))
    res_bad = await env_bad.step(bad_action)
    print("[bad step] compiled=%s reward=%.3f (expect r_fail=%.1f)"
          % (env_bad.compiled, res_bad.reward, cfg.r_fail))
    if env_bad.compiled or res_bad.reward != cfg.r_fail:
        print("   !! expected gate fail with r_fail")
        ok_all = False

    # the good render must not trip the blank-image gate
    print("[blank gate] good render blank=%s" % env.blank)
    if env.blank:
        print("   !! good sketch flagged as blank")
        ok_all = False

    # a blank env (flat render) must be excluded from judging: group reward (0.0, {})
    env_blank = PaintEnv(renderer, COMBO, SYS, cfg, str(OUT))
    env_blank.compiled, env_blank.blank, env_blank.png_path = True, True, env.png_path

    # compute_group_rewards over the three envs (null scorer, pairwise off)
    builder = PaintEnvGroupBuilder(
        renderer=renderer, combo=COMBO, system_prompt=SYS, reward_cfg=cfg,
        group_size=3, out_dir=str(OUT), refpool_root=str(ROOT / "refpool"), seed=0,
    )
    rewards = await builder.compute_group_rewards([None] * 3, [env, env_bad, env_blank])
    print("[group rewards] %s" % rewards)
    # compiled env gets w_aesth*0.5; failed AND blank envs get (0.0, {})
    if (abs(rewards[0][0] - cfg.w_aesth * 0.5) > 1e-6
            or rewards[1] != (0.0, {}) or rewards[2] != (0.0, {})):
        print("   !! unexpected group rewards")
        ok_all = False

    # group_reward weight math (pure): independent weights per term
    from reward.reward import group_reward as gr_fn
    full = gr_fn(0.6, 0.4, cfg)                          # no tournament
    split = gr_fn(0.6, 0.4, cfg, tournament_winrate=0.5)  # with tournament
    want_full = cfg.w_aesth * 0.6 + cfg.w_pair * 0.4
    want_split = want_full + cfg.w_tour * 0.5
    print("[group_reward math] full=%.4f split=%.4f" % (full, split))
    if abs(full - want_full) > 1e-9 or abs(split - want_split) > 1e-9:
        print("   !! group_reward weight math wrong")
        ok_all = False

    # code file saved next to the render, under code/ + renders/ subdirs
    code_files = list((OUT / "code").glob("%s_*.js" % COMBO["id"]))
    png_files = list((OUT / "renders").glob("%s_*.png" % COMBO["id"]))
    print("[artifacts] %d code file(s), %d render(s) under %s" % (len(code_files), len(png_files), OUT))
    if not code_files or not png_files:
        print("   !! expected code/ and renders/ artifacts")
        ok_all = False

    # pickle-roundtrip the builder (must be picklable for distributed rollouts)
    try:
        pickle.loads(pickle.dumps(builder))
        print("[pickle] builder round-trips OK")
    except Exception as ex:
        print("   !! builder not picklable: %s" % ex)
        ok_all = False

    await close_render_bridge()
    print("\nRESULT:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
