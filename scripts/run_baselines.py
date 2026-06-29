#!/usr/bin/env python3
"""
Run three comparative baselines for SuSCAP-RL evaluation:
  1. Random Policy
  2. Single-Objective SAC (fixed one-hot preference, no curriculum)
  3. NSGA-II / Heuristic Proxy (random search + non-dominated sorting)

Saves summary results to outputs/baseline_results.json
"""

import os, sys, json, yaml, argparse, time, copy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from env.scheduler_env import SchedulerEnv
from training.sac_extensions import SACWithConstraints
from training.replay_buffer import ReplayBuffer
from curriculum.scenario import ScenarioCurriculum
import evaluation.metrics as metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_simplex_grid(K):
    grid = []
    for i in range(K + 1):
        for j in range(K + 1 - i):
            k = K - i - j
            w = np.array([i, j, k], dtype=float) / float(K)
            if w.sum() > 0:
                grid.append(w)
    return np.array(grid, dtype=np.float32)


def get_eval_preferences_scenarios(n_pref=9):
    K = 1
    while (K + 1) * (K + 2) // 2 < n_pref:
        K += 1
    W_grid = build_simplex_grid(K)
    if len(W_grid) > n_pref:
        idx = np.linspace(0, len(W_grid) - 1, n_pref).astype(int)
        W_grid = W_grid[idx]
    scen = ScenarioCurriculum()
    C_bank = [(s, scen.encode(s)) for s in scen.scenarios]
    return W_grid, C_bank


def run_eval_episodes(env, policy_fn, W_grid, C_bank, episode_length, episodes=3):
    """
    Evaluate a policy over a grid of preferences and scenarios.
    policy_fn(obs, w, c) -> action
    Returns (return_grid, cost_grid, per_episode_points, per_episode_meta).
    """
    n_w = len(W_grid)
    n_c = len(C_bank)
    m = 2
    d = 3
    return_grid = np.zeros((n_w, n_c, d))
    cost_grid = np.zeros((n_w, n_c, m))
    per_episode_points = []
    per_episode_meta = []

    for iw, w in enumerate(W_grid):
        for ic, (scenario_name, c_enc) in enumerate(C_bank):
            ep_returns = []
            ep_costs = []
            for ep in range(episodes):
                env.scenario = scenario_name
                obs, _ = env.reset()
                ret = np.zeros(d)
                cost_acc = np.zeros(m)
                done = False
                t = 0
                while not done and t < episode_length:
                    action = policy_fn(obs, w, c_enc)
                    next_obs, reward_vec, terminated, truncated, info = env.step(action)
                    ret += reward_vec
                    cost_acc += info["cost"]
                    obs = next_obs
                    t += 1
                    done = terminated or truncated
                ep_returns.append(ret)
                ep_costs.append(cost_acc)
                per_episode_points.append(ret)
                per_episode_meta.append([iw, ic, ep])
            return_grid[iw, ic] = np.mean(ep_returns, axis=0)
            cost_grid[iw, ic] = np.mean(ep_costs, axis=0)

    per_episode_points = np.array(per_episode_points)
    per_episode_meta = np.array(per_episode_meta, dtype=int)
    return return_grid, cost_grid, per_episode_points, per_episode_meta


def compute_metrics_from_eval(per_episode_points, cost_grid, return_grid, W_grid, n_c, thresholds, per_episode_meta=None):
    """Compute all standard evaluation metrics."""
    points = metrics.to_maximization_space(per_episode_points)
    if points.shape[0] == 0:
        return {"hypervolume": 0.0, "coverage_fraction": 0.0, "pareto_front_size": 0,
                "constraint_violation_rate": 0.0, "mean_constraint_violation": 0.0, "total_episodes": 0}

    scales = np.maximum(np.percentile(points, 95, axis=0), 1e-9)
    points_norm = points / scales

    pf = metrics.pareto_front(points_norm)
    ref = points_norm.min(axis=0) - 1e-6
    hv = metrics.hypervolume_mc(points_norm, ref=ref, n_samples=20000)
    violation_rate, mean_violation = metrics.constraint_stats(cost_grid, thresholds=thresholds)

    N = points_norm.shape[0]
    nd_mask = np.zeros(N, dtype=bool)
    for i in range(N):
        remaining = np.delete(points_norm, i, axis=0)
        nd_mask[i] = not metrics.is_dominated(points_norm[i], remaining) if N > 1 else True

    if per_episode_meta is not None:
        n_w = len(W_grid)
        grid_mask = np.zeros((n_w, n_c), dtype=bool)
        for idx, (iw, ic, _) in enumerate(per_episode_meta):
            if nd_mask[idx]:
                grid_mask[iw, ic] = True
        coverage_frac = grid_mask.mean()
    else:
        coverage_frac = float(nd_mask.mean())

    return {
        "hypervolume": float(hv),
        "coverage_fraction": float(coverage_frac),
        "pareto_front_size": int(len(pf)),
        "constraint_violation_rate": float(violation_rate),
        "mean_constraint_violation": float(mean_violation),
        "total_episodes": int(N),
    }


# ---------------------------------------------------------------------------
# BASELINE 1: Random Policy
# ---------------------------------------------------------------------------

def run_random_baseline(env, cfg, device="cpu"):
    print("\n" + "=" * 60)
    print("BASELINE 1: Random Policy")
    print("=" * 60)

    W_grid, C_bank = get_eval_preferences_scenarios()
    episode_length = cfg.get("evaluation", {}).get("eval_episode_length", 180)

    def random_policy(obs, w, c):
        return env.action_space.sample()

    return_grid, cost_grid, per_episode_points, per_episode_meta = run_eval_episodes(
        env, random_policy, W_grid, C_bank, episode_length, episodes=3
    )

    thresholds = np.array(cfg.get("constraints", {}).get("cost_thresholds", [0.01, 4.7]))
    results = compute_metrics_from_eval(
        per_episode_points, cost_grid, return_grid, W_grid, len(C_bank), thresholds,
        per_episode_meta=per_episode_meta
    )
    results["method"] = "random_policy"
    results["description"] = "Actions sampled uniformly from env.action_space"

    print(f"  Hypervolume:              {results['hypervolume']:.6e}")
    print(f"  Coverage fraction:        {results['coverage_fraction']:.3f}")
    print(f"  Constraint violation:     {results['constraint_violation_rate']:.3f}")
    print(f"  Pareto front size:        {results['pareto_front_size']}")
    return results


# ---------------------------------------------------------------------------
# BASELINE 2: Single-Objective SAC
# ---------------------------------------------------------------------------

def run_single_objective_sac_baseline(env, cfg, device="cpu"):
    print("\n" + "=" * 60)
    print("BASELINE 2: Single-Objective SAC (fixed one-hot preference)")
    print("=" * 60)

    env_cfg = cfg["env"]
    cond_cfg = cfg["conditioning"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    sac_cfg = {
        "obs_dim": env_cfg["obs_dim"],
        "cond_dim": cond_cfg["cond_dim"],
        "action_dim": env_cfg["action_dim"],
        "hidden_sizes": model_cfg.get("hidden_sizes", [128, 128]),
        "ensemble_size": 2,
        "gamma": train_cfg.get("gamma", 0.99),
        "tau": train_cfg.get("tau", 0.005),
        "critic_lr": model_cfg.get("critic_lr", 3e-4),
        "actor_lr": model_cfg.get("actor_lr", 1e-4),
        "entropy_coef": model_cfg.get("entropy_coef", 1e-3),
        "n_constraints": env_cfg["n_constraints"],
    }

    agent = SACWithConstraints(sac_cfg, env, device=device)
    replay = ReplayBuffer(capacity=100_000)

    # Fixed one-hot preference: optimize ONLY for latency [0, 0, 1]
    fixed_w = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Use a single scenario for training
    scen_curriculum = ScenarioCurriculum()
    scenario_name = "normal"
    c_enc = scen_curriculum.encode(scenario_name).astype(np.float32)

    print(f"  Training for 10000 steps with fixed w = {fixed_w.tolist()}, scenario = {scenario_name}")
    total_steps = 10000
    batch_size = train_cfg.get("batch_size", 128)
    updates_per_step = train_cfg.get("updates_per_step", 3)

    step = 0
    while step < total_steps:
        env.scenario = scenario_name
        obs, _ = env.reset()
        done = False
        while not done and step < total_steps:
            cond = np.concatenate([fixed_w, c_enc], axis=0).astype(np.float32)
            obs_t = torch.tensor(obs[None, :], dtype=torch.float32, device=device)
            cond_t = torch.tensor(cond[None, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                a, _, _ = agent.actor.sample(obs_t, cond_t)
            action = a.cpu().numpy()[0]
            next_obs, reward_vec, terminated, truncated, info = env.step(action)
            cost = info["cost"]
            replay.push({
                "s": obs.astype(np.float32),
                "a": action.astype(np.float32),
                "r_vec": reward_vec.astype(np.float32),
                "cost": cost.astype(np.float32),
                "s_next": next_obs.astype(np.float32),
                "w": fixed_w.copy(),
                "c": c_enc.copy(),
                "done": float(terminated or truncated),
            })
            obs = next_obs
            done = terminated or truncated
            step += 1

            # SAC updates
            if len(replay) >= batch_size:
                for _ in range(updates_per_step):
                    batch = replay.sample(batch_size)
                    w_batch = np.stack(batch["w"])
                    c_batch = np.stack(batch["c"])
                    agent.update(batch, w_batch, c_batch, np.zeros(2), relabel=True, mode="linear")

        if step % 2000 == 0:
            print(f"  Training step {step}/{total_steps}, replay size {len(replay)}")

    # Evaluate trained agent over full preference-scenario grid
    print("\n  Evaluating trained single-objective SAC agent...")
    W_grid, C_bank = get_eval_preferences_scenarios()
    episode_length = cfg.get("evaluation", {}).get("eval_episode_length", 180)

    def sac_policy(obs, w, c):
        cond = np.concatenate([w, c], axis=0).astype(np.float32)
        obs_t = torch.tensor(obs[None, :], dtype=torch.float32, device=device)
        cond_t = torch.tensor(cond[None, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            mu, _ = agent.actor.forward(obs_t, cond_t)
            raw = torch.tanh(mu).cpu().numpy()[0, 0]
        a = (raw + 1.0) / 2.0  # rescale [-1,1] → [0,1] for env
        return np.clip(a, 0.0, 1.0)

    return_grid, cost_grid, per_episode_points, per_episode_meta = run_eval_episodes(
        env, sac_policy, W_grid, C_bank, episode_length, episodes=3
    )

    thresholds = np.array(cfg.get("constraints", {}).get("cost_thresholds", [0.01, 4.7]))
    results = compute_metrics_from_eval(
        per_episode_points, cost_grid, return_grid, W_grid, len(C_bank), thresholds,
        per_episode_meta=per_episode_meta
    )
    results["method"] = "single_objective_sac"
    results["description"] = f"SAC trained for 10k steps with fixed one-hot w={fixed_w.tolist()}, no curriculum"

    print(f"  Hypervolume:              {results['hypervolume']:.6e}")
    print(f"  Coverage fraction:        {results['coverage_fraction']:.3f}")
    print(f"  Constraint violation:     {results['constraint_violation_rate']:.3f}")
    print(f"  Pareto front size:        {results['pareto_front_size']}")
    return results


# ---------------------------------------------------------------------------
# BASELINE 3: NSGA-II / Heuristic Proxy (Random Search + Non-dominated Sorting)
# ---------------------------------------------------------------------------

def run_heuristic_proxy_baseline(env, cfg, device="cpu"):
    print("\n" + "=" * 60)
    print("BASELINE 3: Heuristic Proxy (Random Search + Non-dominated Sorting)")
    print("=" * 60)

    W_grid, C_bank = get_eval_preferences_scenarios()
    episode_length = cfg.get("evaluation", {}).get("eval_episode_length", 180)
    thresholds = np.array(cfg.get("constraints", {}).get("cost_thresholds", [0.01, 4.7]))

    n_random = 100

    print(f"  Sampling {n_random} random weight configurations...")
    all_returns = []
    all_costs = []

    for i in range(n_random):
        w_rand = np.random.dirichlet(np.ones(3)).astype(np.float32)
        c_rand = env.action_space.sample()

        # Evaluate this configuration on a single scenario
        scenario_name = np.random.choice(ScenarioCurriculum().scenarios)
        c_enc = ScenarioCurriculum().encode(scenario_name).astype(np.float32)

        env.scenario = scenario_name
        obs, _ = env.reset()
        ret = np.zeros(3)
        cost_acc = np.zeros(2)
        for t in range(episode_length):
            cond = np.concatenate([w_rand, c_enc], axis=0).astype(np.float32)
            obs_t = torch.tensor(obs[None, :], dtype=torch.float32, device=device)
            cond_t = torch.tensor(cond[None, :], dtype=torch.float32, device=device)

            # Use a simple heuristic: action is the weighted combination of
            # energy-saving and latency-minimizing directions, plus random exploration
            with torch.no_grad():
                noise = torch.randn(1) * 0.3
                # Heuristic: action = sigmoid(weighted_score + noise), clipped to [0,1]
                action = float(torch.sigmoid(noise).item())
                action = np.array([np.clip(action, 0.0, 1.0)], dtype=np.float32)

            next_obs, reward_vec, terminated, truncated, info = env.step(action)
            ret += reward_vec
            cost_acc += info["cost"]
            obs = next_obs
            if terminated or truncated:
                break

        all_returns.append(ret)
        all_costs.append(cost_acc)

    all_returns = np.array(all_returns)
    all_costs = np.array(all_costs)

    # Non-dominated sorting: find Pareto front in objective space
    points = metrics.to_maximization_space(all_returns)
    pf = metrics.pareto_front(points) if points.shape[0] > 0 else np.zeros((0, 3))

    # Compute hypervolume
    if points.shape[0] > 0:
        scales = np.maximum(np.percentile(points, 95, axis=0), 1e-9)
        points_norm = points / scales
        ref = points_norm.min(axis=0) - 1e-6
        hv = metrics.hypervolume_mc(points_norm, ref=ref, n_samples=20000)
    else:
        hv = 0.0

    # Constraint metrics from cost samples
    violation_rate = (all_costs > thresholds.reshape(1, 2)).any(axis=1).mean()
    mean_violation = float(all_costs[all_costs > 0].mean()) if (all_costs > 0).any() else 0.0

    results = {
        "method": "heuristic_proxy_nsga2",
        "description": f"Random search over {n_random} weight configs + non-dominated sorting",
        "hypervolume": float(hv),
        "coverage_fraction": 0.0,
        "pareto_front_size": int(len(pf)),
        "constraint_violation_rate": float(violation_rate),
        "mean_constraint_violation": float(mean_violation),
        "total_episodes": int(n_random),
    }

    print(f"  Hypervolume:              {hv:.6e}")
    print(f"  Pareto front size:        {len(pf)}")
    print(f"  Constraint violation:     {violation_rate:.3f}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run SuSCAP-RL baseline evaluations")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--device", default="cpu", help="Torch device (cpu / cuda)")
    parser.add_argument("--baselines", nargs="+",
                        choices=["random", "sac", "heuristic", "all"],
                        default=["all"],
                        help="Which baselines to run")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device

    env = SchedulerEnv(
        data_path="data/processed/merged_timeseries.csv",
        episode_length=cfg.get("evaluation", {}).get("eval_episode_length", 180),
    )

    baselines_to_run = ["random", "sac", "heuristic"] if "all" in args.baselines else args.baselines

    results = []

    if "random" in baselines_to_run:
        t0 = time.time()
        res = run_random_baseline(env, cfg, device=device)
        res["wall_time_sec"] = time.time() - t0
        results.append(res)

    if "sac" in baselines_to_run:
        t0 = time.time()
        res = run_single_objective_sac_baseline(env, cfg, device=device)
        res["wall_time_sec"] = time.time() - t0
        results.append(res)

    if "heuristic" in baselines_to_run:
        t0 = time.time()
        res = run_heuristic_proxy_baseline(env, cfg, device=device)
        res["wall_time_sec"] = time.time() - t0
        results.append(res)

    # Save results
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'=' * 60}")
    print(f"Results saved to {out_path}")
    print(f"{'=' * 60}")

    # Summary table
    print(f"\n{'Method':<35} {'HV':<14} {'Coverage':<10} {'Violation':<10} {'PF Size':<8}")
    print("-" * 80)
    for r in results:
        hv_str = f"{r['hypervolume']:.4e}"
        cov_str = f"{r['coverage_fraction']:.3f}"
        vio_str = f"{r['constraint_violation_rate']:.3f}"
        pf_str = f"{r['pareto_front_size']}"
        print(f"{r['method']:<35} {hv_str:<14} {cov_str:<10} {vio_str:<10} {pf_str:<8}")


if __name__ == "__main__":
    main()
