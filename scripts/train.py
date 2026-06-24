import yaml
import argparse
from env.scheduler_env import SchedulerEnv
from training.trainer import Trainer

# CLI arguments
parser = argparse.ArgumentParser(description="Multi-phase curriculum training for SuSCAP-RL")
parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
parser.add_argument("--phases", nargs="+", default=None, help="Phases to run (default: all)")
args = parser.parse_args()

cfg = yaml.safe_load(open(args.config))

# Initialize environment and trainer
ep_len = cfg.get("training", {}).get("episode_length", 180)
env = SchedulerEnv(data_path="data/processed/merged_timeseries.csv", episode_length=ep_len)
trainer = Trainer(cfg, env, device=args.device)

# Get phase schedule from config
phase_config = cfg.get("curriculum", {})
phase_schedule = phase_config.get("phase_schedule", ["corners", "edges", "grid", "adversarial", "refinement"])
phase_steps = phase_config.get("phase_steps", {
    "corners": 5000,
    "edges": 10000,
    "grid": 15000,
    "adversarial": 5000,
    "refinement": 5000
})

# Determine which phases to run
if args.phases:
    phases_to_run = args.phases
else:
    phases_to_run = phase_schedule

print(f"\n{'='*70}")
print(f"SuSCAP-RL Multi-Phase Curriculum Training")
print(f"{'='*70}")
print(f"Config: {args.config}")
print(f"Device: {args.device}")
print(f"Phases: {phases_to_run}")
print(f"{'='*70}\n")

# Run training phases
for phase in phases_to_run:
    if phase not in phase_steps:
        print(f"WARNING: Phase '{phase}' not found in config, skipping")
        continue
    
    # Set current phase
    trainer.set_phase(phase)
    
    # Train this phase
    steps = phase_steps[phase]
    trainer.train_phase(steps)
    
    print(f"\n[PHASE_SUMMARY] Completed '{phase}' phase")
    print(f"  Global step: {trainer.global_step}")
    print(f"  Replay buffer size: {len(trainer.replay)}")
    print(f"  Dual variables (lambdas): {trainer.dual.lambdas}")
    print(f"  Cost EWMA: {trainer.cost_ewma}")
    print()

print(f"\n{'='*70}")
print(f"All training phases completed!")
print(f"Total steps: {trainer.global_step}")
print(f"Final checkpoint: checkpoints/final.pt")
print(f"{'='*70}\n")
