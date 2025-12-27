import yaml
from env.scheduler_env import SchedulerEnv
from training.trainer import Trainer

cfg = yaml.safe_load(open("configs/default.yaml"))

env = SchedulerEnv(data_path="data/processed/merged_timeseries.csv", episode_length=3)
trainer = Trainer(cfg, env, device="cpu")
trainer.train()
