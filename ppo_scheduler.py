import argparse
import random
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import json
import os
class CPUSchedulingEnv(gym.Env):
    def __init__(self, processes=None, max_proc=10, time_quantum=1):
        super().__init__()
        self.max_proc = max_proc
        self.time_quantum = time_quantum
        self.action_space = spaces.Discrete(self.max_proc)

        low = np.zeros((self.max_proc * 4,), dtype=np.float32)
        high = np.ones((self.max_proc * 4,), dtype=np.float32) * 100
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.process_template = processes
        self._reset_internal()

    def _reset_internal(self):
        self.procs = []
        self.time = 0.0
        self.done = False
        self.gantt_chart = []

    def _reset_processes(self, processes):
        self.procs = []
        for p in processes[:self.max_proc]:
            self.procs.append({
                "pid": p.get("PID", p.get("pid")),
                "arrival": float(p.get("Arrival", p.get("arrival"))),
                "burst": float(p.get("Burst", p.get("burst"))),
                "remaining": float(p.get("Burst", p.get("burst"))),
            })

        # Pad to max_proc slots
        while len(self.procs) < self.max_proc:
            self.procs.append({
                "pid": None,
                "arrival": 1e6,
                "burst": 0.0,
                "remaining": 0.0,
            })

    def reset(self, processes=None):
        if processes is None:
            processes = self._sample_processes()
        self._reset_processes(processes)
        self.time = 0.0
        self.done = False
        self.gantt_chart = []
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for p in self.procs:
            # Calculate waiting time for processes that have arrived
            if p["arrival"] <= self.time and p["remaining"] > 0:
                executed_time = p["burst"] - p["remaining"]
                wait_time = self.time - p["arrival"] - executed_time
            else:
                wait_time = 0.0

            obs.extend([
                p["arrival"],
                p["burst"],
                p["remaining"],
                wait_time
            ])
        return np.array(obs, dtype=np.float32)

    def _available_indices(self):
        return [
            i for i, p in enumerate(self.procs)
            if p["arrival"] <= self.time and p["remaining"] > 0.0
        ]

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, {}

        action = int(action)
        available = self._available_indices()

        if action not in available:
            if available:
                action = min(available, key=lambda i: self.procs[i]["remaining"])
            else:
                # Advance time to next arrival
                next_arrivals = [p["arrival"] for p in self.procs
                                 if p["arrival"] > self.time and p["remaining"] > 0]
                if next_arrivals:
                    self.time = min(next_arrivals)
                else:
                    self.time += self.time_quantum
                return self._get_obs(), -5.0, False, {"invalid": True}

        proc = self.procs[action]

        # Execute process and record in Gantt chart
        run_time = min(self.time_quantum, proc["remaining"])
        start_time = self.time

        proc["remaining"] -= run_time
        self.time += run_time

        # Add execution segment to Gantt chart
        self.gantt_chart.append({
            "PID": proc["pid"],
            "Start": start_time,
            "Finish": self.time
        })
        reward = 0.0
        #reward for completing work
        reward += run_time * 1.0
        #bonus for completing a process
        if proc["remaining"] <= 0.0:
            completion_time = self.time
            turnaround = completion_time - proc["arrival"]
            # Bonus inversely proportional to turnaround time
            reward += 100.0 / max(turnaround, 1.0)

        #waiting processes penalty
        waiting_penalty = 0.0
        for p in self.procs:
            if p["pid"] is None or p["arrival"] > 1e5:
                continue
            if p["arrival"] <= self.time and p["remaining"] > 0:
                executed = p["burst"] - p["remaining"]
                wait = self.time - p["arrival"] - executed
                waiting_penalty += wait * 0.8

        reward -= waiting_penalty

        #penalty for context switches
        if len(self.gantt_chart) >= 2:
            if self.gantt_chart[-1]["PID"] != self.gantt_chart[-2]["PID"]:
                reward -= 0.5

        # Check if all processes are complete
        if all(p["remaining"] <= 0.0 or p["arrival"] > 1e5 for p in self.procs):
            self.done = True
            # Final reward based on total completion time
            reward += 50.0

        return self._get_obs(), reward, self.done, {}

    def _sample_processes(self):
        n = random.randint(3, min(self.max_proc, 7))
        procs = []
        scenario = random.choice(['early_burst', 'late_burst', 'mixed', 'uniform'])
        for i in range(n):
            if scenario == 'early_burst':
                # Multiple processes arrive early
                arrival = round(random.uniform(0, 3), 1)
                burst = round(random.uniform(2, 10), 1)
            elif scenario == 'late_burst':
                # Processes arrive spread out
                arrival = round(random.uniform(0, 20), 1)
                burst = round(random.uniform(1, 8), 1)
            elif scenario == 'mixed':
                # Mix of short and long jobs
                arrival = round(random.uniform(0, 15), 1)
                burst = round(random.choice([2, 3, 8, 10, 15]), 1)
            else:  # uniform
                arrival = round(i * 2.5, 1)
                burst = round(random.uniform(3, 12), 1)

            procs.append({
                "PID": f"P{i + 1}",
                "Arrival": arrival,
                "Burst": burst
            })

        return sorted(procs, key=lambda x: x["Arrival"])

    def get_gantt_data(self):
        if not self.gantt_chart:
            return []

        # Merge consecutive segments of the same process
        merged = []
        for segment in self.gantt_chart:
            if merged and merged[-1]["PID"] == segment["PID"] and merged[-1]["Finish"] == segment["Start"]:
                merged[-1]["Finish"] = segment["Finish"]
            else:
                merged.append(segment.copy())

        return merged


MODEL_PATH = "ppo_cpu_scheduler.zip"
METADATA_PATH = "ppo_metadata.json"


def make_vec_env(max_proc=10):
    return DummyVecEnv([
        lambda: CPUSchedulingEnv(processes=None, max_proc=max_proc, time_quantum=1)
    ])
def train_ppo_model(episodes=500, max_proc=10, verbose=1):
    print(f"\n{'=' * 60}")
    print(f"Starting PPO Training")
    print(f"{'=' * 60}")
    print(f"Episodes: {episodes}")
    print(f"Max Processes: {max_proc}")
    print(f"{'=' * 60}\n")

    env = make_vec_env(max_proc=max_proc)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=verbose,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=128,
        n_epochs=15,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        )
    )

    timesteps_per_episode = 1000
    total_timesteps = episodes * timesteps_per_episode

    model.learn(total_timesteps=total_timesteps)

    model.save(MODEL_PATH)
    print(f"\n✅ Model saved to: {MODEL_PATH}")

    metadata = {
        "episodes": episodes,
        "max_proc": max_proc,
        "total_timesteps": total_timesteps,
        "timesteps_per_episode": timesteps_per_episode
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved to: {METADATA_PATH}")

    print(f"\n{'=' * 60}")
    print(f"Training Complete!")
    print(f"{'=' * 60}\n")

    return model


def load_model():
    """Load trained PPO model"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            f"Please train the model first using:\n"
            f"  python ppo_scheduler.py --episodes 500"
        )

    env = make_vec_env()
    model = PPO.load(MODEL_PATH, env=env)

    # Load metadata if available
    metadata = None
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)

    return model, metadata


def ppo_schedule(processes, model):
    max_proc = max(10, len(processes))
    env = CPUSchedulingEnv(
        processes=processes,
        max_proc=max_proc,
        time_quantum=1
    )
    obs = env.reset(processes=processes)
    done = False

    max_iterations = sum(p["Burst"] for p in processes) * 3
    iterations = 0

    while not done and iterations < max_iterations:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        # Get available processes
        available = env._available_indices()

        # If chosen action is not valid, pick the best valid one
        if action not in available:
            if available:
                # Fallback to shortest remaining time
                action = min(available, key=lambda i: env.procs[i]["remaining"])
            else:
                # No process ready, advance time
                env.time += 1
                iterations += 1
                continue

        obs, _, done, _ = env.step(action)
        iterations += 1

    return env.get_gantt_data()


def continue_training(existing_model_path, additional_episodes=200, max_proc=10, verbose=1):
    print(f"\n{'=' * 60}")
    print(f"Continuing PPO Training")
    print(f"{'=' * 60}")
    print(f"Loading existing model from: {existing_model_path}")
    print(f"Additional Episodes: {additional_episodes}")
    print(f"Max Processes: {max_proc}")
    print(f"{'=' * 60}\n")

    env = make_vec_env(max_proc=max_proc)
    model = PPO.load(existing_model_path, env=env)

    # Train for additional episodes
    timesteps_per_episode = 1000
    additional_timesteps = additional_episodes * timesteps_per_episode

    print(f"Training for {additional_timesteps} additional timesteps...")
    model.learn(total_timesteps=additional_timesteps)

    # Save model
    model.save(MODEL_PATH)
    print(f"\n✅ Model saved to: {MODEL_PATH}")

    # Update metadata
    prev_metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            prev_metadata = json.load(f)

    total_episodes = prev_metadata.get("episodes", 0) + additional_episodes
    total_timesteps = prev_metadata.get("total_timesteps", 0) + additional_timesteps

    metadata = {
        "episodes": episodes,
        "max_proc": max_proc,
        "total_timesteps": total_timesteps,
        "timesteps_per_episode": timesteps_per_episode
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata updated to: {METADATA_PATH}")
    print(f"   Total episodes trained: {total_episodes}")

    print(f"\n{'=' * 60}")
    print(f"Continued Training Complete!")
    print(f"{'=' * 60}\n")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO model for CPU scheduling"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of training episodes (default: 500)"
    )
    parser.add_argument(
        "--max_proc",
        type=int,
        default=10,
        help="Maximum number of processes (default: 10)"
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level: 0=silent, 1=info, 2=debug (default: 1)"
    )
    parser.add_argument(
        "--continue-training",
        action="store_true",
        help="Continue training existing model (adds to existing training)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Force reset and train from scratch (ignores existing model)"
    )

    args = parser.parse_args()

    # Check if model exists
    model_exists = os.path.exists(MODEL_PATH)

    if args.continue_training:
        if not model_exists:
            print("❌ No existing model found. Training from scratch instead...")
            train_ppo_model(
                episodes=args.episodes,
                max_proc=args.max_proc,
                verbose=args.verbose
            )
        else:
            continue_training(
                existing_model_path=MODEL_PATH,
                additional_episodes=args.episodes,
                max_proc=args.max_proc,
                verbose=args.verbose
            )
    elif args.reset or not model_exists:
        if model_exists and args.reset:
            print("⚠️  Resetting and training from scratch (existing model will be overwritten)")
        train_ppo_model(
            episodes=args.episodes,
            max_proc=args.max_proc,
            verbose=args.verbose
        )
    else:
        # Model exists but no flag specified - ask user
        print(f"\n{'=' * 60}")
        print("⚠️  Model already exists!")
        print(f"{'=' * 60}")
        response = input(
            "\nDo you want to:\n"
            "  1. Continue training (add more episodes)\n"
            "  2. Reset and train from scratch\n"
            "  3. Cancel\n"
            "Enter choice (1/2/3): "
        ).strip()

        if response == "1":
            continue_training(
                existing_model_path=MODEL_PATH,
                additional_episodes=args.episodes,
                max_proc=args.max_proc,
                verbose=args.verbose
            )
        elif response == "2":
            train_ppo_model(
                episodes=args.episodes,
                max_proc=args.max_proc,
                verbose=args.verbose
            )
        else:
            print("Training cancelled.")
            return
if __name__ == "__main__":
    main()