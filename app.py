import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# Import the PPO scheduler module
try:
    from ppo_scheduler import load_model, CPUSchedulingEnv
    import random

    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    CPUSchedulingEnv = None
    st.warning("⚠️ ppo_scheduler.py not found. PPO algorithm will not be available.")

st.set_page_config(page_title="CPU Scheduling Visualizer", layout="wide")
st.title("CPU Scheduling Algorithm Simulator")


def ppo_schedule_wrapper(processes, model, metadata):
    max_proc = metadata.get('max_proc', 10) if metadata else 10
    env = CPUSchedulingEnv(
        processes=processes,
        max_proc=max_proc,
        time_quantum=1
    )
    obs = env.reset(processes=processes)
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        # Get available processes
        available = env._available_indices()

        if action not in available:
            if available:
                action = random.choice(available)
            else:
                obs, _, done, _ = env.step(0)
                continue

        obs, _, done, _ = env.step(action)

    gantt = env.get_gantt_data()
    real_pids = {p["PID"] for p in processes}
    return [g for g in gantt if g["PID"] in real_pids]


# Check if PPO model exists
MODEL_EXISTS = os.path.exists("ppo_cpu_scheduler.zip")
# ---- Algorithm Selection ----
available_algos = [
    "FCFS",
    "SJF (Non-Preemptive)",
    "SRTF (Preemptive)",
    "Round Robin",
    "Priority (Preemptive)",
    "Priority (Non-Preemptive)",
]

if PPO_AVAILABLE and MODEL_EXISTS:
    available_algos.insert(0, "PPO (ML-based)")

algo = st.selectbox("Select Scheduling Algorithm", available_algos)

# ---- Session state setup ----
if "processes" not in st.session_state:
    st.session_state.processes = [
        {"PID": "P1", "Arrival": 0, "Burst": 5, "Priority": 1}
    ]


def add_process():
    n = len(st.session_state.processes)
    st.session_state.processes.append(
        {"PID": f"P{n + 1}", "Arrival": 0, "Burst": 1, "Priority": 1}
    )


def remove_process():
    if len(st.session_state.processes) > 1:
        st.session_state.processes.pop()


# ---- User Input Section ----
st.subheader("Enter Process Details")

col1, col2 = st.columns(2)
col1.button("➕ Add Process", on_click=add_process)
col2.button("➖ Remove Last", on_click=remove_process)

for i, proc in enumerate(st.session_state.processes):
    with st.expander(f"Process {i + 1}"):
        proc["PID"] = st.text_input(f"Process ID", proc["PID"], key=f"pid_{i}")
        proc["Arrival"] = st.number_input(
            "Arrival Time", min_value=0, step=1, value=proc["Arrival"], key=f"arr_{i}"
        )
        proc["Burst"] = st.number_input(
            "Burst Time (ms)", min_value=1, step=1, value=proc["Burst"], key=f"burst_{i}"
        )
        if "Priority" in algo:
            proc["Priority"] = st.number_input(
                "Priority (lower = higher priority)",
                min_value=1,
                step=1,
                value=proc["Priority"],
                key=f"prio_{i}",
            )

processes = [dict(p) for p in st.session_state.processes]

# ---- Round Robin quantum ----
tq = None
if algo == "Round Robin":
    tq = st.number_input("Enter Time Quantum (ms)", min_value=1, step=1, value=2)
def fcfs(proc):
    proc = sorted(proc, key=lambda x: x["Arrival"])
    t, gantt = 0, []
    for p in proc:
        start = max(t, p["Arrival"])
        finish = start + p["Burst"]
        gantt.append({"PID": p["PID"], "Start": start, "Finish": finish})
        t = finish
    return gantt


def sjf_np(proc):
    proc = sorted(proc, key=lambda x: x["Arrival"])
    gantt, ready, time = [], [], 0
    remaining = proc.copy()
    while remaining or ready:
        ready += [p for p in remaining if p["Arrival"] <= time]
        remaining = [p for p in remaining if p["Arrival"] > time]
        if ready:
            p = min(ready, key=lambda x: x["Burst"])
            ready.remove(p)
            start = max(time, p["Arrival"])
            finish = start + p["Burst"]
            gantt.append({"PID": p["PID"], "Start": start, "Finish": finish})
            time = finish
        else:
            if remaining:
                time = min(p["Arrival"] for p in remaining)
    return gantt


def srtf(proc):
    proc = sorted(proc, key=lambda x: x["Arrival"])
    rem = {p["PID"]: p["Burst"] for p in proc}
    gantt, current, time = [], None, 0
    max_time = max(p["Arrival"] for p in proc) + sum(p["Burst"] for p in proc)

    while time <= max_time and any(rem[p] > 0 for p in rem):
        ready = [p for p in proc if p["Arrival"] <= time and rem[p["PID"]] > 0]
        if ready:
            p = min(ready, key=lambda x: rem[x["PID"]])
            if current != p["PID"]:
                if current is not None:
                    gantt[-1]["Finish"] = time
                gantt.append({"PID": p["PID"], "Start": time, "Finish": time})
                current = p["PID"]
            rem[p["PID"]] -= 1
            time += 1
        else:
            time += 1

    if gantt:
        gantt[-1]["Finish"] = time

    # Merge consecutive segments of same process
    merged = []
    for g in gantt:
        if merged and merged[-1]["PID"] == g["PID"] and merged[-1]["Finish"] == g["Start"]:
            merged[-1]["Finish"] = g["Finish"]
        else:
            merged.append(g)
    return merged


def round_robin(proc, tq):
    proc = sorted(proc, key=lambda x: x["Arrival"])
    rem = {p["PID"]: p["Burst"] for p in proc}
    gantt, queue, arrived = [], [], set()
    t = 0

    while any(rem[pid] > 0 for pid in rem):
        # Add newly arrived processes
        for p in proc:
            if p["Arrival"] <= t and p["PID"] not in arrived:
                queue.append(p)
                arrived.add(p["PID"])

        if queue:
            p = queue.pop(0)
            start = t
            run = min(tq, rem[p["PID"]])
            t += run
            rem[p["PID"]] -= run
            gantt.append({"PID": p["PID"], "Start": start, "Finish": t})

            # Add processes that arrived during execution
            for q in proc:
                if q["Arrival"] <= t and q["PID"] not in arrived:
                    queue.append(q)
                    arrived.add(q["PID"])

            # Re-add process if not finished
            if rem[p["PID"]] > 0:
                queue.append(p)
        else:
            # Jump to next arrival if queue is empty
            next_arrivals = [p["Arrival"] for p in proc if p["PID"] not in arrived]
            if next_arrivals:
                t = min(next_arrivals)

    return gantt


def priority_np(proc):
    proc = sorted(proc, key=lambda x: x["Arrival"])
    gantt, ready, t = [], [], 0
    remaining = proc.copy()

    while remaining or ready:
        ready += [p for p in remaining if p["Arrival"] <= t]
        remaining = [p for p in remaining if p["Arrival"] > t]

        if ready:
            p = min(ready, key=lambda x: (x["Priority"], x["Arrival"]))
            ready.remove(p)
            start = max(t, p["Arrival"])
            finish = start + p["Burst"]
            gantt.append({"PID": p["PID"], "Start": start, "Finish": finish})
            t = finish
        else:
            if remaining:
                t = min(p["Arrival"] for p in remaining)

    return gantt


def priority_preemptive(proc):
    proc = sorted(proc, key=lambda x: x["Arrival"])
    rem = {p["PID"]: p["Burst"] for p in proc}
    gantt, current, time = [], None, 0
    max_time = max(p["Arrival"] for p in proc) + sum(p["Burst"] for p in proc)

    while time <= max_time and any(rem[p] > 0 for p in rem):
        ready = [p for p in proc if p["Arrival"] <= time and rem[p["PID"]] > 0]
        if ready:
            p = min(ready, key=lambda x: (x["Priority"], x["Arrival"]))
            if current != p["PID"]:
                if current is not None:
                    gantt[-1]["Finish"] = time
                gantt.append({"PID": p["PID"], "Start": time, "Finish": time})
                current = p["PID"]
            rem[p["PID"]] -= 1
            time += 1
        else:
            time += 1

    if gantt:
        gantt[-1]["Finish"] = time

    # Merge consecutive segments
    merged = []
    for g in gantt:
        if merged and merged[-1]["PID"] == g["PID"] and merged[-1]["Finish"] == g["Start"]:
            merged[-1]["Finish"] = g["Finish"]
        else:
            merged.append(g)
    return merged


def calc_times(proc, gantt):
    results = []
    for p in proc:
        bursts = [g for g in gantt if g["PID"] == p["PID"]]
        if not bursts:  # Handle case where process wasn't scheduled
            continue
        start_time = min(g["Start"] for g in bursts)
        finish_time = max(g["Finish"] for g in bursts)
        turnaround = finish_time - p["Arrival"]
        waiting = turnaround - p["Burst"]
        results.append({
            "PID": p["PID"],
            "Arrival": p["Arrival"],
            "Burst": p["Burst"],
            "Start": start_time,
            "Finish": finish_time,
            "Waiting": waiting,
            "Turnaround": turnaround,
        })
    return results


# ---- Run Scheduling ----
if st.button("▶ Run Scheduling"):
    try:
        if algo == "PPO (ML-based)":
            if not PPO_AVAILABLE:
                st.error("❌ PPO module not available. Check if ppo_scheduler.py exists.")
            elif not MODEL_EXISTS:
                st.error(
                    "❌ No trained model found!\n\n"
                    "Train the model first using PyCharm:\n\n"
                    "`python ppo_scheduler.py --episodes 50`"
                )
            else:
                with st.spinner("Loading PPO model and running scheduling..."):
                    model, metadata = load_model()
                    gantt = ppo_schedule_wrapper(processes, model, metadata)
        elif algo == "FCFS":
            gantt = fcfs(processes)
        elif algo == "SJF (Non-Preemptive)":
            gantt = sjf_np(processes)
        elif algo == "SRTF (Preemptive)":
            gantt = srtf(processes)
        elif algo == "Round Robin":
            gantt = round_robin(processes, tq)
        elif algo == "Priority (Preemptive)":
            gantt = priority_preemptive(processes)
        elif algo == "Priority (Non-Preemptive)":
            gantt = priority_np(processes)
        else:
            gantt = []

        # ---- Gantt Chart ----
        st.subheader("Gantt Chart")
        gantt_df = pd.DataFrame(gantt)

        fig, ax = plt.subplots(figsize=(14, 3))
        colors = plt.cm.Set3.colors
        pid_colors = {p["PID"]: colors[i % len(colors)] for i, p in enumerate(processes)}

        # Draw the Gantt bars (no text inside)
        for _, row in gantt_df.iterrows():
            ax.barh(0, row["Finish"] - row["Start"], left=row["Start"],
                    color=pid_colors[row["PID"]], edgecolor='black', linewidth=1.5, height=0.6)

        # Configure axes
        ax.set_xlabel("Time (ms)", fontsize=12)
        ax.set_yticks([])
        ax.set_xlim(0, max(g["Finish"] for g in gantt) + 1)
        ax.set_ylim(-0.5, 0.5)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Add legend for process colors
        from matplotlib.patches import Patch

        legend_elements = [Patch(facecolor=pid_colors[p["PID"]],
                                 edgecolor='black',
                                 label=p["PID"])
                           for p in processes]
        ax.legend(handles=legend_elements, loc='upper right', ncol=min(len(processes), 6),
                  frameon=True, fontsize=10)

        plt.tight_layout()
        st.pyplot(fig)

        # ---- Metrics Table ----
        st.subheader("Process Metrics Table")
        results = calc_times(processes, gantt)
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        avg_wait = df["Waiting"].mean()
        avg_turn = df["Turnaround"].mean()
        st.markdown(f"""
        **Average Waiting Time:** {avg_wait:.2f} ms  
        **Average Turnaround Time:** {avg_turn:.2f} ms
        """)
    except Exception as e:
        st.error(f"❌ Error during scheduling: {str(e)}")
        st.exception(e)
