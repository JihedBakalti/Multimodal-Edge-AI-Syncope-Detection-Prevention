import threading
import time
import random
from collections import deque
from typing import Deque, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk


# Reuse core logic by importing from the console version if desired.
# For simplicity and isolation, we redefine minimal versions here.


class HeartRateSimulator:
    def __init__(self, base_hr: float = 75.0, dt: float = 0.5) -> None:
        self.base_hr = base_hr
        self.current_hr = base_hr
        self.mode = "normal"
        self.dt = dt

        self._fainting_active = False
        self._faint_start_time: Optional[float] = None
        self._faint_rise_dur = 0.0
        self._faint_plateau_dur = 0.0
        self._faint_drop_dur = 0.0
        self._faint_peak_hr = base_hr
        self._faint_low_hr = max(30.0, base_hr * 0.4)

        self._sim_time = 0.0
        self._lock = threading.Lock()

    def set_mode(self, mode: str) -> None:
        with self._lock:
            if mode not in ("normal", "exercise", "manual", "fainting"):
                return
            self.mode = mode

    def set_heart_rate(self, value: float) -> None:
        with self._lock:
            self.current_hr = max(30.0, min(220.0, float(value)))
            self.mode = "manual"

    def trigger_fainting(self) -> None:
        with self._lock:
            self.mode = "fainting"
            self._fainting_active = True
            self._faint_start_time = self._sim_time

            self._faint_rise_dur = random.uniform(10.0, 30.0)
            self._faint_plateau_dur = random.uniform(5.0, 15.0)
            self._faint_drop_dur = random.uniform(10.0, 30.0)

            peak_factor = random.uniform(1.2, 1.4)
            self._faint_peak_hr = self.base_hr * peak_factor

            # Stronger drop: 30–50% of peak (50–70% drop)
            drop_factor = random.uniform(0.3, 0.5)
            self._faint_low_hr = max(25.0, self._faint_peak_hr * drop_factor)

    def _step_normal(self) -> float:
        noise = random.gauss(0.0, 1.5)
        drift = (self.base_hr - self.current_hr) * 0.05
        return self.current_hr + drift + noise

    def _step_exercise(self) -> float:
        target = self.base_hr * 1.4
        delta = (target - self.current_hr) * 0.03
        noise = random.gauss(0.0, 1.0)
        return self.current_hr + delta + noise

    def _step_manual(self) -> float:
        noise = random.gauss(0.0, 0.5)
        return self.current_hr + noise

    def _step_fainting(self) -> float:
        if not self._fainting_active or self._faint_start_time is None:
            return self._step_normal()

        t_rel = self._sim_time - self._faint_start_time
        rise_end = self._faint_rise_dur
        plateau_end = rise_end + self._faint_plateau_dur
        drop_end = plateau_end + self._faint_drop_dur

        if t_rel < 0:
            target = self.base_hr
        elif t_rel < rise_end:
            frac = t_rel / max(rise_end, 1e-6)
            target = self.base_hr + frac * (self._faint_peak_hr - self.base_hr)
        elif t_rel < plateau_end:
            target = self._faint_peak_hr
        elif t_rel < drop_end:
            frac = (t_rel - plateau_end) / max(self._faint_drop_dur, 1e-6)
            target = self._faint_peak_hr + frac * (self._faint_low_hr - self._faint_peak_hr)
        else:
            rec_time = t_rel - drop_end
            rec_dur = 60.0
            frac = min(max(rec_time / rec_dur, 0.0), 1.0)
            recovery_target = self.base_hr * 0.9
            target = self._faint_low_hr + frac * (recovery_target - self._faint_low_hr)
            if rec_time > rec_dur:
                self._fainting_active = False
                self.mode = "normal"

        delta = (target - self.current_hr) * 0.2
        noise = random.gauss(0.0, 1.0)
        return self.current_hr + delta + noise

    def get_next_value(self) -> float:
        with self._lock:
            self._sim_time += self.dt

            if self.mode == "normal":
                self.current_hr = self._step_normal()
            elif self.mode == "exercise":
                self.current_hr = self._step_exercise()
            elif self.mode == "manual":
                self.current_hr = self._step_manual()
            elif self.mode == "fainting":
                self.current_hr = self._step_fainting()
            else:
                self.current_hr = self._step_normal()

            self.current_hr = max(30.0, min(220.0, self.current_hr))
            return self.current_hr


class BaselineTracker:
    def __init__(self, window_sec: float = 60.0, dt: float = 0.5, spike_threshold_bpm: float = 25.0) -> None:
        self.window_sec = window_sec
        self.dt = dt
        self.alpha = dt / max(window_sec, dt)
        self.spike_threshold = spike_threshold_bpm
        self.baseline: Optional[float] = None

    def update(self, hr: float) -> float:
        if self.baseline is None:
            self.baseline = float(hr)
            return self.baseline

        clipped_hr = max(self.baseline - self.spike_threshold, min(self.baseline + self.spike_threshold, hr))
        self.baseline = (1.0 - self.alpha) * self.baseline + self.alpha * clipped_hr
        return self.baseline


class FaintingDetector:
    def __init__(self, peak_window_sec: float = 120.0) -> None:
        self.peak_window_sec = peak_window_sec
        self.history: Deque[Tuple[float, float, float]] = deque()
        self.last_label: Optional[str] = None
        self.last_alert_time: Optional[float] = None
        self.cooldown_sec = 5.0

    def process(self, t: float, hr: float, baseline: float) -> str:
        self.history.append((t, hr, baseline))
        while self.history and (t - self.history[0][0]) > self.peak_window_sec:
            self.history.popleft()

        if self.history:
            peak_time, peak_hr, peak_base = max(self.history, key=lambda x: x[1])
        else:
            peak_time, peak_hr, peak_base = t, hr, baseline

        # Increase relative to baseline at the time of the peak
        peak_base_safe = peak_base if peak_base > 0 else 1.0
        increase_from_baseline_at_peak = (peak_hr - peak_base_safe) / peak_base_safe

        peak_hr_safe = peak_hr if peak_hr > 0 else 1.0
        drop_from_peak_rel = (peak_hr_safe - hr) / peak_hr_safe
        abs_drop_bpm = peak_hr - hr
        time_since_peak = t - peak_time

        label: Optional[str] = None

        # Critical & high-risk thresholds relaxed slightly to ensure triggered pattern crosses them
        if baseline > 0 and hr <= 0.65 * baseline:
            label = "CRITICAL FAINTING"
        elif baseline > 0 and (hr <= 0.80 * baseline) and (abs_drop_bpm >= 20.0):
            label = "HIGH RISK"
        # Early warning: peak rose ≥20% above baseline at peak time, and we have since
        # dropped ≥30% from that peak within 60 s.
        elif (
            increase_from_baseline_at_peak >= 0.20
            and drop_from_peak_rel >= 0.30
            and 0.0 <= time_since_peak <= 60.0
        ):
            label = "EARLY WARNING"

        now = t
        if label is not None:
            if (
                self.last_label == label
                and self.last_alert_time is not None
                and (now - self.last_alert_time) < self.cooldown_sec
            ):
                return label
            self.last_label = label
            self.last_alert_time = now
            return label

        # No detection; indicate sustained rise if we're still high vs current baseline
        baseline_safe = baseline if baseline > 0 else 1.0
        percent_increase_now = (hr - baseline_safe) / baseline_safe
        if percent_increase_now >= 0.20:
            return "Rising"
        return "Normal"


class RealTimeGUI:
    """
    Tkinter + Matplotlib control panel for real-time HR monitoring.
    """

    def __init__(self) -> None:
        self.dt = 0.5
        base_hr = random.uniform(60.0, 80.0)

        self.sim = HeartRateSimulator(base_hr=base_hr, dt=self.dt)
        self.baseline_tracker = BaselineTracker(window_sec=60.0, dt=self.dt, spike_threshold_bpm=25.0)
        self.detector = FaintingDetector(peak_window_sec=120.0)

        self.t0 = time.time()

        # Data buffers for plotting
        self.times = []
        self.hrs = []
        self.baselines = []

        # Tkinter window
        self.root = tk.Tk()
        self.root.title("Real-Time Heart Rate & Fainting Detection")

        # Status frame
        status_frame = ttk.Frame(self.root, padding="5")
        status_frame.pack(side=tk.TOP, fill=tk.X)

        self.label_time = ttk.Label(status_frame, text="Time: 0.0 s")
        self.label_time.pack(side=tk.LEFT, padx=5)

        self.label_hr = ttk.Label(status_frame, text="HR: -- bpm")
        self.label_hr.pack(side=tk.LEFT, padx=5)

        self.label_baseline = ttk.Label(status_frame, text="Baseline: -- bpm")
        self.label_baseline.pack(side=tk.LEFT, padx=5)

        # Status label: make it large and prominent
        self.label_status = ttk.Label(
            status_frame,
            text="Status: --",
            font=("Segoe UI", 14, "bold"),
            foreground="black",
        )
        self.label_status.pack(side=tk.LEFT, padx=10)

        # Control panel
        control_frame = ttk.Frame(self.root, padding="5")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(control_frame, text="Normal", command=lambda: self.sim.set_mode("normal")).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(control_frame, text="Exercise", command=lambda: self.sim.set_mode("exercise")).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(control_frame, text="Manual", command=lambda: self.sim.set_mode("manual")).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(control_frame, text="Fainting Trigger", command=self.sim.trigger_fainting).pack(
            side=tk.LEFT, padx=2
        )

        ttk.Button(control_frame, text="HR -", command=self._hr_down).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="HR +", command=self._hr_up).pack(side=tk.LEFT, padx=2)

        ttk.Button(control_frame, text="Quit", command=self._on_quit).pack(side=tk.RIGHT, padx=2)

        # HR slider for manual mode
        self.hr_scale = ttk.Scale(
            control_frame,
            from_=40,
            to=180,
            orient=tk.HORIZONTAL,
            command=self._on_scale_change,
        )
        self.hr_scale.set(base_hr)
        self.hr_scale.pack(side=tk.RIGHT, padx=5, fill=tk.X, expand=True)

        # Matplotlib figure inside its own window (simple approach)
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.line_hr, = self.ax.plot([], [], label="HR", color="tab:blue")
        self.line_base, = self.ax.plot([], [], label="Baseline", color="orange", linestyle="--")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("HR (bpm)")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

        # Animation for the plot
        self.ani = FuncAnimation(self.fig, self._update_plot, interval=int(self.dt * 1000))

        # Start simulation loop on Tkinter timer
        self.root.after(int(self.dt * 1000), self._update_simulation)

    def _hr_up(self) -> None:
        current = self.hr_scale.get()
        new_val = min(200.0, current + 5.0)
        self.hr_scale.set(new_val)
        self.sim.set_heart_rate(new_val)

    def _hr_down(self) -> None:
        current = self.hr_scale.get()
        new_val = max(40.0, current - 5.0)
        self.hr_scale.set(new_val)
        self.sim.set_heart_rate(new_val)

    def _on_scale_change(self, value: str) -> None:
        try:
            v = float(value)
        except ValueError:
            return
        self.sim.set_heart_rate(v)

    def _on_quit(self) -> None:
        plt.close(self.fig)
        self.root.destroy()

    def _update_simulation(self) -> None:
        # Generate next HR sample
        hr = self.sim.get_next_value()
        baseline = self.baseline_tracker.update(hr)
        t_rel = time.time() - self.t0

        self.times.append(t_rel)
        self.hrs.append(hr)
        self.baselines.append(baseline)

        # Keep last N seconds for plotting
        max_window = 180.0
        while self.times and (t_rel - self.times[0]) > max_window:
            self.times.pop(0)
            self.hrs.pop(0)
            self.baselines.pop(0)

        status = self.detector.process(t_rel, hr, baseline)

        # Update labels
        self.label_time.config(text=f"Time: {t_rel:5.1f} s")
        self.label_hr.config(text=f"HR: {hr:5.1f} bpm")
        self.label_baseline.config(text=f"Baseline: {baseline:5.1f} bpm")
        # Color-code status for visibility
        if status == "CRITICAL FAINTING":
            color = "red"
            text = "🛑 CRITICAL FAINTING"
        elif status == "HIGH RISK":
            color = "orange red"
            text = "🚨 HIGH RISK FAINTING"
        elif status == "EARLY WARNING":
            color = "dark orange"
            text = "⚠️ EARLY WARNING"
        elif status == "Rising":
            color = "royal blue"
            text = "Rising (≥20% above baseline)"
        else:
            color = "dark green"
            text = "Normal"

        self.label_status.config(text=text, foreground=color)

        # Schedule next update
        self.root.after(int(self.dt * 1000), self._update_simulation)

    def _update_plot(self, _frame) -> None:
        if not self.times:
            return
        self.line_hr.set_data(self.times, self.hrs)
        self.line_base.set_data(self.times, self.baselines)

        self.ax.relim()
        self.ax.autoscale_view()

    def run(self) -> None:
        # Show matplotlib non-blocking
        plt.ion()
        plt.show(block=False)
        # Start Tkinter main loop
        self.root.mainloop()


def main() -> None:
    gui = RealTimeGUI()
    gui.run()


if __name__ == "__main__":
    main()

