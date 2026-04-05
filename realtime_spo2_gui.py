import threading
import time
import random
from collections import deque
from typing import Deque, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk


class SpO2Simulator:
    """Simulates pulse oximeter SpO₂ (%). Fainting pattern: stable → gradual fall → possible sharp drop."""

    def __init__(self, base_spo2: float = 97.0, dt: float = 0.5) -> None:
        self.base_spo2 = base_spo2
        self.current_spo2 = base_spo2
        self.mode = "normal"
        self.dt = dt

        self._episode_active = False
        self._episode_start_time: Optional[float] = None
        self._stable_dur = 0.0
        self._decline_dur = 0.0
        self._sharp_dur = 0.0
        self._plateau_low_dur = 0.0
        self._post_peak = 0.0
        self._valley_spo2 = 88.0

        self._sim_time = 0.0
        self._lock = threading.Lock()

    def set_mode(self, mode: str) -> None:
        with self._lock:
            if mode not in ("normal", "exercise", "manual", "fainting"):
                return
            self.mode = mode

    def set_spo2(self, value: float) -> None:
        with self._lock:
            self.current_spo2 = max(70.0, min(100.0, float(value)))
            self.mode = "manual"

    def trigger_fainting(self) -> None:
        """Pre-syncope-style: hold near baseline, gradual decline, then sharp desaturation."""
        with self._lock:
            self.mode = "fainting"
            self._episode_active = True
            self._episode_start_time = self._sim_time

            self._stable_dur = random.uniform(8.0, 14.0)
            self._decline_dur = random.uniform(25.0, 50.0)
            self._sharp_dur = random.uniform(6.0, 18.0)
            self._plateau_low_dur = random.uniform(8.0, 20.0)

            # Valley between ~78–88% (below 85% possible for critical alerts)
            self._valley_spo2 = random.uniform(78.0, 88.0)
            self._post_peak = self.base_spo2 * random.uniform(0.94, 0.98)

    def _step_normal(self) -> float:
        noise = random.gauss(0.0, 0.25)
        drift = (self.base_spo2 - self.current_spo2) * 0.08
        return self.current_spo2 + drift + noise

    def _step_exercise(self) -> float:
        target = self.base_spo2 - random.uniform(0.5, 2.0)
        delta = (target - self.current_spo2) * 0.04
        noise = random.gauss(0.0, 0.2)
        return self.current_spo2 + delta + noise

    def _step_manual(self) -> float:
        noise = random.gauss(0.0, 0.15)
        return self.current_spo2 + noise

    def _step_fainting(self) -> float:
        if not self._episode_active or self._episode_start_time is None:
            return self._step_normal()

        t_rel = self._sim_time - self._episode_start_time
        stable_end = self._stable_dur
        decline_end = stable_end + self._decline_dur
        sharp_end = decline_end + self._sharp_dur
        low_end = sharp_end + self._plateau_low_dur

        if t_rel < 0:
            target = self.base_spo2
        elif t_rel < stable_end:
            target = self.base_spo2
        elif t_rel < decline_end:
            frac = (t_rel - stable_end) / max(self._decline_dur, 1e-6)
            target = self.base_spo2 + frac * (self._post_peak - self.base_spo2)
        elif t_rel < sharp_end:
            frac = (t_rel - decline_end) / max(self._sharp_dur, 1e-6)
            target = self._post_peak + frac * (self._valley_spo2 - self._post_peak)
        elif t_rel < low_end:
            target = self._valley_spo2
        else:
            rec_time = t_rel - low_end
            rec_dur = 90.0
            frac = min(max(rec_time / rec_dur, 0.0), 1.0)
            recovery_target = min(99.0, self.base_spo2 + 0.5)
            target = self._valley_spo2 + frac * (recovery_target - self._valley_spo2)
            if rec_time > rec_dur:
                self._episode_active = False
                self.mode = "normal"

        delta = (target - self.current_spo2) * 0.25
        noise = random.gauss(0.0, 0.35)
        return self.current_spo2 + delta + noise

    def get_next_value(self) -> float:
        with self._lock:
            self._sim_time += self.dt

            if self.mode == "normal":
                self.current_spo2 = self._step_normal()
            elif self.mode == "exercise":
                self.current_spo2 = self._step_exercise()
            elif self.mode == "manual":
                self.current_spo2 = self._step_manual()
            elif self.mode == "fainting":
                self.current_spo2 = self._step_fainting()
            else:
                self.current_spo2 = self._step_normal()

            self.current_spo2 = max(70.0, min(100.0, self.current_spo2))
            return self.current_spo2


class SpO2BaselineTracker:
    """EWMA baseline on SpO₂, clipping large jumps so brief artifacts do not dominate."""

    def __init__(self, window_sec: float = 60.0, dt: float = 0.5, spike_threshold_pp: float = 4.0) -> None:
        self.window_sec = window_sec
        self.dt = dt
        self.alpha = dt / max(window_sec, dt)
        self.spike_threshold = spike_threshold_pp
        self.baseline: Optional[float] = None

    def update(self, spo2: float) -> float:
        if self.baseline is None:
            self.baseline = float(spo2)
            return self.baseline

        lo = self.baseline - self.spike_threshold
        hi = self.baseline + self.spike_threshold
        clipped = max(lo, min(hi, spo2))
        self.baseline = (1.0 - self.alpha) * self.baseline + self.alpha * clipped
        return self.baseline


class PreSyncopeSpO2Detector:
    """
    Heuristics aligned with typical pre-syncope SpO₂ patterns:
    critical if SpO₂ < 85%; rapid ≥3–5% drop from recent peak; sustained decline vs baseline.
    """

    def __init__(self, peak_window_sec: float = 120.0) -> None:
        self.peak_window_sec = peak_window_sec
        self.history: Deque[Tuple[float, float, float]] = deque()
        self.last_label: Optional[str] = None
        self.last_alert_time: Optional[float] = None
        self.cooldown_sec = 5.0

    def process(self, t: float, spo2: float, baseline: float) -> str:
        self.history.append((t, spo2, baseline))
        while self.history and (t - self.history[0][0]) > self.peak_window_sec:
            self.history.popleft()

        if self.history:
            peak_time, peak_spo2, peak_base = max(self.history, key=lambda x: x[1])
        else:
            peak_time, peak_spo2, peak_base = t, spo2, baseline

        peak_safe = max(peak_spo2, 1e-6)
        drop_from_peak_pp = peak_spo2 - spo2
        drop_from_peak_rel = drop_from_peak_pp / peak_safe
        time_since_peak = t - peak_time

        base_safe = baseline if baseline > 0 else 1.0
        below_baseline_pp = baseline - spo2

        label: Optional[str] = None

        if spo2 < 85.0:
            label = "CRITICAL LOW O2"
        elif spo2 < 88.0 and below_baseline_pp >= 5.0:
            label = "HIGH RISK"
        elif (
            drop_from_peak_pp >= 5.0
            and drop_from_peak_rel >= 0.03
            and 0.0 <= time_since_peak <= 45.0
            and spo2 < 92.0
        ):
            label = "HIGH RISK"
        elif (
            drop_from_peak_pp >= 3.0
            and 0.0 <= time_since_peak <= 60.0
            and below_baseline_pp >= 2.5
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

        if below_baseline_pp >= 2.0 and spo2 < peak_spo2 - 1.0:
            return "Declining"
        return "Normal"


class RealTimeSpO2GUI:
    """Tkinter + Matplotlib control panel for real-time SpO₂ monitoring."""

    def __init__(self) -> None:
        self.dt = 0.5
        base_spo2 = random.uniform(96.0, 99.0)

        self.sim = SpO2Simulator(base_spo2=base_spo2, dt=self.dt)
        self.baseline_tracker = SpO2BaselineTracker(window_sec=60.0, dt=self.dt, spike_threshold_pp=4.0)
        self.detector = PreSyncopeSpO2Detector(peak_window_sec=120.0)

        self.t0 = time.time()

        self.times: list[float] = []
        self.spo2_values: list[float] = []
        self.baselines: list[float] = []

        self.root = tk.Tk()
        self.root.title("Real-Time SpO₂ & Pre-Syncope Pattern Detection")

        status_frame = ttk.Frame(self.root, padding="5")
        status_frame.pack(side=tk.TOP, fill=tk.X)

        self.label_time = ttk.Label(status_frame, text="Time: 0.0 s")
        self.label_time.pack(side=tk.LEFT, padx=5)

        self.label_spo2 = ttk.Label(status_frame, text="SpO₂: -- %")
        self.label_spo2.pack(side=tk.LEFT, padx=5)

        self.label_baseline = ttk.Label(status_frame, text="Baseline: -- %")
        self.label_baseline.pack(side=tk.LEFT, padx=5)

        self.label_status = ttk.Label(
            status_frame,
            text="Status: --",
            font=("Segoe UI", 14, "bold"),
            foreground="black",
        )
        self.label_status.pack(side=tk.LEFT, padx=10)

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
        ttk.Button(control_frame, text="Pre-syncope trigger", command=self.sim.trigger_fainting).pack(
            side=tk.LEFT, padx=2
        )

        ttk.Button(control_frame, text="SpO₂ −", command=self._spo2_down).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="SpO₂ +", command=self._spo2_up).pack(side=tk.LEFT, padx=2)

        ttk.Button(control_frame, text="Quit", command=self._on_quit).pack(side=tk.RIGHT, padx=2)

        self.spo2_scale = ttk.Scale(
            control_frame,
            from_=75,
            to=100,
            orient=tk.HORIZONTAL,
            command=self._on_scale_change,
        )
        self.spo2_scale.set(base_spo2)
        self.spo2_scale.pack(side=tk.RIGHT, padx=5, fill=tk.X, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.line_spo2, = self.ax.plot([], [], label="SpO₂", color="tab:cyan")
        self.line_base, = self.ax.plot([], [], label="Baseline", color="orange", linestyle="--")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("SpO₂ (%)")
        self.ax.set_ylim(70, 101)
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

        self.ani = FuncAnimation(self.fig, self._update_plot, interval=int(self.dt * 1000))

        self.root.after(int(self.dt * 1000), self._update_simulation)

    def _spo2_up(self) -> None:
        current = self.spo2_scale.get()
        new_val = min(100.0, current + 1.0)
        self.spo2_scale.set(new_val)
        self.sim.set_spo2(new_val)

    def _spo2_down(self) -> None:
        current = self.spo2_scale.get()
        new_val = max(70.0, current - 1.0)
        self.spo2_scale.set(new_val)
        self.sim.set_spo2(new_val)

    def _on_scale_change(self, value: str) -> None:
        try:
            v = float(value)
        except ValueError:
            return
        self.sim.set_spo2(v)

    def _on_quit(self) -> None:
        plt.close(self.fig)
        self.root.destroy()

    def _update_simulation(self) -> None:
        spo2 = self.sim.get_next_value()
        baseline = self.baseline_tracker.update(spo2)
        t_rel = time.time() - self.t0

        self.times.append(t_rel)
        self.spo2_values.append(spo2)
        self.baselines.append(baseline)

        max_window = 180.0
        while self.times and (t_rel - self.times[0]) > max_window:
            self.times.pop(0)
            self.spo2_values.pop(0)
            self.baselines.pop(0)

        status = self.detector.process(t_rel, spo2, baseline)

        self.label_time.config(text=f"Time: {t_rel:5.1f} s")
        self.label_spo2.config(text=f"SpO₂: {spo2:5.1f} %")
        self.label_baseline.config(text=f"Baseline: {baseline:5.1f} %")

        if status == "CRITICAL LOW O2":
            color = "red"
            text = "🛑 CRITICAL (SpO₂ < 85%)"
        elif status == "HIGH RISK":
            color = "orange red"
            text = "🚨 HIGH RISK (rapid / deep desaturation)"
        elif status == "EARLY WARNING":
            color = "dark orange"
            text = "⚠️ EARLY WARNING (sustained decline)"
        elif status == "Declining":
            color = "royal blue"
            text = "Declining vs baseline"
        else:
            color = "dark green"
            text = "Normal"

        self.label_status.config(text=text, foreground=color)

        self.root.after(int(self.dt * 1000), self._update_simulation)

    def _update_plot(self, _frame) -> None:
        if not self.times:
            return
        self.line_spo2.set_data(self.times, self.spo2_values)
        self.line_base.set_data(self.times, self.baselines)

        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.set_ylim(70, 101)

    def run(self) -> None:
        plt.ion()
        plt.show(block=False)
        self.root.mainloop()


def main() -> None:
    gui = RealTimeSpO2GUI()
    gui.run()


if __name__ == "__main__":
    main()
