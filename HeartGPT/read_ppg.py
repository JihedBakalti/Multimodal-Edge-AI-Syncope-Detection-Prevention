import wfdb
import numpy as np
import matplotlib.pyplot as plt

# --- Load the record ---
record = wfdb.rdrecord('data/3000386_0001')

# --- Check available channels ---
print("Channels:", record.sig_name)
# Output: ['RESP', 'PLETH', 'III', 'II']

# --- Extract PPG (PLETH is index 1) ---
pleth_index = record.sig_name.index('PLETH')
ppg_raw = record.p_signal[:, pleth_index]  # shape: [668125]

# --- Remove NaN values ---
ppg_raw = ppg_raw[~np.isnan(ppg_raw)]

print(f"Sampling rate : {record.fs} Hz")       # 125 Hz
print(f"Total samples : {len(ppg_raw)}")        # ~668125
print(f"Duration      : {len(ppg_raw)/record.fs:.1f} seconds")  # ~5345 sec

# --- Plot a 10-second window to verify ---
fs = record.fs  # 125
plt.figure(figsize=(12, 4))
plt.plot(ppg_raw[:10 * fs])
plt.title("Raw PPG (PLETH) — first 10 seconds")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("ppg_raw.png")
plt.show()

# --- Save for next step ---
np.save("data/ppg_raw.npy", ppg_raw)
print("Saved to data/ppg_raw.npy")