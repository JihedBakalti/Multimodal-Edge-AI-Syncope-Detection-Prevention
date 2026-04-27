#spo2_writer.py
import json
import os
import random
import time
from pathlib import Path

FILE = "spo2.json"

spo2 = 98
bpm = 76

print("Simulator started...")

while True:
    # normal small variation
    spo2_change = random.uniform(-0.5, 0.2)
    bpm_change = random.uniform(-2.0, 2.0)

    # occasional drop (simulate fainting)
    if random.random() < 0.07:
        spo2_change -= random.uniform(2.0, 6.0)
        # during stress/pre-syncope we can briefly rise or dip in HR
        if random.random() < 0.6:
            bpm_change += random.uniform(12.0, 25.0)
        else:
            bpm_change -= random.uniform(10.0, 20.0)

    spo2 = max(75.0, min(100.0, spo2 + spo2_change))
    bpm = max(40.0, min(170.0, bpm + bpm_change))

    data = {
        "spo2": round(spo2, 1),
        "bpm": round(bpm, 1),
        "timestamp": time.time()
    }

    path = Path(FILE)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)

    print("Written:", data)

    time.sleep(2)