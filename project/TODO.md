# main2.py Implementation ✅ COMPLETE

## Status
- [x] Plan confirmed
- [x] Created main2.py with spo2.json polling + PPGPT anomaly detection
- [ ] Test: Run `python project/rewrite.py &` then `python project/main2.py`
- [ ] Verify: New scores printed every change (~2s), anomalies on SpO2 drops

## Usage
```
# Terminal 1: Start simulator
cd project &amp;&amp; python rewrite.py

# Terminal 2: Monitor + scores
cd project &amp;&amp; python main2.py
```
Press Ctrl+C to stop main2.py.

Features:
- Polls spo2.json every 1s (configurable)
- Detects changes via mtime/timestamp
- Re-synthesizes PPG/tokens, runs PPGPT, MAE score
- Prints clinical flags + verdict + JSON each update
- Optional plots per change
