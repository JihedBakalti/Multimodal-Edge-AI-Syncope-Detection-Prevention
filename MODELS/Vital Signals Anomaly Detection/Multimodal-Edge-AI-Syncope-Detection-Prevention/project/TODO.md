# main2.py Implementation ✅ COMPLETE

## Status
- [x] Plan confirmed
- [x] main2.py polls spo2.json + PPGPT + unified results.json
- [ ] Test: run your vitals writer (e.g. spo2_writer.py → project/spo2.json) then `python project/main2.py`
- [ ] Verify: scores on each file change, anomalies when expected

## Usage
```
# Terminal 1: process that writes bpm/spo2/timestamp JSON to project/spo2.json
# Terminal 2:
cd project && python main2.py
```
Press Ctrl+C to stop main2.py.

Features:
- Polls spo2.json every 1s (configurable)
- Detects changes via mtime/timestamp
- Writes merged results.json (vitals + score + anomaly_detected + threshold)
