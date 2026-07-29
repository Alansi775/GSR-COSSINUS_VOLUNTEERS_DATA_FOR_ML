---
license: cc-by-nc-4.0
language:
- en
tags:
- time-series
- stress-detection
- biomedical-signal-processing
- gsr
- galvanic-skin-response
- heart-rate-variability
- wearable-sensors
size_categories:
- n<1K
---

# GSR + HRV Stress Detection — Cleaned Volunteer Dataset

Cleaned, anonymized physiological recordings from 124 volunteers used to train the [GSR+HRV stress detection model](https://huggingface.co/malansi/gsr-hrv-stress-detection-cnn-lstm). Collected as part of an applied research project at **Almutlaq United Company**.

**Full data-cleaning pipeline and training code:** [github.com/Alansi775/GSR-COSSINUS_VOLUNTEERS_DATA_FOR_ML](https://github.com/Alansi775/GSR-COSSINUS_VOLUNTEERS_DATA_FOR_ML/tree/main)

## Protocol

Each volunteer completed an identical 4-phase session, recorded at 1 Hz:

| Stage | Duration | Purpose |
|---|---|---|
| Calibration | ≈30s | Sensor settle / per-volunteer baseline |
| Normal | ≈4min | Relaxed baseline |
| Stress | ≈3min | Scripted time-pressure task |
| Relaxation | ≈1min | Recovery |

## Files

One CSV per volunteer: `cleaned_data_<id>.csv`, plus a diagnostic plot pair per volunteer (`plot_<id>.png`, `plot_<id>_nn_interval.png`).

| Column | Description |
|---|---|
| `Time_sec` | Seconds since session start |
| `Conductance_microS` | Galvanic skin conductance (µS) |
| `Stage` | Protocol stage label (`Calibration`, `Normal`, `Stress`, `Relaxation`) |
| `RR_Interval_ms` | Beat-to-beat heart interval (ms), sparse |
| `Conductance_microS_Normalized` | Per-volunteer baseline-normalized conductance |
| `RR_Interval_ms_Normalized` | Per-volunteer baseline-normalized RR interval |

No names or personal identifiers are included — volunteers are referenced only by numeric/alphanumeric ID.

## Known Data-Quality Notes

- Volunteers 35–144: sensor originally logged electrical **resistance**, corrected to conductance via `conductance = 1 / resistance`.
- 14 volunteers have `RR_Interval_ms` populated for fewer than 10 rows — excluded from model training (see the model card for the full exclusion list), but left in this dataset for transparency.

## License

`cc-by-nc-4.0` — non-commercial use, with attribution.

## Author

**Mohammed Saleh**
Almutlaq United Company
