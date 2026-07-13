# v2 — Engineered features + hybrid model + training curves

## What changed vs. your current working version

1. **Sequence channels: 2 → 3.** Added a GSR first-difference channel
   (rate of change), alongside GSR(z) and RR(z).
2. **New engineered scalar features (5 per window):** RR RMSSD, RR SDNN,
   GSR slope, GSR range, GSR peak count. Fed through a separate small MLP
   branch that's concatenated with the CNN+LSTM output before the final
   classifier layers (`model.py` → `CNNLSTMHybrid`).
3. **Training now plots and saves `outputs/training_curves.png`** — two
   panels, train vs. validation loss and train vs. validation accuracy,
   one point per epoch. Use this directly in your presentation to show
   how the model learned.
4. **`EXCLUDED_VOLUNTEERS` in `config.py` now bakes in the 14 volunteers
   you found with ~99% missing `RR_Interval_ms`** (110, 112, 113, 115,
   116, 120, 124, 135, 136, 137, 143, 144, 81, 158), on top of the
   original 6. This just documents what was already happening silently —
   training behavior doesn't change, but the printed skip-list and your
   methods section now match.

## How to update your local `AIModel` folder

Every file here is a straight replacement — same filenames, same
functions your model calls from `main.py`, just heavier internals.
Overwrite these in place:

```bash
cp config.py data_loader.py feature_extraction.py dataset.py model.py train.py evaluate.py main.py \
   /path/to/GSR-COSSINUS_VOLUNTEERS_DATA_FOR_ML/AIModel/
```

`data_loader.py` is included but unchanged from before — only listed here
for completeness, no need to worry about it.

Then just run as before:
```bash
python3 main.py
```

## What to expect

- Training will take a bit longer per epoch (three sequence channels
  instead of two, plus the extra MLP branch), but nothing dramatic on
  ~2,600 windows.
- Expect somewhere in the **80-88%** cross-subject test accuracy range,
  not a jump to near-100%. A cross-subject physiological classifier that
  reports zero errors on unseen people is a red flag for reviewers, not
  a win — real inter-subject variability means some genuine overlap
  between "relaxed" and "stressed" windows is expected and healthy to see
  in the confusion matrix.
- `outputs/training_curves.png` will show train and val curves — if val
  loss flattens out or ticks back up while train loss keeps dropping,
  that's the early-stopping signal doing its job, and worth mentioning
  in your methods write-up.

## Still on the table if 80-88% isn't enough

- **k-fold subject-wise cross-validation** (report mean ± std across folds
  instead of one train/val/test split) — this is what reviewers will
  actually want to see for a paper claim, since one 21-volunteer test set
  has real variance in what accuracy it reports depending on who lands in
  it.
- Worker-level **calibration at deployment** (2-3 min baseline per worker,
  reusing the same per-volunteer normalization logic already in
  `feature_extraction.py`) — this is separate from training and doesn't
  touch model weights, but is likely your biggest real-world accuracy
  lever for the oil-field deployment specifically.