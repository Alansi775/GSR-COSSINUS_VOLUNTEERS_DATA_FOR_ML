import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

input_dir = '.'
output_dir = 'cleaned'
os.makedirs(output_dir, exist_ok=True)

csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

for file in csv_files:
    print(f"Processing {file}...")
    file_path = os.path.join(input_dir, file)
    df = pd.read_csv(file_path)

    # Ensure required columns exist
    required_columns = ["Time", "Resistance (Ohms)", "Stage"]
    if not all(col in df.columns for col in required_columns):
        print(f"Skipping {file}: Missing required columns.")
        continue

    # Find first 'Time == 0'
    start_idx = df[df["Time"] == 0].index
    if len(start_idx) == 0:
        print(f"Skipping {file}: No 'Time == 0' found.")
        continue
    df = df.loc[start_idx[0]:].reset_index(drop=True)

    # Remove everything after "Session Finished"
    session_finished_idx = df[df["Stage"].str.contains("Session Finished", na=False)].index
    if len(session_finished_idx) > 0:
        df = df.loc[:session_finished_idx[0] - 1]

    # Reset Time to be 0, 1, 2, ..., N
    df["Time"] = range(len(df))

    # Normalize Resistance column to 0–1
    scaler = MinMaxScaler()
    df["Resistance (Ohms)"] = scaler.fit_transform(df[["Resistance (Ohms)"]])

    # Save to new file
    output_file = os.path.join(output_dir, f"cleaned_{file}")
    df.to_csv(output_file, index=False)
    print(f"✅ Saved cleaned data to {output_file}")