import numpy as np
import pandas as pd
import os

# Define the two main files
files_to_process = {
    'engine': 'engine_maps.npz',
    'motor': 'motor_maps.npz'
}

for label, file_path in files_to_process.items():
    if not os.path.exists(file_path):
        print(f"⚠️ Could not find '{file_path}' in this folder.")
        continue

    print(f"\n--- Processing {file_path} into a single CSV ---")
    data = np.load(file_path)

    combined_data = {}

    for key in data.files:
        array = data[key]

        # If it's a simple 1D array, add it as a column
        if array.ndim == 1:
            combined_data[key] = array
        # If it's a 2D map matrix, flatten it or convert it to a tabular format
        elif array.ndim == 2:
            # If the 2D array matches a standard grid, we flatten it to keep it in one table
            combined_data[key] = array.flatten()
        else:
            print(f"⚠️ Skipped '{key}' due to complex higher dimensions.")

    # Try to build a single consolidated table
    try:
        # Pad shorter arrays with NaN if lengths don't match perfectly
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in combined_data.items()]))

        # Save to exactly one CSV file per component
        output_filename = f"{label}_maps_consolidated.csv"
        df.to_csv(output_filename, index=False)
        print(f"✅ Success! Created exactly one file: {output_filename}")

    except Exception as e:
        print(f"❌ Failed to combine data automatically: {e}")

print("\nProcessing complete. Check your sidebar for the two consolidated CSV files!")
