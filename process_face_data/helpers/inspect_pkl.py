import pandas as pd
import pickle
from pathlib import Path

PKL_FILE_PATH = Path("./final_facial_data_processed_p1/sentence_001_facial_processed_bagts.pkl")
OUTPUT_CSV_PATH = Path("./final_facial_data_processed_p1/sentence_001_facial_processed_bagts_INSPECT.csv")

N_ROWS_TO_DISPLAY = 5

def inspect_pkl_to_csv(pkl_path, csv_output_path=None, display_rows=5):
    print(f"--- Inspecting PKL File: {pkl_path} ---")

    if not pkl_path.exists():
        print(f"Error: PKL file not found at '{pkl_path}'")
        return

    try:
        with open(pkl_path, 'rb') as f:
            data_object = pickle.load(f)
        print(f"Successfully loaded data from '{pkl_path}'.")
    except Exception as e:
        print(f"Error loading PKL file '{pkl_path}': {e}")
        return

    if not isinstance(data_object, pd.DataFrame):
        print(f"Error: The loaded object is not a Pandas DataFrame. It's a {type(data_object)}.")
        print("This script expects a DataFrame as saved by 'prepare_facial_data_from_bag_timestamps.py'.")
        return

    df: pd.DataFrame = data_object
    print("\n--- DataFrame Info ---")
    df.info(verbose=True, show_counts=True)

    print(f"\n--- DataFrame Shape ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print(f"\n--- DataFrame Index ---")
    print(f"Index Type: {type(df.index)}")
    if isinstance(df.index, pd.TimedeltaIndex):
        print(f"Index Name: {df.index.name}")
        print(f"Index Min: {df.index.min()}")
        print(f"Index Max: {df.index.max()}")
        print(f"Index is monotonic increasing: {df.index.is_monotonic_increasing}")
        if df.index.has_duplicates:
            print(f"WARNING: Index contains {df.index.duplicated().sum()} duplicate values!")
    else:
        print("Index is not a TimedeltaIndex as expected.")


    print(f"\n--- First {display_rows} Rows (Head) ---")
    print(df.head(display_rows))

    print(f"\n--- Last {display_rows} Rows (Tail) ---")
    print(df.tail(display_rows))

    print(f"\n--- Summary Statistics (describe) ---")
    if df.shape[1] > 10:
        cols_to_describe = list(df.columns[:3]) + list(df.columns[-3:])
        print(df[cols_to_describe].describe(include='all'))
    else:
        print(df.describe(include='all'))

    print("\n--- Missing Values Check ---")
    missing_values = df.isnull().sum()
    missing_cols = missing_values[missing_values > 0]
    if not missing_cols.empty:
        print("Columns with missing values:")
        print(missing_cols)
    else:
        print("No missing values found in the DataFrame. (Good!)")

    if csv_output_path:
        try:
            df.to_csv(csv_output_path, index=True)
            print(f"\nDataFrame successfully saved to CSV: {csv_output_path}")
        except Exception as e:
            print(f"\nError saving DataFrame to CSV '{csv_output_path}': {e}")

if __name__ == "__main__":
    inspect_pkl_to_csv(PKL_FILE_PATH, OUTPUT_CSV_PATH, N_ROWS_TO_DISPLAY)