import pandas as pd
import pickle
from pathlib import Path

# --- Configuration ---
# !!! MODIFICĂ ACESTE CĂI PENTRU A SE POTRIVI CU FIȘIERUL TĂU !!!
PKL_FILE_PATH = Path("./final_facial_data_processed_marinela/sentence_001_facial_processed_bagts.pkl") # Calea către fișierul .pkl pe care vrei să-l inspectezi
OUTPUT_CSV_PATH = Path("./final_facial_data_processed_marinela/sentence_001_facial_processed_bagts_INSPECT.csv") # Unde să salvezi CSV-ul (opțional)

# Numărul de rânduri de afișat din capul și coada DataFrame-ului
N_ROWS_TO_DISPLAY = 5

# --- Main Inspection Logic ---
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

    # Verifică tipul obiectului încărcat
    if not isinstance(data_object, pd.DataFrame):
        print(f"Error: The loaded object is not a Pandas DataFrame. It's a {type(data_object)}.")
        print("This script expects a DataFrame as saved by 'prepare_facial_data_from_bag_timestamps.py'.")
        return

    df: pd.DataFrame = data_object
    print("\n--- DataFrame Info ---")
    df.info(verbose=True, show_counts=True) # verbose=True și show_counts=True pentru mai multe detalii

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
    # Afișează statistici pentru câteva coloane (primele și ultimele) pentru a nu fi prea lung
    if df.shape[1] > 10:
        cols_to_describe = list(df.columns[:3]) + list(df.columns[-3:])
        print(df[cols_to_describe].describe(include='all'))
    else:
        print(df.describe(include='all'))

    # Verifică valorile lipsă
    print("\n--- Missing Values Check ---")
    missing_values = df.isnull().sum()
    missing_cols = missing_values[missing_values > 0]
    if not missing_cols.empty:
        print("Columns with missing values:")
        print(missing_cols)
    else:
        print("No missing values found in the DataFrame. (Good!)")

    # Salvează opțional ca CSV
    if csv_output_path:
        try:
            # La salvarea în CSV, indexul (care e Timedelta) va fi scris ca o coloană
            # Poți alege să salvezi milisecundele indexului dacă preferi un număr
            # df_to_save = df.copy()
            # df_to_save.index = df_to_save.index.total_seconds() * 1000 # Convertește Timedelta în ms
            # df_to_save.index.name = 'normalized_timestamp_ms'
            # df_to_save.to_csv(csv_output_path, index=True) # Salvează cu indexul ca o coloană
            df.to_csv(csv_output_path, index=True) # Salvează cu TimedeltaIndex ca o coloană
            print(f"\nDataFrame successfully saved to CSV: {csv_output_path}")
        except Exception as e:
            print(f"\nError saving DataFrame to CSV '{csv_output_path}': {e}")

if __name__ == "__main__":
    inspect_pkl_to_csv(PKL_FILE_PATH, OUTPUT_CSV_PATH, N_ROWS_TO_DISPLAY)