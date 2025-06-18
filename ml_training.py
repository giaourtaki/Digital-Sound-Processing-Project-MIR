
#region Imports
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sklearn as skl 
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm, sklearn.model_selection, sklearn.metrics 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ijson
import numpy as np
import os
import json

#endregion
# No need for SGDClassifier or StandardScaler imports if not actually using them
# from sklearn.linear_model import SGDClassifier
# from sklearn.preprocessing import StandardScaler

#region stream data from the large JSON file in batches
def stream_raw_json_data(json_file_path, ijson_prefix='item', keys_to_extract=None, batch_size=1000):
    batch_records = []
    try:
        with open(json_file_path, 'rb') as file:
            parser = ijson.items(file, ijson_prefix)

            # Now, 'full_record_dict' will be the entire top-level object like
            # {"yin_processed_data_pitch": {"42851": {...}}}
            # or {"processed_data_melodia": {"12345": {...}}}
            for full_record_dict in parser:
                if not isinstance(full_record_dict, dict):
                    continue

                # Iterate through the top-level keys (e.g., "yin_processed_data_pitch", "processed_data_melodia")
                for feature_set_name, entries_by_id in full_record_dict.items():
                    if not isinstance(entries_by_id, dict):
                        continue # Skip if the value under the feature_set_name is not a dictionary of IDs

                    # Now iterate through the individual records within this feature set
                    for record_id, record_data in entries_by_id.items():
                        data_to_yield = {'feature_set_name': feature_set_name, 'id': record_id}
                        if keys_to_extract:
                            extracted = {key: record_data[key] for key in keys_to_extract if key in record_data}
                            data_to_yield['data'] = extracted
                        else:
                            data_to_yield['data'] = record_data

                        batch_records.append(data_to_yield)
                        if len(batch_records) >= batch_size:
                            yield batch_records
                            batch_records = []

            if batch_records:
                yield batch_records

    except FileNotFoundError:
        yield []
    except Exception as e:
        yield []
if __name__ == "__main__":
    your_json_file_path = "processed_audio_features.json"

    print("Simulating raw data streaming from the JSON file:")

    total_records_processed = 0
    total_batches_processed = 0

    for batch_data in stream_raw_json_data(your_json_file_path, batch_size=100):
        if not batch_data:
            continue

        total_batches_processed += 1
        print(f"\n--- Raw Batch {total_batches_processed} ---")
        for record_entry in batch_data:
            #print(f"  Record ID: {record_entry['id']}")
            #if 'data' in record_entry:  
               # if isinstance(record_entry['data'], dict):
                    #print(f"  Feature Set: {record_entry['feature_set_name']}")   
               # else:
                    #print(f"Raw Data: {(record_entry)}")
           # else:
                #print(f"  NK print Raw Data: {list(record_entry['data'])}")
            total_records_processed += 1
        # For very large files, uncomment this to limit output
        if total_batches_processed >= 2: # Print only first 2 batches as an example
            break


    if total_records_processed > 0:
        print(f"\nCompleted processing {total_records_processed} records across {total_batches_processed} batches.")
    else:
        print("No records were processed. Check your file path and JSON structure.")

def fixed_size_feature_transform(extracted_data, list_stats=['mean', 'std', 'min', 'max'], default_value=0.0):

    features = []
    for key in sorted(extracted_data.keys()):
        if key == 'genre':
            continue  # Skip genre, it's not a numerical feature
        value = extracted_data[key]
        if isinstance(value, list):
            try:
                arr = np.array(value, dtype=float)
            except ValueError:
                arr = np.array([], dtype=float)
            if arr.size == 0:
                features.extend([default_value] * len(list_stats))
            else:
                for stat_type in list_stats:
                    try:
                        if stat_type == 'mean':
                            features.append(np.nanmean(arr))
                        elif stat_type == 'std':
                            features.append(np.nanstd(arr))
                        elif stat_type == 'min':
                            features.append(np.nanmin(arr))
                        elif stat_type == 'max':
                            features.append(np.nanmax(arr))
                        elif stat_type == 'median':
                            features.append(np.nanmedian(arr))
                        else:
                            features.append(default_value)
                    except:
                        features.append(default_value)
        elif isinstance(value, (int, float)):
            features.append(float(value))
    return np.nan_to_num(np.array(features, dtype=float), nan=default_value)
#endregion

#region Train Model
# Embedded label function using streamed JSON data directly
# Define genre mapping
genre_label_map = {
    'Pop': 0,
    'Rock': 1,
    'Metal': 2,
    'Punk': 3
}

# Extract label directly from the streamed data
# This assumes each streamed record's data includes genre info like:
# 'data': { ..., 'genre': [{'genre_id': '10', 'genre_title': 'Pop', ...}] }
def label_func_from_streamed_record(record_entry):

    genre_list = record_entry['data'].get('track_genres')
    if not genre_list or not isinstance(genre_list, list):
        return None

    genre_title = genre_list[0].get('genre_title') if genre_list else None
    if not genre_title:
        return None

    return genre_label_map.get(genre_title)


def train_scikit_learn_incrementally(data_stream_generator):
    model = SGDClassifier(loss='log_loss', random_state=42, warm_start=True)
    scaler = StandardScaler()
    first_batch_processed = False
    all_possible_classes = np.array([0, 1, 2, 3])

    print("Starting incremental training...")
    batch_count = 0
    total_records_trained = 0
    skipped_batches = 0

    for batch_of_raw_records in data_stream_generator:
        if not batch_of_raw_records:
            continue

        batch_count += 1
        X_batch_features = []
        y_batch_labels = []

        for record_entry in batch_of_raw_records:
            # Ensure essential keys exist
            if 'data' not in record_entry or not isinstance(record_entry['data'], dict):
                print(f"Record {record_entry.get('id')} has no 'data' field.")
                continue

            label = label_func_from_streamed_record(record_entry)
            if label is None:
                print(f"Record {record_entry.get('id')} has no valid genre label.")
                continue

            features = fixed_size_feature_transform(record_entry['data'])
            print(f"Record {record_entry.get('id')} has no usable features.")
            if features.size == 0:
                continue

            X_batch_features.append(features)
            y_batch_labels.append(label)

        if not X_batch_features or not y_batch_labels:
            print(f"Skipping empty or invalid batch {batch_count}.")
            skipped_batches += 1
            continue

        X_batch = np.array(X_batch_features, dtype=float)
        y_batch = np.array(y_batch_labels, dtype=int)

        if not first_batch_processed:
            scaler.partial_fit(X_batch)
            X_batch_scaled = scaler.transform(X_batch)
            model.partial_fit(X_batch_scaled, y_batch, classes=all_possible_classes)
            first_batch_processed = True
            print(f"Batch {batch_count}: Initial partial_fit performed with {len(y_batch)} samples.")
        else:
            X_batch_scaled = scaler.transform(X_batch)
            model.partial_fit(X_batch_scaled, y_batch)
            print(f"Batch {batch_count}: Subsequent partial_fit performed with {len(y_batch)} samples.")

        total_records_trained += len(y_batch)

    if total_records_trained == 0:
        print("No records were trained. Make sure your data includes valid features and genre labels.")
    else:
        print(f"\nIncremental training complete. Total records trained: {total_records_trained} from {batch_count} batches "
            f"({skipped_batches} skipped).")

    return model, scaler

# Usage:
train_generator = stream_raw_json_data(
    json_file_path='processed_audio_features.json',
    keys_to_extract=['track_genres', 'processed_data_key'],  # genre must be extracted too
    batch_size=128
)

model, scaler = train_scikit_learn_incrementally(train_generator)

#endregion
#region Evaluate Model
def evaluate_incremental_model(model, scaler, test_data_generator, target_labels_func):

    all_preds = []
    all_true = []

    for batch in test_data_generator:
        X_batch = []
        y_batch = []

        for record_entry in batch:
            features = fixed_size_feature_transform(record_entry['data'])
            if features.size == 0:
                continue

            label = target_labels_func(record_entry)
            X_batch.append(features)
            y_batch.append(label)

        if not X_batch:
            continue

        X_batch = np.array(X_batch, dtype=float)
        y_batch = np.array(y_batch, dtype=int)
        X_batch_scaled = scaler.transform(X_batch)

        preds = model.predict(X_batch_scaled)
        all_preds.extend(preds)
        all_true.extend(y_batch)

    # Convert to numpy arrays for evaluation
    y_true = np.array(all_true)
    y_pred = np.array(all_preds)

    print("\nEvaluation Results:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, output_dict=True)
    }

# Example genre-based label function
GENRE_TO_LABEL = {'Pop': 0, 'Rock': 1, 'Metal': 2, 'Punk': 3}

def genre_label_func(record_entry):
    genres = record_entry.get('data', {}).get('genre', [])
    if genres and isinstance(genres, list):
        genre_title = genres[0].get('genre_title')
        return GENRE_TO_LABEL.get(genre_title, -1)  
    return -1

# Recreate test generator (ensure 'genre' is in keys_to_extract)
test_generator = stream_raw_json_data(
    json_file_path='processed_audio_features.json',
    keys_to_extract=['track_genres', 'processed_data_key'], 
    batch_size=128,
)

# Call evaluation method with correct label function
model_evaluation = evaluate_incremental_model(
    model,
    scaler,
    test_data_generator=test_generator,
    target_labels_func=genre_label_func
)  # Now fully aligned with your training setup

#endregion