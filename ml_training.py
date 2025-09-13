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
import pandas as pd
import numpy as np
import os
import sqlite3
import sys

#endregion

#region stream data from the database in batches
def stream_data_from_db(db_path, batch_size=1000, table_name='processed_sound_data'):
    batch_records = []
    
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        
        # Get all column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()
        column_names = [col[1] for col in columns_info]
        
        # Query all data
        cursor.execute(f"SELECT * FROM {table_name}")
        
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
                
            batch_records = []
            for row in rows:
                # Create dictionary from row data
                row_dict = dict(zip(column_names, row))
                
                # Extract track_id and genre
                record_id = row_dict.get('track_id', 'unknown')
                genre_value = row_dict.get('genre', '')
                
                # Create data dictionary excluding genre and track_id columns
                data_dict = {}
                for key, value in row_dict.items():
                    if key not in ['genre', 'track_id']:
                        # Convert BLOB data to numeric if possible
                        if value is not None:
                            if isinstance(value, bytes):
                                try:
                                    # Try to decode BLOB as string first
                                    decoded_value = value.decode('utf-8')
                                    # Try to convert to float
                                    data_dict[key] = float(decoded_value)
                                except (UnicodeDecodeError, ValueError):
                                    # If can't decode or convert, assign numeric hash
                                    data_dict[key] = hash(value) % 10000
                            elif isinstance(value, str):
                                try:
                                    data_dict[key] = float(value)
                                except ValueError:
                                    # Assign numeric value based on string hash
                                    data_dict[key] = hash(value) % 10000
                            else:
                                data_dict[key] = float(value) if value is not None else 0.0
                        else:
                            data_dict[key] = 0.0
                
                # Add genre information in the expected format
                if genre_value:
                    data_dict['genre'] = [{'genre_title': genre_value}]
                
                data_to_yield = {
                    'feature_set_name': 'db_data',
                    'id': record_id,
                    'data': data_dict
                }
                
                batch_records.append(data_to_yield)
            
            if batch_records:
                yield batch_records
        
        connection.close()
                
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        yield []
    except Exception as e:
        print(f"Error reading database: {e}")
        yield []

#endregion

if __name__ == "__main__":
    db_path = "sound_database.db"  

    print("Simulating raw data streaming from the database:")

    total_records_processed = 0
    total_batches_processed = 0

    for batch_data in stream_data_from_db(db_path, batch_size=100):
        if not batch_data:
            continue

        total_batches_processed += 1
        print(f"\n--- Raw Batch {total_batches_processed} ---")
        for record_entry in batch_data:
            print(f"  Record ID: {record_entry['id']}")
            #if 'data' in record_entry:  
                #if isinstance(record_entry['data'], dict):
                    #print(f"  Feature Set: {record_entry['feature_set_name']}")
                    #print(f"  Available features: {list(record_entry['data'].keys())}")
                #else:
                    #print(f"Raw Data: {record_entry}")
            #else:
                #print(f"  No data field in record")
            total_records_processed += 1
        # For very large files, uncomment this to limit output
        #if total_batches_processed >= 1: # Print only first 1 batch as an example TODO:comment after to run the whole dataset
            #break

    if total_records_processed > 0:
        print(f"\nCompleted processing {total_records_processed} records across {total_batches_processed} batches.")
    else:
        print("No records were processed. Check your database path and table structure.")

def fixed_size_feature_transform(extracted_data, default_value=0.0):
    """Transform database features to fixed-size feature vector.
    Since database contains single-dimensional statistical values, no need for complex processing."""
    features = []
    for key in sorted(extracted_data.keys()):
        if key in ['genre']:  # Skip genre-related fields
            continue
        value = extracted_data[key]
        
        # Convert all values to float, using default_value for problematic cases
        if isinstance(value, (int, float)):
            features.append(float(value))
        elif isinstance(value, str):
            try:
                features.append(float(value))
            except ValueError:
                features.append(default_value)
        elif value is None:
            features.append(default_value)
        else:
            # For any other type, convert to float or use default
            try:
                features.append(float(value))
            except (ValueError, TypeError):
                features.append(default_value)
    
    return np.nan_to_num(np.array(features, dtype=float), nan=default_value)

#region Train Model
# Define genre mapping
genre_label_map = {
    'Pop': 0,
    'Rock': 1,
    'Metal': 2,
    'Punk': 3
}

# Extract label directly from the streamed data
def label_func_from_streamed_record(record_entry):
    #record_id = record_entry.get('id', 'unknown')
    
    # Debug: Print the entire data structure for problematic records
    genre_list = record_entry['data'].get('genre')
    #Test Prints
    ###print(f"  Type of genre: {type(genre_list)}")
    
    if not genre_list:
        print(f"  → Genre list is empty or None")
        return None
        
    if not isinstance(genre_list, list):
        print(f"  → Genre list is not a list, trying to extract as string...")
        # If it's a string, try to extract genre from it
        if isinstance(genre_list, str):
            print(f"  → Attempting to parse string: '{genre_list}'")
            # Check if it's a direct genre match
            if genre_list in genre_label_map:
                print(f"  → Found direct match: {genre_list}")
                return genre_label_map.get(genre_list)
            else:
                print(f"  → No direct match found for: '{genre_list}'")
        return None

    # If it is a list, check its contents
    print(f"  → Genre list contents: {genre_list}")
    if len(genre_list) == 0:
        print(f"  → Genre list is empty")
        return None
        
    first_item = genre_list[0]
    print(f"  → First item in list: {first_item} (type: {type(first_item)})")
    
    if isinstance(first_item, dict):
        genre_title = first_item.get('genre_title')
        print(f"  → Extracted genre_title: {genre_title}")
    else:
        # If first item is not a dict, maybe it's directly the genre string
        genre_title = first_item
        print(f"  → Using first item as genre: {genre_title}")
    
    if not genre_title:
        print(f"  → No genre_title found")
        return None

    label = genre_label_map.get(genre_title)
    #Test Prints
    #print(f"  → Genre: '{genre_title}' → Label: {label}")
    #print(f"  → Available genre mappings: {list(genre_label_map.keys())}")
    
    return label

FEATURE_LENGTH = None  # Will store the expected feature vector length

def train_scikit_learn_incrementally(data_stream_generator):
    global FEATURE_LENGTH
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
            if 'data' not in record_entry or not isinstance(record_entry['data'], dict):
                print(f"Record {record_entry.get('id')} has no 'data' field.")
                continue

            label = label_func_from_streamed_record(record_entry)
            if label is None:
                print(f"Record {record_entry.get('id')} has no valid genre label.")
                continue

            features = fixed_size_feature_transform(record_entry['data'])
            if features.size == 0:
                print(f"Record {record_entry.get('id')} has no usable features.")
                continue

            X_batch_features.append(features)
            y_batch_labels.append(label)

        if not X_batch_features or not y_batch_labels:
            print(f"Skipping empty or invalid batch {batch_count}.")
            skipped_batches += 1
            continue

        # Set the expected feature length from the first valid feature vector
        if FEATURE_LENGTH is None:
            FEATURE_LENGTH = len(X_batch_features[0])
            print(f"Feature vector length set to: {FEATURE_LENGTH}")

        # Ensure all feature vectors are the same length
        for i, features in enumerate(X_batch_features):
            if len(features) < FEATURE_LENGTH:
                X_batch_features[i] = np.pad(features, (0, FEATURE_LENGTH - len(features)), 'constant')
            elif len(features) > FEATURE_LENGTH:
                X_batch_features[i] = features[:FEATURE_LENGTH]

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


train_generator = stream_data_from_db(
    db_path='sound_database.db',  
    batch_size=128
)

model, scaler = train_scikit_learn_incrementally(train_generator)

#endregion

#region Evaluate Model
def evaluate_incremental_model(model, scaler, test_data_generator, target_labels_func):
    global FEATURE_LENGTH
    if FEATURE_LENGTH is None:
        print("Error: Feature length not set. Model must be trained first.")
        return None

    all_preds = []
    all_true = []
    print(f"Evaluating with feature length: {FEATURE_LENGTH}")

    for batch in test_data_generator:
        X_batch = []
        y_batch = []

        for record_entry in batch:
            features = fixed_size_feature_transform(record_entry['data'])
            if features.size == 0:
                continue

            label = target_labels_func(record_entry)
            if label is None or label == -1:
                continue

            # Ensure feature vector matches expected length
            if len(features) < FEATURE_LENGTH:
                features = np.pad(features, (0, FEATURE_LENGTH - len(features)), 'constant')
            elif len(features) > FEATURE_LENGTH:
                features = features[:FEATURE_LENGTH]

            X_batch.append(features)
            y_batch.append(label)

        if not X_batch:
            continue

        try:
            X_batch = np.array(X_batch, dtype=float)
            y_batch = np.array(y_batch, dtype=int)
            X_batch_scaled = scaler.transform(X_batch)
            preds = model.predict(X_batch_scaled)
            all_preds.extend(preds)
            all_true.extend(y_batch)
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            continue

    if not all_true:
        print("No valid predictions made during evaluation.")
        return None

    # Convert to numpy arrays for evaluation
    y_true = np.array(all_true)
    y_pred = np.array(all_preds)

    print("\nEvaluation Results:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred, zero_division=0))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, output_dict=True)
    }

# Example genre-based label function
GENRE_TO_LABEL = {'Pop': 0, 'Rock': 1, 'Metal': 2, 'Punk': 3}

def genre_label_func(record_entry):
    record_id = record_entry.get('id', 'unknown')
    
    # Debug: Print the entire data structure for problematic records
    genres = record_entry.get('data', {}).get('genre', [])
    
    #Test Prints
    #print(f"DEBUG - Record {record_id} (evaluation):")
    #print(f"  Raw genre value: {genres}")
    #print(f"  Type of genre: {type(genres)}")
    
    if not genres:
        print(f"  → No genres found")
        return -1
        
    if isinstance(genres, list) and len(genres) > 0:
        first_item = genres[0]
        #Test Print
        #print(f"  → First item: {first_item} (type: {type(first_item)})")
        
        if isinstance(first_item, dict):
            genre_title = first_item.get('genre_title')
        #Test Print    
        #print(f"  → Extracted genre_title: {genre_title}")
        else:
            genre_title = first_item
            print(f"  → Using first item as genre: {genre_title}")
    elif isinstance(genres, str):
        genre_title = genres
        print(f"  → Using string as genre: {genre_title}")
    else:
        print(f"  → Unexpected genres format")
        return -1
        
    label = GENRE_TO_LABEL.get(genre_title, -1)
    #Test Prints
    #print(f"  → Genre: '{genre_title}' → Label: {label}")
    #print(f"  → Available genre mappings: {list(GENRE_TO_LABEL.keys())}")
    
    return label

# Recreate test generator for database
test_generator = stream_data_from_db(
    db_path='sound_database.db',  
    batch_size=128
)

# Call evaluation method with correct label function
model_evaluation = evaluate_incremental_model(
    model,
    scaler,
    test_data_generator=test_generator,
    target_labels_func=genre_label_func
)

#endregion