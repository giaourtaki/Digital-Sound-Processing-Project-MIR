import json
import pandas as pd
import os
from typing import Dict, List, Any, Optional

# Configure logging

class JSONReassembler:
    """
    A class to reassemble shattered JSON data and prepare it for machine learning models.
    
    Input JSON structure: [{"variable_1": {"row_id": {"column"}}}]
    Output structure: {"row_id": "row_1", "variable_1": "value_1", "variable_2": "value_2", ...}    """
    
    def __init__(self):
        self.reassembled_data = []
        self.all_variables = set()
        self.all_row_ids = set()
    
    def load_json(self, json_file_path: str) -> List[Dict]:
        """Load JSON data from file."""
        try:
            print(f"Loading JSON data from {json_file_path}...")
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Successfully loaded {len(data)} items from JSON")
            return data
        except FileNotFoundError:
            print(f"Error: File {json_file_path} not found")
            return []
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in {json_file_path}")
            return []
    
    def _flatten_nested_dict(self, nested_dict: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """
        Flatten a nested dictionary for ML-friendly format.
        
        Args:
            nested_dict: Dictionary to flatten
            parent_key: Parent key for nested structure
            sep: Separator for flattened keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for key, value in nested_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                items.extend(self._flatten_nested_dict(value, new_key, sep=sep).items())
            elif isinstance(value, list):                # Handle lists by creating separate columns for each element
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        items.extend(self._flatten_nested_dict(item, f"{new_key}_{i}", sep=sep).items())
                    else:
                        items.append((f"{new_key}_{i}", item))
            else:
                items.append((new_key, value))
        
        return dict(items)
    
    def reassemble_json_data(self, shattered_data: List[Dict], flatten_nested: bool = True) -> List[Dict]:
        """
        Reassemble shattered JSON data into a structure suitable for ML models.
        
        Args:
            shattered_data: List of dictionaries with structure [{"variable_1": {"row_id": {"column"}}}]
            flatten_nested: Whether to flatten nested structures for ML compatibility
        
        Returns:
            List of dictionaries with structure [{"row_id": "row_1", "variable_1": "value_1", ...}]
        """
        print("Starting data reassembly...")
        # Dictionary to store reassembled data: {row_id: {variable: value}}
        data_dict = {}
        
        # Process each item in the shattered data
        print(f"Processing {len(shattered_data)} data items...")
        for i, item in enumerate(shattered_data):
            if i % 10 == 0:  # Progress logging every 10 items
                print(f"Processing item {i+1}/{len(shattered_data)}")
            
            for variable_name, row_data in item.items():
                self.all_variables.add(variable_name)
                
                for row_id, column_data in row_data.items():
                    self.all_row_ids.add(row_id)
                    
                    # Initialize row if not exists
                    if row_id not in data_dict:
                        data_dict[row_id] = {"row_id": row_id}
                    
                    # Handle different types of column data
                    if isinstance(column_data, dict):
                        if flatten_nested:
                            # Flatten nested dictionaries
                            flattened = self._flatten_nested_dict(column_data)
                            for flat_key, flat_value in flattened.items():
                                full_key = f"{variable_name}_{flat_key}" if flat_key else variable_name
                                data_dict[row_id][full_key] = flat_value
                                self.all_variables.add(full_key)
                        else:
                            # Keep as single value (convert to string if complex)
                            if len(column_data) == 1:
                                value = list(column_data.values())[0]
                            else:
                                value = str(column_data)
                            data_dict[row_id][variable_name] = value
                    elif isinstance(column_data, list):
                        # Keep arrays as single column - convert to string representation
                        data_dict[row_id][variable_name] = str(column_data)
                    else:
                        # Direct value
                        data_dict[row_id][variable_name] = column_data
        
        # Convert to list of dictionaries
        self.reassembled_data = list(data_dict.values())
        print(f"Data reassembly completed. Total tracks: {len(self.reassembled_data)}")
        print(f"Total variables: {len(self.all_variables)}")
        
        return self.reassembled_data
    
    def save_to_csv(self, output_csv_path: str, data: Optional[List[Dict]] = None) -> bool:
        """
        Save reassembled data to CSV file.
        
        Args:
            output_csv_path: Path to save the CSV file
            data: Data to save (uses self.reassembled_data if None)
        
        Returns:
            bool: True if successful, False otherwise
        """
        if data is None:
            data = self.reassembled_data
        if not data:
            return False
        
        try:
            df = pd.DataFrame(data)
            df.to_csv(output_csv_path, index=False)
            return True
        except Exception as e:
            return False
    
    def load_and_merge_csv(self, csv_file_path: str) -> pd.DataFrame:
        """
        Load existing CSV file and merge with reassembled data.
        
        Args:
            csv_file_path: Path to the existing CSV file
        
        Returns:
            pd.DataFrame: Merged dataframe
        """
        try:
            print(f"Loading existing CSV from {csv_file_path}...")
            # Load existing CSV
            existing_df = pd.read_csv(csv_file_path)
            print(f"Loaded existing CSV with {len(existing_df)} rows and {len(existing_df.columns)} columns")
            print(f"Existing CSV columns: {list(existing_df.columns)}")
            
            # Convert reassembled data to DataFrame
            if not self.reassembled_data:
                print("No reassembled data available, returning existing CSV")
                return existing_df
            
            print(f"Converting {len(self.reassembled_data)} reassembled records to DataFrame...")
            new_df = pd.DataFrame(self.reassembled_data)
            print(f"New DataFrame has {len(new_df)} rows and {len(new_df.columns)} columns")
            print(f"New DataFrame columns: {list(new_df.columns)}")
            
            # Create a proper identifier for merging
            # First, check if there's a track_id or similar column in existing CSV
            merge_column = None
            if 'track_id' in existing_df.columns:
                merge_column = 'track_id'
                # Map row_id to track_id format if needed
                if 'row_id' in new_df.columns:
                    new_df['track_id'] = new_df['row_id']
            elif 'id' in existing_df.columns:
                merge_column = 'id'
                if 'row_id' in new_df.columns:
                    new_df['id'] = new_df['row_id']
            elif 'row_id' in existing_df.columns:
                merge_column = 'row_id'
            else:
                # Create a merge key based on index
                existing_df['merge_key'] = existing_df.index.astype(str)
                new_df['merge_key'] = new_df['row_id'] if 'row_id' in new_df.columns else new_df.index.astype(str)
                merge_column = 'merge_key'
            
            print(f"Using '{merge_column}' as merge column")
            
            # Fix data type mismatch before merging
            if merge_column in existing_df.columns and merge_column in new_df.columns:
                print(f"Existing column type: {existing_df[merge_column].dtype}")
                print(f"New column type: {new_df[merge_column].dtype}")
                
                # Convert both columns to string to avoid type mismatch
                existing_df[merge_column] = existing_df[merge_column].astype(str)
                new_df[merge_column] = new_df[merge_column].astype(str)
                
                print(f"After conversion - Existing: {existing_df[merge_column].dtype}, New: {new_df[merge_column].dtype}")
                
                # Merge dataframes - use 'left' join to keep all original CSV data
                merged_df = pd.merge(existing_df, new_df, on=merge_column, how='left', suffixes=('', '_new'))
                print(f"Merged successfully using column '{merge_column}'")
            else:
                print("Could not find suitable merge column, concatenating side by side")
                # If no suitable merge column, just concatenate
                merged_df = pd.concat([existing_df, new_df], axis=1)
            
            print(f"Merged DataFrame has {len(merged_df)} rows and {len(merged_df.columns)} columns")
            return merged_df
            
        except FileNotFoundError:
            print(f"CSV file {csv_file_path} not found")
            return pd.DataFrame(self.reassembled_data) if self.reassembled_data else pd.DataFrame()
        except Exception as e:
            print(f"Error during CSV merge: {str(e)}")
            return pd.DataFrame(self.reassembled_data) if self.reassembled_data else pd.DataFrame()
    def prepare_for_sklearn(self, df: pd.DataFrame, target_column: Optional[str] = None, 
                        handle_missing: str = 'drop', encode_categoricals: bool = True) -> tuple:
        """
        Prepare data for scikit-learn models with comprehensive preprocessing.
        
        Args:mer
            df: DataFrame to prepare
            target_column: Name of the target column (if any)
            handle_missing: How to handle missing values ('drop', 'mean', 'median', 'mode', 'zero')
            encode_categoricals: Whether to encode categorical variables
        
        Returns:
            tuple: (features_df, target_series, preprocessing_info)
        """
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.impute import SimpleImputer
        
        # Create a copy to avoid modifying original
        df_copy = df.copy()
        preprocessing_info = {
            'label_encoders': {},
            'scaler': None,
            'imputers': {},
            'dropped_columns': [],
            'feature_names': []
        }
        
        # Remove row_id column as it's not a feature
        if 'row_id' in df_copy.columns:
            df_copy = df_copy.drop(columns=['row_id'])
            preprocessing_info['dropped_columns'].append('row_id')
        
        # Separate target if specified
        target = None
        if target_column and target_column in df_copy.columns:
            target = df_copy[target_column].copy()
            df_copy = df_copy.drop(columns=[target_column])
            preprocessing_info['target_column'] = target_column
        
        # Handle missing values
        if handle_missing != 'drop':
            for column in df_copy.columns:
                if df_copy[column].isnull().any():
                    if df_copy[column].dtype in ['object', 'category']:
                        # Categorical columns - use mode
                        imputer = SimpleImputer(strategy='most_frequent')
                    elif handle_missing == 'mean':
                        imputer = SimpleImputer(strategy='mean')
                    elif handle_missing == 'median':
                        imputer = SimpleImputer(strategy='median')
                    elif handle_missing == 'zero':
                        imputer = SimpleImputer(strategy='constant', fill_value=0)
                    else:
                        continue
                    
                    df_copy[column] = imputer.fit_transform(df_copy[[column]]).ravel()
                    preprocessing_info['imputers'][column] = imputer
        else:
            # Drop rows with missing values
            initial_rows = len(df_copy)
            df_copy = df_copy.dropna()
            if target is not None:
                target = target.loc[df_copy.index]
        
        # Encode categorical variables
        if encode_categoricals:
            categorical_columns = df_copy.select_dtypes(include=['object', 'category']).columns
            for column in categorical_columns:
                le = LabelEncoder()
                df_copy[column] = le.fit_transform(df_copy[column].astype(str))
                preprocessing_info['label_encoders'][column] = le
        
        # Convert boolean columns to integers
        boolean_columns = df_copy.select_dtypes(include=['bool']).columns
        for column in boolean_columns:
            df_copy[column] = df_copy[column].astype(int)
          # Store feature names
        preprocessing_info['feature_names'] = list(df_copy.columns)
        
        return df_copy, target, preprocessing_info
    
    def apply_preprocessing(self, df: pd.DataFrame, preprocessing_info: Dict) -> pd.DataFrame:
        """
        Apply previously fitted preprocessing to new data.
        
        Args:
            df: New data to preprocess
            preprocessing_info: Preprocessing information from prepare_for_sklearn
            
        Returns:
            Preprocessed DataFrame
        """
        df_copy = df.copy()
        
        # Remove row_id if present
        if 'row_id' in df_copy.columns:
            df_copy = df_copy.drop(columns=['row_id'])
        
        # Apply imputers
        for column, imputer in preprocessing_info.get('imputers', {}).items():
            if column in df_copy.columns:
                df_copy[column] = imputer.transform(df_copy[[column]]).ravel()
        
        # Apply label encoders
        for column, encoder in preprocessing_info.get('label_encoders', {}).items():
            if column in df_copy.columns:
                # Handle unseen categories
                unique_values = set(df_copy[column].astype(str))
                known_values = set(encoder.classes_)
                unknown_values = unique_values - known_values
                if unknown_values:
                    # Replace unknown values with the most frequent known value
                    most_frequent = encoder.classes_[0]  # First class is typically most frequent
                    df_copy[column] = df_copy[column].astype(str).replace(list(unknown_values), most_frequent)
                
                df_copy[column] = encoder.transform(df_copy[column].astype(str))
        
        # Ensure all expected columns are present
        expected_features = preprocessing_info.get('feature_names', [])
        for feature in expected_features:
            if feature not in df_copy.columns:
                df_copy[feature] = 0  # Add missing columns with default value
        
        # Reorder columns to match training data
        df_copy = df_copy[expected_features]
        
        return df_copy
    def display_sample(self, data: List[Dict], num_samples: int = 5):
        """Display sample of the data."""
        if not data:
            return
        
        sample_data = []
        for i, row in enumerate(data[:num_samples]):
            sample_data.append(f"Row {i+1}: {row}")
        return sample_data


def main():
    """Main function to demonstrate usage."""
    # Initialize the reassembler
    reassembler = JSONReassembler()
    
    # Example usage
    json_file = 'processed_audio_features.json'
    
    if not json_file:
        # Create example data for demonstration
        example_data = [
            {"variable_1": {"row_1": {"value": "A"}, "row_2": {"value": "B"}}},
            {"variable_2": {"row_1": {"score": 85}, "row_2": {"score": 92}}},
            {"variable_3": {"row_1": {"status": "active"}, "row_2": {"status": "inactive"}}}
        ]
        
        print("Using example data:")
        for item in example_data:
            print(item)
        
        # Reassemble the data
        reassembled = reassembler.reassemble_json_data(example_data)
        reassembler.display_sample(reassembled)
        
    else:
        if os.path.exists(json_file):
            # Load and process real JSON file
            shattered_data = reassembler.load_json(json_file)            
            if shattered_data:
                reassembled = reassembler.reassemble_json_data(shattered_data)
                reassembler.display_sample(reassembled)
        else:
            print(f"File {json_file} does not exist.")
            return
    
    # Process CSV operations - only use merge feature
    print("Starting CSV operations...")
    csv_operation = 2
    
    if csv_operation == 1:
        print("Saving reassembled data to CSV...")
        output_path = "output.csv"
        success = reassembler.save_to_csv(output_path)
        if success:
            print(f"Data saved to {output_path}")
        else:            print("Failed to save data to CSV")
        
    elif csv_operation == 2:
        print("Attempting to merge with existing CSV...")
        existing_csv = 'modified_raw_tracks_genres.csv'
        
        if existing_csv and os.path.exists(existing_csv):
            print(f"Found existing CSV: {existing_csv}")
            merged_df = reassembler.load_and_merge_csv(existing_csv)
            
            # Save the merged data without filtering
            output_path = "merged_output_of_extracted_features.csv"
            merged_df.to_csv(output_path, index=False)
            print(f"Merged data saved to {output_path}")
            print(f"Final CSV contains {len(merged_df)} rows and {len(merged_df.columns)} columns")
            
            # Show sample of first few columns to verify
            print("\nSample of merged data (first 3 rows, first 10 columns):")
            print(merged_df.head(3).iloc[:, :10])
            
        else:
            print(f"CSV file {existing_csv} not found. Creating new CSV with reassembled data...")
            output_path = "reassembled_data.csv"
            success = reassembler.save_to_csv(output_path)
            if success:
                print(f"Reassembled data saved to {output_path}")                # Check the saved file
                import pandas as pd
                df = pd.read_csv(output_path)
                print(f"Saved CSV contains {len(df)} rows and {len(df.columns)} columns")
            else:
                print("Failed to save reassembled data to CSV")
    
    print("Process completed successfully!")


if __name__ == "__main__":
    main()