import ijson
import pandas as pd
from collections import defaultdict
import os

def convert_nested_json_to_multiple_excels(json_file_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Structure: {feature_name: {id: {inner_key: value}}}
    data = defaultdict(lambda: defaultdict(dict))

    with open(json_file_path, 'r', encoding='utf-8') as file:
        objects = ijson.items(file, 'item')

        for obj in objects:
            for feature_name, entries in obj.items():
                for id_key, value_dict in entries.items():
                    if isinstance(value_dict, dict):
                        for inner_key, inner_val in value_dict.items():
                            data[feature_name][id_key][inner_key] = inner_val

    # Create one Excel file per top-level key (feature_name)
    for feature_name, records in data.items():
        df = pd.DataFrame.from_dict(records, orient='index')
        df.index.name = 'id'
        df.reset_index(inplace=True)

        output_path = os.path.join(output_dir, f"{feature_name}.xlsx")
        df.to_excel(output_path, index=False)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    convert_nested_json_to_multiple_excels("processed_audio_features.json", "excel_outputs")