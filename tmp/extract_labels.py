import pandas as pd
import json

test_path = 'datasets/CICIOT23/test/test.csv'
val_path = 'datasets/CICIOT23/validation/validation.csv'

print(f"Reading labels from {test_path}...")
test_labels = pd.read_csv(test_path, usecols=['label'])['label'].unique()

print(f"Reading labels from {val_path}...")
val_labels = pd.read_csv(val_path, usecols=['label'])['label'].unique()

results = {
    "test_labels": sorted(test_labels.tolist()),
    "val_labels": sorted(val_labels.tolist()),
    "count_test": len(test_labels),
    "count_val": len(val_labels)
}

with open('tmp_labels.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Done. Results saved to tmp_labels.json")
