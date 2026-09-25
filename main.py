from pathlib import Path
import pandas as pd

root = Path("./data/student_resource")

files = [
    root / "dataset/train/train_source1.tsv",
    root / "dataset/train/train_source2.tsv",
    root / "dataset/train/train_source3.tsv",
    root / "dataset/train/train_ground_truth.tsv",
    root / "dataset/test/test_source1.tsv",
    root / "dataset/test/test_source2.tsv",
    root / "dataset/test/test_source3.tsv",
]

for path in files:
    print(f"\n--- {path} ---")

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(3).to_string(index=False))
