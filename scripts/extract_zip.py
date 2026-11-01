"""Extract the downloaded ZIP into data/raw/."""
import zipfile, pathlib, sys

zip_path = pathlib.Path("data/raw/cats-and-dogs-breeds-classification-oxford-dataset.zip")
dest = pathlib.Path("data/raw")
print(f"Extracting {zip_path.name} -> {dest} ...")
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(dest)
all_files = list(dest.rglob("*"))
files = [f for f in all_files if f.is_file() and f.suffix != ".zip"]
print(f"Extraction complete: {len(files)} non-zip files.")

