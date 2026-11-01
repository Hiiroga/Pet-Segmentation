"""Quick script to peek inside the downloaded ZIP."""
import zipfile
import pathlib

zip_path = pathlib.Path("data/raw/cats-and-dogs-breeds-classification-oxford-dataset.zip")
print(f"ZIP size: {zip_path.stat().st_size/1e6:.1f} MB")
with zipfile.ZipFile(zip_path, "r") as z:
    names = z.namelist()
    print(f"Total entries in ZIP: {len(names)}")
    tops = sorted(set(n.split("/")[0] for n in names))
    print("Top-level entries:")
    for t in tops:
        print(f"  {t}")

