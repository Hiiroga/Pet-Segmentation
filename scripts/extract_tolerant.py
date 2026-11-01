"""
Tolerant extraction: skip bad entries, report them.
Uses Python's zipfile with individual member extraction to skip corrupt entries.
"""
import zipfile, pathlib, sys

zip_path = pathlib.Path("data/raw/cats-and-dogs-breeds-classification-oxford-dataset.zip")
dest = pathlib.Path("data/raw")

skipped = []
extracted = 0

with zipfile.ZipFile(zip_path, "r") as z:
    members = z.infolist()
    print(f"ZIP reports {len(members)} entries.")
    for info in members:
        target = dest / info.filename
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with z.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            extracted += 1
        except Exception as e:
            skipped.append((info.filename, str(e)))

print(f"\nExtracted : {extracted}")
print(f"Skipped   : {len(skipped)}")
if skipped:
    print("Skipped files:")
    for name, err in skipped[:20]:
        print(f"  {name}: {err}")

