"""
Smart extraction: skip macOS __MACOSX metadata and ._* AppleDouble files.
These are Apple resource fork artifacts that cause BadZipFile on the entries
immediately following them in the ZIP central directory.

The strategy: use allowZip64=True and skip entries whose name starts with
'._' or contains '/__MACOSX/', then extract the rest normally.
"""
import zipfile, pathlib

zip_path = pathlib.Path("data/raw/cats-and-dogs-breeds-classification-oxford-dataset.zip")
dest = pathlib.Path("data/raw")

skipped_meta = 0
skipped_err = 0
extracted = 0
errors = []

# Use a raw read approach: open the zip, iterate entries and read their
# local file data directly to avoid the central-directory mismatch issue.
with open(zip_path, "rb") as raw_fh:
    # Pass the raw file handle; allowZip64 handles large zips
    with zipfile.ZipFile(raw_fh, "r", allowZip64=True) as z:
        all_members = z.infolist()
        print(f"ZIP central directory: {len(all_members)} entries")

        for info in all_members:
            fname = info.filename

            # Skip macOS metadata artifacts
            basename = pathlib.PurePosixPath(fname).name
            if basename.startswith("._") or "__MACOSX" in fname:
                skipped_meta += 1
                continue

            target = dest / fname

            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)

            # Try reading the local file entry
            try:
                data = z.read(info)
                target.write_bytes(data)
                extracted += 1
            except Exception as exc:
                skipped_err += 1
                errors.append((fname, str(exc)))

print(f"\nExtracted          : {extracted:,}")
print(f"Skipped (metadata) : {skipped_meta:,}")
print(f"Skipped (errors)   : {skipped_err:,}")

if errors:
    print(f"\nFirst 10 errors:")
    for name, err in errors[:10]:
        print(f"  {name}: {err}")

# Summary of what landed
all_extracted = [f for f in dest.rglob("*") if f.is_file() and f.suffix != ".zip"]
print(f"\nTotal non-zip files now in data/raw: {len(all_extracted):,}")

