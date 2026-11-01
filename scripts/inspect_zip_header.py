"""Inspect the first bytes and try multiple extraction strategies."""
import pathlib, struct, zipfile, gzip, shutil

zip_path = pathlib.Path("data/raw/cats-and-dogs-breeds-classification-oxford-dataset.zip")
dest = pathlib.Path("data/raw")

# Read magic bytes
with open(zip_path, "rb") as f:
    header = f.read(16)
print(f"First 16 bytes: {header.hex()}")
print(f"File size: {zip_path.stat().st_size:,} bytes")

# Check signatures
if header[:4] == b"PK\x03\x04":
    print("Detected: Standard ZIP (PK signature)")
elif header[:2] == b"\x1f\x8b":
    print("Detected: GZIP stream")
elif header[:7] == b"Rar!\x1a\x07\x00":
    print("Detected: RAR archive")
elif header[:6] in (b"7z\xbc\xaf'\x1c", ):
    print("Detected: 7-zip archive")
else:
    print(f"Unknown format. Hex: {header.hex()}")

# Try to test ZIP integrity at various offsets
print("\nSearching for PK signature offset in file...")
with open(zip_path, "rb") as f:
    content = f.read(1024)
idx = content.find(b"PK\x03\x04")
if idx >= 0:
    print(f"  PK\\x03\\x04 found at offset {idx}")
else:
    print("  PK\\x03\\x04 NOT found in first 1024 bytes")

