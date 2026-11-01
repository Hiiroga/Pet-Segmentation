"""
Robust ZIP extraction using zipfile with allowZip64 and error recovery.

The zippyz Kaggle dataset ZIP has a corrupted central directory where
file-name entries in the central directory don't match the local headers.
We fix this by:
  1. Reading the central directory to get local header offsets.
  2. Seeking to each local header offset.
  3. Reading the actual local header (which has the correct filename).
  4. Decompressing with zlib directly.

This bypasses the central-directory/local-header mismatch entirely.
"""
import struct
import pathlib
import zlib

zip_path = pathlib.Path("data/raw/cats-and-dogs-breeds-classification-oxford-dataset.zip")
dest = pathlib.Path("data/raw")

# ---------------------------------------------------------------------------
# Parse the End-of-Central-Directory (EOCD) record to find CD offset
# ---------------------------------------------------------------------------
EOCD_SIG   = b"PK\x05\x06"
LFH_SIG    = b"PK\x03\x04"   # Local File Header
CDH_SIG    = b"PK\x01\x02"   # Central Directory Header

def find_eocd(fh, file_size):
    """Search backward for EOCD signature."""
    # EOCD is at most 65535+22 bytes from the end
    search_start = max(0, file_size - 65536 - 22)
    fh.seek(search_start)
    data = fh.read()
    idx = data.rfind(EOCD_SIG)
    if idx < 0:
        raise ValueError("EOCD record not found — not a valid ZIP file")
    return search_start + idx

def read_eocd(fh, eocd_offset):
    fh.seek(eocd_offset)
    sig = fh.read(4)
    assert sig == EOCD_SIG
    (disk_num, disk_start, num_cd_this, num_cd_total,
     cd_size, cd_offset, comment_len) = struct.unpack("<HHHHIIH", fh.read(18))
    return cd_offset, num_cd_total

def read_central_directory(fh, cd_offset, num_entries):
    """Return list of (local_header_offset, cd_filename) for each CD entry."""
    fh.seek(cd_offset)
    entries = []
    for _ in range(num_entries):
        sig = fh.read(4)
        if sig != CDH_SIG:
            # Try to re-sync
            break
        (ver_made, ver_needed, gp_flag, compression,
         mod_time, mod_date, crc32, comp_size, uncomp_size,
         fname_len, extra_len, comment_len, disk_start, int_attr, ext_attr,
         lh_offset) = struct.unpack("<HHHHHIIIIHHHHHHII", fh.read(42))
        fname = fh.read(fname_len).decode("utf-8", errors="replace")
        fh.seek(extra_len + comment_len, 1)
        entries.append((lh_offset, fname))
    return entries

def read_local_header_and_data(fh, lh_offset):
    """Read local file header at given offset; return (actual_fname, data)."""
    fh.seek(lh_offset)
    sig = fh.read(4)
    if sig != LFH_SIG:
        raise ValueError(f"Expected LFH at offset {lh_offset}, got {sig.hex()}")

    (ver_needed, gp_flag, compression, mod_time, mod_date,
     crc32, comp_size, uncomp_size, fname_len, extra_len) = struct.unpack("<HHHHHIIIHH", fh.read(26))

    fname = fh.read(fname_len).decode("utf-8", errors="replace")
    fh.seek(extra_len, 1)  # skip extra field

    has_descriptor = bool(gp_flag & 0x08)

    if has_descriptor:
        # Read until next signature when sizes are unknown
        buf = b""
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            buf += chunk
            for marker in (b"PK\x07\x08", b"PK\x03\x04", b"PK\x01\x02"):
                idx = buf.find(marker)
                if idx >= 0:
                    fh.seek(lh_offset + 30 + fname_len + extra_len + idx)
                    buf = buf[:idx]
                    break
            else:
                continue
            break
        comp_data = buf
    else:
        comp_data = fh.read(comp_size)

    if compression == 0:
        data = comp_data
    elif compression == 8:
        data = zlib.decompress(comp_data, -15)
    else:
        raise ValueError(f"Unsupported compression: {compression}")

    return fname, data


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------
extracted = 0
skipped_meta = 0
skipped_err = 0
errors = []

file_size = zip_path.stat().st_size
print(f"ZIP size: {file_size/1e6:.1f} MB")

with open(zip_path, "rb") as fh:
    eocd_offset = find_eocd(fh, file_size)
    cd_offset, num_entries = read_eocd(fh, eocd_offset)
    print(f"CD offset: {cd_offset}, Entries: {num_entries}")

    cd_entries = read_central_directory(fh, cd_offset, num_entries)
    print(f"Read {len(cd_entries)} CD entries")

    for lh_offset, cd_fname in cd_entries:
        # Skip directories
        if cd_fname.endswith("/"):
            tgt = dest / cd_fname
            tgt.mkdir(parents=True, exist_ok=True)
            continue

        # Skip macOS metadata
        bname = pathlib.PurePosixPath(cd_fname).name
        if bname.startswith("._") or "__MACOSX" in cd_fname:
            skipped_meta += 1
            continue

        try:
            actual_fname, data = read_local_header_and_data(fh, lh_offset)
        except Exception as exc:
            skipped_err += 1
            errors.append((cd_fname, f"LH read error: {exc}"))
            continue

        # Use actual_fname from local header (more reliable)
        target_fname = actual_fname if actual_fname and not actual_fname.endswith("/") else cd_fname
        bname2 = pathlib.PurePosixPath(target_fname).name
        if bname2.startswith("._") or "__MACOSX" in target_fname:
            skipped_meta += 1
            continue

        tgt = dest / target_fname
        tgt.parent.mkdir(parents=True, exist_ok=True)
        try:
            tgt.write_bytes(data)
            extracted += 1
            if extracted % 500 == 0:
                print(f"  Extracted {extracted:,} ...")
        except Exception as exc:
            skipped_err += 1
            errors.append((target_fname, str(exc)))

print(f"\nExtracted          : {extracted:,}")
print(f"Skipped (metadata) : {skipped_meta:,}")
print(f"Skipped (errors)   : {skipped_err:,}")
if errors:
    print("First 10 errors:")
    for n, e in errors[:10]:
        print(f"  {n}: {e}")

all_files = [f for f in dest.rglob("*") if f.is_file() and f.suffix != ".zip"]
print(f"\nTotal non-zip files in data/raw: {len(all_files):,}")

# Summary by extension
from collections import Counter
exts = Counter(f.suffix.lower() for f in all_files)
print("By extension:", dict(exts))

