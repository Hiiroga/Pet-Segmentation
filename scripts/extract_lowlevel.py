"""
Low-level ZIP extraction: scan local file headers directly, ignoring the
corrupted central directory.  This is necessary because the Kaggle upload
of this dataset has a central directory that doesn't match the local headers.

We walk the file looking for PK\x03\x04 (local file header signature) and
extract each entry we find.
"""
import struct
import pathlib
import zlib
import io

ZIP_LOCAL_MAGIC = b"PK\x03\x04"
ZIP_DATA_DESCRIPTOR = b"PK\x07\x08"

zip_path = pathlib.Path("data/raw/cats-and-dogs-breeds-classification-oxford-dataset.zip")
dest = pathlib.Path("data/raw")

extracted = 0
skipped_meta = 0
skipped_err = 0
errors = []

def read_local_entry(fh):
    """
    Read one local file header + data at the current position.
    Returns (filename, data_bytes) or raises an exception.
    """
    # Local file header layout:
    # 0  signature         4 bytes  PK\x03\x04
    # 4  version needed    2 bytes
    # 6  general purpose   2 bytes
    # 8  compression       2 bytes
    # 10 mod time          2 bytes
    # 12 mod date          2 bytes
    # 14 crc-32            4 bytes
    # 18 compressed size   4 bytes
    # 22 uncompressed size 4 bytes
    # 26 filename length   2 bytes
    # 28 extra length      2 bytes
    # 30 filename          n bytes
    # 30+n extra           m bytes
    # 30+n+m data          ...

    header = fh.read(26)   # 30 - 4 (signature already consumed)
    if len(header) < 26:
        return None, None

    (version, gp_flag, compression, mod_time, mod_date,
     crc32, comp_size, uncomp_size, fname_len, extra_len) = struct.unpack_from(
        "<HHHHHIIIHHH", b"\x00\x00" + header  # pad 2 bytes to align
    )[1:]  # drop the first H (dummy)

    # Re-unpack correctly
    (version, gp_flag, compression, mod_time, mod_date,
     crc32, comp_size, uncomp_size, fname_len, extra_len) = struct.unpack_from(
        "<HHHHHIIIHHH", header + b"\x00\x00\x00\x00"
    )[:10]

    fname_bytes = fh.read(fname_len)
    extra_bytes = fh.read(extra_len)

    try:
        fname = fname_bytes.decode("utf-8", errors="replace")
    except Exception:
        fname = fname_bytes.decode("latin-1", errors="replace")

    # Data descriptor flag: sizes may be 0 in header, follow in descriptor
    has_descriptor = bool(gp_flag & 0x08)

    if not has_descriptor:
        # Read compressed data of known size
        comp_data = fh.read(comp_size)
    else:
        # Must scan for PK\x07\x08 or PK\x03\x04 or PK\x01\x02
        # to find the end of data
        buf = b""
        while True:
            chunk = fh.read(4096)
            if not chunk:
                break
            buf += chunk
            # Look for data descriptor or next local header
            for marker in (b"PK\x07\x08", b"PK\x03\x04", b"PK\x01\x02", b"PK\x05\x06"):
                idx = buf.find(marker)
                if idx >= 0:
                    # Rewind
                    fh.seek(-(len(buf) - idx), 1)
                    buf = buf[:idx]
                    break
            else:
                continue
            break
        comp_data = buf
        comp_size = len(comp_data)

    if compression == 0:
        data = comp_data
    elif compression == 8:
        try:
            data = zlib.decompress(comp_data, -15)
        except zlib.error as e:
            raise ValueError(f"zlib error: {e}")
    else:
        raise ValueError(f"Unsupported compression method: {compression}")

    return fname, data


with open(zip_path, "rb") as fh:
    file_size = zip_path.stat().st_size
    processed_bytes = 0

    while True:
        # Scan for next local file header
        pos = fh.tell()
        if pos >= file_size - 30:
            break

        magic = fh.read(4)
        if not magic:
            break

        if magic != ZIP_LOCAL_MAGIC:
            # Not a local header — scan forward byte by byte (slow but safe)
            # Back up 3 bytes and try again
            fh.seek(-3, 1)
            continue

        try:
            fname, data = read_local_entry(fh)
        except Exception as exc:
            skipped_err += 1
            errors.append(("?", str(exc)))
            continue

        if fname is None:
            break

        # Skip directories and macOS artifacts
        if fname.endswith("/") or fname.endswith("\\"):
            tgt = dest / fname
            tgt.mkdir(parents=True, exist_ok=True)
            continue

        bname = pathlib.PurePosixPath(fname).name
        if bname.startswith("._") or "__MACOSX" in fname:
            skipped_meta += 1
            continue

        tgt = dest / fname
        tgt.parent.mkdir(parents=True, exist_ok=True)
        try:
            tgt.write_bytes(data)
            extracted += 1
            if extracted % 1000 == 0:
                print(f"  Extracted {extracted:,} files...")
        except Exception as exc:
            skipped_err += 1
            errors.append((fname, str(exc)))

print(f"\nExtracted          : {extracted:,}")
print(f"Skipped (metadata) : {skipped_meta:,}")
print(f"Skipped (errors)   : {skipped_err:,}")
if errors[:5]:
    for n, e in errors[:5]:
        print(f"  ERR: {n}: {e}")

all_files = [f for f in dest.rglob("*") if f.is_file() and f.suffix != ".zip"]
print(f"\nTotal non-zip files in data/raw: {len(all_files):,}")

