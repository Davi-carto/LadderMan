"""Losslessly archive/restore TensorBoard events, JSONL data and console logs.

Only generated archives/restored copies are written; pack never changes sources.
Gzip streams are split into <50 MiB pieces to fit ordinary GitHub repositories.
"""

import argparse
import hashlib
import json
from pathlib import Path
import zlib


CHUNK = 1024 * 1024
PART_BYTES = 48 * CHUNK
REPO = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = REPO / "training_artifacts/logs_snapshot_2026-10-19"


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside destination: {relative}")
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(CHUNK):
            digest.update(data)
    return digest.hexdigest()


def pack_file(source, root, archive):
    before = source.stat()
    relative = source.relative_to(root).as_posix()
    compressor = zlib.compressobj(level=6, wbits=31)
    digest = hashlib.sha256()
    parts = []
    output = None
    count = 0

    def write(data):
        nonlocal output, count
        data = memoryview(data)
        while data:
            if output is None or output.tell() == PART_BYTES:
                if output is not None:
                    output.close()
                name = f"{relative}.gz.part{len(parts):03d}"
                target = safe_path(archive, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                output = target.open("xb")
                parts.append(name)
            size = min(len(data), PART_BYTES - output.tell())
            output.write(data[:size])
            count += size
            data = data[size:]

    try:
        with source.open("rb") as stream:
            while data := stream.read(CHUNK):
                digest.update(data)
                write(compressor.compress(data))
        write(compressor.flush())
    finally:
        if output is not None:
            output.close()
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed during snapshot: {source}")
    return dict(path=relative, bytes=before.st_size, sha256=digest.hexdigest(),
                compressed_bytes=count, parts=parts)


def pack(root, archive):
    # Refuse to overwrite any previous snapshot.
    archive.mkdir(parents=True, exist_ok=False)
    files = sorted(p for name in ("logs", "outputs") for p in (root / name).rglob("*")
                   if p.is_file() and not p.is_symlink()
                   and (p.name.startswith("events.out.tfevents.") or p.suffix in (".log", ".jsonl")))
    records = []
    for index, source in enumerate(files, 1):
        records.append(pack_file(source, root, archive))
        if index % 20 == 0:
            print(f"Packed {index}/{len(files)} logs", flush=True)
    manifest = dict(format="split-gzip-v1", part_bytes=PART_BYTES, files=records)
    (archive / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Archived {len(records)} files: "
          f"{sum(x['bytes'] for x in records):,} -> "
          f"{sum(x['compressed_bytes'] for x in records):,} bytes")


def unpack(root, archive, verify_only):
    manifest = json.loads((archive / "manifest.json").read_text())
    if manifest["format"] != "split-gzip-v1":
        raise ValueError("Unsupported archive format")
    for record in manifest["files"]:
        target = safe_path(root, record["path"])
        if not verify_only and target.exists():
            if target.stat().st_size != record["bytes"] or sha256_file(target) != record["sha256"]:
                raise FileExistsError(f"Existing file differs; refusing overwrite: {target}")
            continue
        temporary = target.with_name(target.name + ".restoring")
        output = None
        digest = hashlib.sha256()
        size = 0
        decompressor = zlib.decompressobj(wbits=31)
        try:
            if not verify_only:
                target.parent.mkdir(parents=True, exist_ok=True)
                output = temporary.open("xb")
            for part in record["parts"]:
                with safe_path(archive, part).open("rb") as stream:
                    while data := stream.read(CHUNK):
                        raw = decompressor.decompress(data)
                        digest.update(raw)
                        size += len(raw)
                        if output is not None:
                            output.write(raw)
            if not decompressor.eof or decompressor.unused_data:
                raise ValueError(f"Invalid gzip stream: {record['path']}")
            if size != record["bytes"] or digest.hexdigest() != record["sha256"]:
                raise ValueError(f"Checksum mismatch: {record['path']}")
            if output is not None:
                output.close()
                if target.exists():
                    raise FileExistsError(target)
                temporary.rename(target)
        finally:
            if output is not None and not output.closed:
                output.close()
    print(f"{'Verified' if verify_only else 'Restored/checked'} {len(manifest['files'])} logs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pack", "verify", "restore"))
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    root, archive = args.root.resolve(), args.archive_dir.resolve()
    if args.action == "pack":
        pack(root, archive)
    else:
        unpack(root, archive, verify_only=args.action == "verify")


if __name__ == "__main__":
    main()
