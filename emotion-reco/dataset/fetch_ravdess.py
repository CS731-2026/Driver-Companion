"""Download RAVDESS (video-only) and lay it out for prepare_data.py.

RAVDESS is published on Zenodo under CC BY-NC-SA 4.0:
    https://zenodo.org/records/1188976

Each "Video_Speech_Actor_XX.zip" contains MP4 clips with this filename schema:
    03-01-EE-II-SS-RR-AA.mp4
    └ modality (03=AV)
       └ channel (01=speech)
          └ emotion (01..08)
             └ intensity (01 normal, 02 strong)
                └ statement
                   └ repetition
                      └ actor (01..24)

We download a user-specified actor range, extract frames per clip with OpenCV,
sample the most expressive middle window (the actor's face typically reaches
apex around the spoken statement's midpoint), and write the CK+-style layout:

    out_root/<emotion>/Actor<NN>_<EE-II-SS-RR>/0001.png ...

Notes
-----
* Each actor zip is ~553 MB (~13 GB for all 24 actors). Default is actors 1-3
  (~1.7 GB) so you can iterate quickly; pass `--actors all` for the full set.
  Interrupted downloads resume automatically on the next run.
* RAVDESS emotion 02=calm has no analogue in the paper. By default we drop it.
* We keep the *middle* `--num_frames` frames per clip (skipping silent lead-in
  and trail-out). Override with `--strategy uniform` to take evenly-spaced
  frames over the whole clip instead.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import cv2
from tqdm import tqdm


RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

ZENODO_RECORD = "1188976"
BASE_URL = f"https://zenodo.org/record/{ZENODO_RECORD}/files"


def is_valid_zip(path: Path) -> bool:
    """True only if `path` is a structurally complete zip (central directory present).

    A truncated download keeps the leading 'PK' bytes but loses the end-of-
    central-directory record, so `zipfile.ZipFile` raises BadZipFile — that is
    exactly the corruption we want to catch before treating a file as cached.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path):
            return True
    except zipfile.BadZipFile:
        return False


def download(url: str, dest: Path, max_retries: int = 4) -> None:
    """Download `url` to `dest`, resuming partial transfers and verifying size.

    Fixes the truncated-download trap: the file is only promoted to its final
    name after the byte count matches Content-Length *and* it opens as a valid
    zip. An incomplete previous attempt is resumed via an HTTP Range request
    instead of being re-fetched from scratch.
    """
    if is_valid_zip(dest):
        print(f"  cached: {dest.name}")
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    # Salvage a previous incomplete download (it may have been renamed straight
    # to `dest` by an older buggy run) so we can resume rather than restart.
    if dest.exists() and not tmp.exists():
        dest.rename(tmp)

    for attempt in range(1, max_retries + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                # Server ignored our Range header → start over from byte 0.
                if have and getattr(r, "status", 200) != 206:
                    have = 0
                    tmp.unlink(missing_ok=True)
                remaining = int(r.headers.get("Content-Length", 0))
                total = have + remaining
                with open(tmp, "ab" if have else "wb") as f, tqdm(
                    total=total or None, initial=have, unit="B",
                    unit_scale=True, desc=dest.name, leave=False,
                ) as bar:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        bar.update(len(chunk))

            got = tmp.stat().st_size
            if total and got != total:
                raise IOError(f"truncated: got {got} of {total} bytes")
            tmp.rename(dest)
            if not is_valid_zip(dest):
                dest.rename(tmp)  # keep bytes for the next resume attempt
                raise zipfile.BadZipFile("download is not a complete zip")
            return
        except (urllib.error.URLError, IOError, zipfile.BadZipFile) as e:
            print(f"  attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                raise
            time.sleep(2 * attempt)


def extract_frames(clip: Path, dst_dir: Path, num_frames: int, strategy: str) -> int:
    """Write `num_frames` PNG frames from `clip` into `dst_dir`. Returns count written."""
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < num_frames:
        cap.release()
        return 0

    if strategy == "middle":
        # Take a centered window covering the middle 70% of the clip
        start = int(n * 0.15)
        end = int(n * 0.85)
        idx = [int(round(start + i * (end - start) / (num_frames - 1)))
               for i in range(num_frames)]
    else:  # uniform
        idx = [int(round(i * (n - 1) / (num_frames - 1))) for i in range(num_frames)]

    dst_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for out_i, frame_i in enumerate(idx, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imwrite(str(dst_dir / f"{out_i:04d}.png"), frame)
        written += 1
    cap.release()
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_root", required=True, help="Where to write the CK+-style layout")
    ap.add_argument("--cache", default="cache/ravdess_zips",
                    help="Directory to keep the downloaded zip files")
    ap.add_argument("--actors", default="1-3",
                    help='Actor range, e.g. "1-3", "5,7,9", or "all"')
    ap.add_argument("--num_frames", type=int, default=12,
                    help="Frames extracted per clip (the pipeline samples 5 from these)")
    ap.add_argument("--strategy", choices=["middle", "uniform"], default="middle")
    ap.add_argument("--include_calm", action="store_true",
                    help="Keep the 'calm' class (excluded by default)")
    ap.add_argument("--keep_zips", action="store_true",
                    help="Do not delete downloaded zips after extraction")
    args = ap.parse_args()

    if args.actors.lower() == "all":
        actors = list(range(1, 25))
    elif "-" in args.actors:
        a, b = args.actors.split("-")
        actors = list(range(int(a), int(b) + 1))
    else:
        actors = [int(s) for s in args.actors.split(",")]
    for a in actors:
        if not 1 <= a <= 24:
            print(f"Actor {a} is out of RAVDESS range 1..24", file=sys.stderr)
            return 1

    out_root = Path(args.out_root)
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    total_seqs = 0
    for actor in actors:
        zip_name = f"Video_Speech_Actor_{actor:02d}.zip"
        url = f"{BASE_URL}/{zip_name}"
        local_zip = cache_dir / zip_name
        print(f"\n[actor {actor:02d}] downloading {zip_name}")
        try:
            download(url, local_zip)
        except Exception as e:  # network / 404
            print(f"  failed: {e}", file=sys.stderr)
            continue

        # Extract MP4s straight into a per-actor temp dir, then frame-sample.
        actor_tmp = cache_dir / f"_unpack_{actor:02d}"
        if actor_tmp.exists():
            shutil.rmtree(actor_tmp)
        actor_tmp.mkdir(parents=True)
        try:
            with zipfile.ZipFile(local_zip) as zf:
                zf.extractall(actor_tmp)
        except zipfile.BadZipFile:
            print(f"  corrupt zip, deleting {zip_name} — re-run to retry",
                  file=sys.stderr)
            local_zip.unlink(missing_ok=True)
            shutil.rmtree(actor_tmp, ignore_errors=True)
            continue

        clips = sorted(actor_tmp.rglob("*.mp4"))
        print(f"  unpacked {len(clips)} clips, extracting frames")
        for clip in tqdm(clips, leave=False):
            parts = clip.stem.split("-")
            if len(parts) != 7:
                continue
            _, _, emo_code, intensity, statement, rep, actor_str = parts
            emotion = RAVDESS_EMOTIONS.get(emo_code)
            if emotion is None:
                continue
            if emotion == "calm" and not args.include_calm:
                continue
            seq_name = f"Actor{actor_str}_{emo_code}-{intensity}-{statement}-{rep}"
            dst = out_root / emotion / seq_name
            written = extract_frames(clip, dst, args.num_frames, args.strategy)
            if written == args.num_frames:
                total_seqs += 1
            else:
                # Skip incomplete sequence
                shutil.rmtree(dst, ignore_errors=True)

        shutil.rmtree(actor_tmp)
        if not args.keep_zips:
            local_zip.unlink()

    print(f"\nDone. {total_seqs} sequences written under {out_root}")
    if total_seqs == 0:
        return 1
    print("Next:  python -m dataset.prepare_data --data_root", out_root, "--output cache/ravdess.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
