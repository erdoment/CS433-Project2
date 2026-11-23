"""Resample PVQD wav files to 16 kHz and extract openSMILE eGeMAPS LLDs."""

import argparse
import re
from pathlib import Path

import numpy as np
import opensmile
import pandas as pd
import soundfile as sf
import soxr
from tqdm import tqdm

TARGET_SR = 16000
FILENAME_RE = re.compile(r"^([A-Za-z]+\d+)_E_?NSS_([A-Za-z]+)$", re.IGNORECASE)


def parse_filename(path):
    m = FILENAME_RE.match(path.stem)
    if not m:
        raise ValueError(f"Unexpected filename format: {path.name}")
    speaker = m.group(1).upper()
    sentence = m.group(2).lower()
    return speaker, sentence, f"{speaker}_{sentence}"


def resample_audio(wav_path, out_path):
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = soxr.resample(audio, sr, TARGET_SR)
    sf.write(out_path, audio, TARGET_SR, subtype="PCM_16")
    return sr, len(audio)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, default=Path("6sentences"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/pvqd_16k"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    wav_files = sorted(args.in_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No wav files found in {args.in_dir}")
    if args.limit is not None:
        wav_files = wav_files[: args.limit]

    audio_dir = args.out_dir / "audio"
    lld_dir = args.out_dir / "lld"
    audio_dir.mkdir(parents=True, exist_ok=True)
    lld_dir.mkdir(parents=True, exist_ok=True)

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    seen = set()
    rows = []
    feature_names = None

    for wav_path in tqdm(wav_files):
        speaker, sentence, name = parse_filename(wav_path)
        if name in seen:
            raise ValueError(f"Duplicate output name: {name} ({wav_path.name})")
        seen.add(name)

        audio_path = audio_dir / f"{name}.wav"
        orig_sr, n_samples = resample_audio(wav_path, audio_path)

        # run openSMILE on the saved 16k file so targets match what gets loaded later
        lld_df = smile.process_file(str(audio_path))
        if feature_names is None:
            feature_names = list(lld_df.columns)

        targets = lld_df.to_numpy(dtype=np.float32)
        assert targets.shape[1] == 25 and np.isfinite(targets).all(), f"Bad targets for {name}"

        lld_path = lld_dir / f"{name}.npy"
        np.save(lld_path, targets)

        duration = n_samples / TARGET_SR
        if not (0 <= duration * 100 - len(targets) <= 10):
            print(f"Warning: {name} has {len(targets)} frames for {duration:.2f}s")

        rows.append({
            "name": name,
            "speaker": speaker,
            "sentence": sentence,
            "original_file": wav_path.name,
            "original_sr": orig_sr,
            "duration_s": round(duration, 4),
            "n_samples": n_samples,
            "n_frames": len(targets),
            "audio_path": audio_path.as_posix(),
            "lld_path": lld_path.as_posix(),
        })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.out_dir / "manifest.csv", index=False)
    (args.out_dir / "feature_names.txt").write_text("\n".join(feature_names) + "\n")

    print(
        f"Done: {len(manifest)} files, {manifest['speaker'].nunique()} speakers, "
        f"{manifest['duration_s'].sum() / 60:.1f} min total"
    )


if __name__ == "__main__":
    main()
