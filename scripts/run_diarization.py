#!/usr/bin/env python3
"""Run NeMo speaker diarization on video files and produce RTTM output.

Uses the ClusteringDiarizer pipeline:
  1. MarbleNet VAD — voice activity detection
  2. TitaNet — speaker embeddings
  3. Clustering — speaker assignment

Designed to run inside the parakeet container which has NeMo installed.

Usage (inside container):
    python3 /scripts/run_diarization.py /input/video.mp4 /output/diarization.rttm

Usage (from host):
    podman run --rm --gpus all \
      -v /path/to/videos:/input \
      -v /path/to/output:/output \
      ghcr.io/clamsproject/app-parakeet-wrapper:v1.0 \
      python3 /scripts/run_diarization.py /input/video.mp4 /output/diarization.rttm
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def extract_audio(video_path: str, audio_path: str):
    """Extract mono 16kHz WAV from video."""
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path, "-y"
    ], capture_output=True, check=True)


def run_diarization(audio_path: str, output_dir: str, max_speakers: int = 10):
    """Run NeMo ClusteringDiarizer on audio file."""
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf

    # Create manifest (NeMo requires JSON manifest)
    manifest_path = os.path.join(output_dir, "manifest.json")
    audio_name = Path(audio_path).stem
    with open(manifest_path, "w") as f:
        entry = {
            "audio_filepath": audio_path,
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": None,
            "rttm_filepath": None,
            "uem_filepath": None,
        }
        f.write(json.dumps(entry) + "\n")

    # Build diarization config
    config = OmegaConf.create({
        "device": "cuda:0",
        "num_workers": 0,
        "sample_rate": 16000,
        "batch_size": 64,
        "verbose": True,
        "diarizer": {
            "manifest_filepath": manifest_path,
            "out_dir": output_dir,
            "oracle_vad": False,
            "collar": 0.25,
            "ignore_overlap": True,
            "vad": {
                "model_path": "vad_multilingual_marblenet",
                "external_vad_manifest": None,
                "parameters": {
                    "window_length_in_sec": 0.63,
                    "shift_length_in_sec": 0.08,
                    "smoothing": False,
                    "onset": 0.5,
                    "offset": 0.3,
                    "pad_onset": 0.2,
                    "pad_offset": 0.2,
                    "min_duration_on": 0.5,
                    "min_duration_off": 0.5,
                },
            },
            "speaker_embeddings": {
                "model_path": "titanet_large",
                "parameters": {
                    "window_length_in_sec": [1.9, 1.2, 0.5],
                    "shift_length_in_sec": [0.95, 0.6, 0.25],
                    "multiscale_weights": [1, 1, 1],
                    "save_embeddings": False,
                },
            },
            "clustering": {
                "parameters": {
                    "oracle_num_speakers": False,
                    "max_num_speakers": max_speakers,
                    "enhanced_count_thres": 80,
                    "max_rp_threshold": 0.25,
                    "sparse_search_volume": 30,
                },
            },
        },
    })

    print(f"Running diarization on {audio_path}...")
    diarizer = nemo_asr.models.ClusteringDiarizer(cfg=config)
    # Fix NeMo 2.4.0 bug: verbose attribute not set
    if not hasattr(diarizer, 'verbose'):
        diarizer.verbose = True
    diarizer.diarize()

    # Find output RTTM
    rttm_path = os.path.join(output_dir, "pred_rttms", f"{audio_name}.rttm")
    if os.path.exists(rttm_path):
        return rttm_path
    # Try alternative location
    for f in Path(output_dir).rglob("*.rttm"):
        return str(f)
    return None


def rttm_to_json(rttm_path: str) -> list:
    """Parse RTTM file to list of speaker segments."""
    segments = []
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                segments.append({
                    "start": float(parts[3]),
                    "duration": float(parts[4]),
                    "end": float(parts[3]) + float(parts[4]),
                    "speaker": parts[7],
                })
    return sorted(segments, key=lambda s: s["start"])


def main():
    if len(sys.argv) < 3:
        print("Usage: run_diarization.py <video_path> <output_json>")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract audio
        audio_path = os.path.join(tmpdir, "audio.wav")
        print(f"Extracting audio from {video_path}...")
        extract_audio(video_path, audio_path)

        # Run diarization
        diar_dir = os.path.join(tmpdir, "diar_output")
        os.makedirs(diar_dir, exist_ok=True)
        rttm_path = run_diarization(audio_path, diar_dir)

        if rttm_path:
            segments = rttm_to_json(rttm_path)
            # Also copy raw RTTM
            rttm_out = output_path.replace(".json", ".rttm")
            if rttm_out != output_path:
                import shutil
                shutil.copy2(rttm_path, rttm_out)
                print(f"RTTM saved to {rttm_out}")
        else:
            segments = []
            print("WARNING: No RTTM output generated")

        # Save JSON
        result = {
            "video": video_path,
            "num_speakers": len(set(s["speaker"] for s in segments)),
            "segments": segments,
        }
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Diarization complete: {len(segments)} segments, "
              f"{result['num_speakers']} speakers -> {output_path}")


if __name__ == "__main__":
    main()
