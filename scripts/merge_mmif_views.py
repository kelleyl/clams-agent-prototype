#!/usr/bin/env python3
"""Merge views from multiple MMIF files into one, renumbering view IDs to avoid collisions.

Usage:
    python merge_mmif_views.py BASE.mmif ADD1.mmif ADD2.mmif ... -o OUTPUT.mmif
    python merge_mmif_views.py --batch WORK_DIR  # batch mode for pipeline

When merging, view IDs in added files are renumbered (v_0 -> v_4, etc.)
and all internal references (annotation IDs, origin, targets) are updated.
"""
import json
import re
import sys
from pathlib import Path


def renumber_view(view: dict, old_prefix: str, new_prefix: str) -> dict:
    """Renumber all IDs in a view from old_prefix to new_prefix.

    Updates: view id, annotation ids, origin refs, target refs, document refs.
    """
    raw = json.dumps(view)
    # Replace view-prefixed IDs: "v_0:tf_1" -> "v_5:tf_1", "v_0:td_2" -> "v_5:td_2"
    raw = re.sub(
        rf'"{re.escape(old_prefix)}:',
        f'"{new_prefix}:',
        raw,
    )
    # Replace standalone view ID: "v_0" -> "v_5" (but not "v_0:" which was handled above)
    # Only match exact "id": "v_0" patterns
    raw = raw.replace(f'"id": "{old_prefix}"', f'"id": "{new_prefix}"')
    return json.loads(raw)


def merge_mmifs(base_path: str, additional_paths: list, output_path: str,
                app_filters: list = None):
    """Merge views from additional MMIFs into base, with ID renumbering.

    Tracks all view ID remappings across files so that later files'
    cross-references to earlier files' views are updated correctly
    (e.g., Qwen TextDoc origin → TransNet TimeFrame after renumber).

    Args:
        app_filters: If provided, only add views whose app URI contains one of
            these substrings. One filter per additional_path (parallel lists).
            Use None for an entry to add all views from that file.
    """
    base = json.load(open(base_path))

    # Find the next available view number
    existing_ids = {v["id"] for v in base.get("views", [])}
    max_num = -1
    for vid in existing_ids:
        m = re.match(r"v_(\d+)", vid)
        if m:
            max_num = max(max_num, int(m.group(1)))
    next_num = max_num + 1

    # Accumulated ID remapping across all additional files
    # Maps old_view_id → new_view_id (e.g., "v_0" → "v_4")
    global_remap: dict = {}

    for idx, add_path in enumerate(additional_paths):
        if not Path(add_path).exists():
            continue
        add = json.load(open(add_path))
        app_filter = app_filters[idx] if app_filters and idx < len(app_filters) else None

        # First pass: assign new IDs only to views we'll actually add
        file_remap: dict = {}
        for view in add.get("views", []):
            old_id = view.get("id", "")
            if not old_id or not view.get("annotations"):
                continue
            app = view.get("metadata", {}).get("app", "").lower()
            passes_filter = not app_filter or app_filter.lower() in app
            if passes_filter:
                new_id = f"v_{next_num}"
                file_remap[old_id] = new_id
                next_num += 1

        # For views NOT being added: if they match a view already remapped
        # from a previous file, DON'T override the global remap.
        # Only add new remaps for views we're actually including.
        global_remap.update(file_remap)

        # Second pass: add filtered views with all remappings applied
        for view in add.get("views", []):
            old_id = view.get("id", "")
            if not old_id or not view.get("annotations"):
                continue
            app = view.get("metadata", {}).get("app", "").lower()
            if app_filter and app_filter.lower() not in app:
                continue

            # Apply ALL known remappings (from this and earlier files)
            raw = json.dumps(view)
            for old_vid, new_vid in global_remap.items():
                raw = raw.replace(f'"{old_vid}:', f'"{new_vid}:')
                raw = raw.replace(f'"id": "{old_vid}"', f'"id": "{new_vid}"')
            base["views"].append(json.loads(raw))

    with open(output_path, "w") as f:
        json.dump(base, f)

    return len(base["views"])


def batch_merge(work_dir: str):
    """Batch merge for the NewsHour pipeline."""
    work = Path(work_dir)
    guids = [
        "cpb-aacip-507-154dn40c26", "cpb-aacip-507-1v5bc3tf81",
        "cpb-aacip-507-4t6f18t178", "cpb-aacip-507-6w96689725",
        "cpb-aacip-507-7659c6sk7z", "cpb-aacip-507-9882j68s35",
        "cpb-aacip-507-cf9j38m509", "cpb-aacip-507-n29p26qt59",
        "cpb-aacip-507-nk3610wp6s", "cpb-aacip-507-pc2t43js98",
    ]

    (work / "final").mkdir(exist_ok=True)

    for i, g in enumerate(guids, 1):
        base = str(work / "step2_whisper" / f"{g}.mmif")
        additional = [
            str(work / "step3_transnet_raw" / f"{g}_transnet.mmif"),
            str(work / "qwen_swt_out" / f"{g}.mmif"),
            str(work / "qwen_shots_out" / f"{g}.mmif"),
        ]
        # Only take transnet views from transnet, qwen views from qwen outputs
        app_filters = [
            "transnet",      # from transnet output: only transnet views
            "captioner",     # from qwen SWT output: only qwen/captioner views
            "captioner",     # from qwen shots output: only qwen/captioner views
        ]
        output = str(work / "final" / f"{g}.mmif")

        n_views = merge_mmifs(base, additional, output, app_filters=app_filters)
        print(f"  [{i}/{len(guids)}] {g}: {n_views} views merged")


def batch_merge_parakeet(work_dir: str):
    """Batch merge using parakeet ASR instead of whisper."""
    work = Path(work_dir)
    guids = [
        "cpb-aacip-507-154dn40c26", "cpb-aacip-507-1v5bc3tf81",
        "cpb-aacip-507-4t6f18t178", "cpb-aacip-507-6w96689725",
        "cpb-aacip-507-7659c6sk7z", "cpb-aacip-507-9882j68s35",
        "cpb-aacip-507-cf9j38m509", "cpb-aacip-507-n29p26qt59",
        "cpb-aacip-507-nk3610wp6s", "cpb-aacip-507-pc2t43js98",
    ]

    (work / "final_parakeet").mkdir(exist_ok=True)

    for i, g in enumerate(guids, 1):
        # Base: parakeet output (has SWT + parakeet views)
        base = str(work / "step2_parakeet" / f"{g}.mmif")
        additional = [
            str(work / "step3_transnet_raw" / f"{g}_transnet.mmif"),
            str(work / "qwen_swt_out" / f"{g}.mmif"),
            str(work / "qwen_shots_out" / f"{g}.mmif"),
        ]
        app_filters = [
            "transnet",
            "captioner",
            "captioner",
        ]
        output = str(work / "final_parakeet" / f"{g}.mmif")

        n_views = merge_mmifs(base, additional, output, app_filters=app_filters)
        print(f"  [{i}/{len(guids)}] {g}: {n_views} views merged")


if __name__ == "__main__":
    if "--batch-parakeet" in sys.argv:
        idx = sys.argv.index("--batch-parakeet")
        batch_merge_parakeet(sys.argv[idx + 1])
    elif "--batch" in sys.argv:
        idx = sys.argv.index("--batch")
        batch_merge(sys.argv[idx + 1])
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("base", help="Base MMIF file")
        parser.add_argument("additional", nargs="+", help="Additional MMIF files to merge")
        parser.add_argument("-o", "--output", required=True, help="Output path")
        args = parser.parse_args()
        n = merge_mmifs(args.base, args.additional, args.output)
        print(f"Merged {n} views into {args.output}")
