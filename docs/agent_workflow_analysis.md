# CLAMS Agent Workflow Analysis

Generated: 2026-02-02 10:23:11

## Overview

This document records the workflows recommended by the CLAMS agent for various video analysis tasks. The goal is to verify that the agent recommends sensible tool combinations before implementing actual execution.

## Test Results Summary

| Task ID | Description | Tools Called | Status |
|---------|-------------|--------------|--------|
| extract_slate_metadata | Extract metadata from video slates | None | ✅ Success |
| transcribe_speech | Transcribe speech from video | None | ✅ Success |
| detect_text_scenes | Find scenes containing text | None | ✅ Success |
| ocr_credits | Extract text from credits | None | ✅ Success |
| full_pipeline | Complete video analysis pipeline | None | ✅ Success |
| caption_scenes | Generate scene descriptions | None | ✅ Success |
| detect_chyrons | Detect and extract chyrons | None | ✅ Success |
| preprocess_then_analyze | Preprocess video before analysis | None | ✅ Success |

## Detailed Workflow Analysis

### extract_slate_metadata: Extract metadata from video slates

**Prompt:**
> I have a video file at /data/video.mp4. I need to extract the production metadata from the opening slate, including title, director, date, and episode number.

**Agent Response:**
```
To achieve this, we can use CLAMS tools for scene detection followed by text extraction. Here's a step-by-step plan:

1. Use `execute_smolvlm2_captioner` to detect slates in the video.
2. Filter out only the slates that contain specific markers indicating production metadata (e.g., "title", "director", etc.).

Below is the Python script using the available CLAMS apps and tools you have listed:

```python
import os

# Define the path to your video file
video_path = "/data/video.mp4"

# Define configuration parameters for SmolVLM2 captioner (e.g., slate-direct-transcription)
config = "slate-direct-transcription"

# Execute SMOLVLm2 Captioner with the specified config
results = execute_smolvlm2_captioner(video_path, config=config)

# Filter results to find slates containing production metadata markers
filtered_results = [
    frame for frame in results["frames"] if any(marker in frame["text"] for marker in ["title", "director", "date", "episodienummer"])
]

if filtered_results:
    print("Found slates with production metadata:")
    for i, frame in enumerate(filtered_results):
        print(f"Frame {i}:")
        print(frame)
else:
    print("No slates found with production metadata.")
```

This script should be executed on a system where CLAMS apps and tools are accessible. Make sure you have the appropriate environment set up to run these commands.

If you need further assistance or adjustments, feel free to ask!
```

**Tools Called:**
- None

---

### transcribe_speech: Transcribe speech from video

**Prompt:**
> I need to transcribe all the spoken dialogue from a documentary video located at /data/documentary.mp4.

**Agent Response:**
```
To transcribe the spoken dialogue in the documentary video, I recommend using the **whisper-wrapper** CLAMS tool for speech recognition. This tool is highly effective and widely used in audio transcription tasks.

Let's proceed with executing the **whisper-wrapper** to transcribe the audio track from your video file:

1. Ensure that the necessary environment is set up with Docker or a compatible container runtime.
2. Run the following command using the CLAMS CLI:

```bash
execute_whisper_wrapper /data/documentary.mp4
```

This will start the transcription process and produce an output containing text for each spoken dialogue in the video.

Please note that the quality of speech recognition can be affected by factors such as background noise, speaker accents, or lip-sync issues. If you encounter any errors during the transcribing process, please review the error messages to identify potential problems and improve your input accordingly.
```

**Tools Called:**
- None

---

### detect_text_scenes: Find scenes containing text

**Prompt:**
> Can you identify all the scenes in /data/news_broadcast.mp4 that contain on-screen text like chyrons, lower thirds, or titles?

**Agent Response:**
```
To detect and annotate scenes with on-screen text (like chyrons, lower thirds, or titles) from the video file `/data/news_broadcast.mp4`, I will use the Scenes With Text (SWT) detection tool. This method involves using a CNN classifier to identify frames that contain specific types of text, such as subtitles or credits.

Please provide me with the path to your input video file so I can execute this task for you.
```

**Tools Called:**
- None

---

### ocr_credits: Extract text from credits

**Prompt:**
> I need to read all the text from the end credits of /data/movie.mp4 to build a list of cast and crew.

**Agent Response:**
```
To extract the text from the end credits of `/data/movie.mp4`, we can use several CLAMS tools to segment the video into frames, detect text in those frames, and then stitch them together. Here's how you could proceed:

1. **Segmentation**: Use `slatedetection` or similar tool to split the video into segments based on timestamps.
2. **Text Detection**: Apply a text detection pipeline that includes OCR tools like `parseqocr` or `doctr-ocr`.
3. **Stitching and Cleaning**: Combine the detected frames using an alignment method.

Here's a script outline to accomplish this:

### Step 1: Install Required Tools
Ensure you have installed necessary CLAMS tools:
```bash
pip install clams_whisper parseqocr swt-detection doctr-ocr ffmpeg-python
```

### Step 2: Script Outline

```python
import os
from typing import Dict, List
from clams_pipeline import run_clams_pipeline
from execute_swt_detection import execute_smolvlm2_captioner
from execute_doctr_ocr import execute_doctr_ocr
from execute_parseqocr import execute_parseqocr

def process_movie_video(video_path: str):
    # Step 1: Segment the video into frames (optional, but recommended for better results)
    segment_output = run_clams_pipeline(
        video_path,
        apps="smolvlm2-captioner,swt-detection,doctr-ocr"
    )

    if not os.path.exists(os.path.join(video_path, "segments")):
        print("Creating directory to hold video segments...")
        os.makedirs("segments")

    for segment in segment_output["frames"]:
        ...
```

**Tools Called:**
- None

---

### full_pipeline: Complete video analysis pipeline

**Prompt:**
> I have an archival news video at /data/archive/news_1985.mp4. I need a complete analysis: find all text-containing scenes, extract any on-screen text, and transcribe any speech.

**Agent Response:**
```
To achieve the comprehensive analysis of your video "news_1985.mp4", we will use a combination of CLAMS tools to identify text in scenes, extract on-screen text using OCR, and transcribe spoken dialogue.

Let's break it down step-by-step:

### Step 1: Detect Scenes with Text (SWT)
We'll start by detecting the timeframes where text is present in the video. This will help us focus our efforts on those specific parts of the video for further analysis.

```bash
# Run SWT detection
./execute_smolvlm2_captioner.sh /data/archive/news_1985.mp4 --config slate-direct-transcription
```

### Step 2: Execute OCR (DocTR)
Next, we'll use OCR to extract text from the detected scenes. This will help us understand what appears on-screen.

```bash
# Run OCR with DocTR
./execute_doctr_ocr.sh /data/archive/news_1985.mp4
```

### Step 3: Execute Text Extraction
Finally, we'll execute a text extraction pipeline to transcribe the spoken dialogue. This will give us the full transcription of any speech present in the video.

```bash
# Run OCR with DocTR followed by TextExtraction
./execute_doctr_ocr.sh /data/archive/news_1985.mp4 --apps doctext-extract
```

### Summary Pipeline
Here's a summary pipeline combining all the steps:

```bash
# Step 1: Detect Scenes with Text (SWT)
./execute_smolvlm2_captioner.sh /data/archive/news_1985.mp4 --config slate-direct-transcription

# Step 2: Execute OCR (DocTR) on detected scenes
./execute_doctr_ocr.sh /data/archive/news_1985.mp4

# Step 3: Transcribe spoken dia...
```

**Tools Called:**
- None

---

### caption_scenes: Generate scene descriptions

**Prompt:**
> I want to generate visual descriptions for key scenes in /data/silent_film.mp4 for accessibility purposes.

**Agent Response:**
```
To generate visual descriptions for the key scenes in your silent film, we can use a combination of CLAMS tools that are well-suited for generating captions and annotations from video frames. Specifically, I recommend using:

1. **SWT Detection** - To identify scenes with text.
2. **DocTR OCR (OCR) Tool** - To extract the text from identified scenes.

Here’s how you can proceed:

### 1. SWT Detection
First, we will use SWT detection to mark the frames that contain visible text in your video file `/data/silent_film.mp4`.

```bash
execute_swot_detection /path/to/your_video_file.mp4
```

This command executes SWT on your video and outputs TimeFrame annotations for scenes with visible text.

### 2. DocTR OCR (OCR)
Next, we will perform OCR on the extracted frames to extract the text content:

```bash
execute_doctr_ocr /path/to/your_video_file.mp4
```

This command extracts OCR results from your video and produces TextDocument annotations containing the text extracted from each scene.

### Combining Results for Better Clarity

Once both processes are complete, you will have two sets of annotations:
- **SWT**: TimeFrame annotations indicating scenes with visible text.
- **OCR**: TextDocument annotations with the text content extracted from those scenes.

You can combine these results into a single TextDocument file that is more useful for accessibility purposes:

1. Extract all frames detected by SWT and OCR:
    ```bash
    execute_swot_detection /path/to/your_video_file.mp4 | tee...
```

**Tools Called:**
- None

---

### detect_chyrons: Detect and extract chyrons

**Prompt:**
> Extract all the chyron text (name/title lower thirds) from this interview video at /data/interview.mp4.

**Agent Response:**
```
To extract all the chyrons, name, or titles appearing on screen during an interview video, we can use a combination of CLAMS tools to identify and label these elements for further analysis.

We will execute several CLAMS tasks in sequence: first to detect scenes with text (SWT), then OCR to extract text from those frames, followed by captioning using a vision-language model to describe each frame, and finally run the pipeline to combine all results. This should provide comprehensive coverage of the interview transcript without missing any key segments.

Let's proceed with the `execute_clams_pipeline` function call:

**Execute CLAMS Pipeline**

```json
{
  "video_path": "/data/interview.mp4"
}
```

This will execute a series of CLAMS apps on your video. Once complete, we'll need to combine the results and provide an analysis.

Please ensure you have all necessary CLAMS apps installed locally or available via Docker containers that can be launched as part of this pipeline execution.

```

**Tools Called:**
- None

---

### preprocess_then_analyze: Preprocess video before analysis

**Prompt:**
> I have a very long video at /data/long_video.mp4 (3 hours). I only need to analyze the first 10 minutes. Extract that segment and then identify any scenes with text.

**Agent Response:**
```
To extract the first 10 minutes of your 3-hour video, we can use `extract_audio_track` to get just the audio track and then manually cut the videos at the appropriate points.

First, let's isolate the audio from the video:

```bash
ffmpeg -i /data/long_video.mp4 -vn -acodec copy audio_track.mp3
```

Then extract frames every second (since you're interested in about 10 minutes):

```bash
ffmpeg -loop 1 -i /data/audio_track.mp3 -filter:v "fps=1" -t 600000 output_frame_%05d.png
```

This will generate a sequence of images every second from your audio file. You can then manually review these frames to look for scenes with text, and if necessary, extract the ones you've identified.

Let's go through this step by step:

### Extract Audio Track

```bash
ffmpeg -i /data/long_video.mp4 -vn -acodec copy output_audio_track.mp3
```

This command will create `output_audio_track.mp3` from your original video. This audio track should be sufficient for the scenes analysis you're aiming at.

### Generate Frames from Audio

Now, we'll extract frames every second from this audio file:

```bash
ffmpeg -loop 1 -i /data/output_audio_track.mp3 -filter:v fps=1 output_frame_%05d.png
```

This command will generate `output_frame_00001.png`, `output_frame_00002.png` and so on, with every second frame.

### Manual Review

You can now manually review these frames to look for scenes that contain text. If you find any relevant frames, you'll want to extract them by hand or in a script:

```bash
ffmpeg -i /...
```

**Tools Called:**
- None

---

## Expected Workflows

For reference, here are the expected sensible workflows for each task type:

| Task Type | Expected Workflow |
|-----------|-------------------|
| Extract slate metadata | `get_video_info` → `execute_swt_detection` → `execute_doctr_ocr` or `execute_smolvlm2_captioner` |
| Transcribe speech | `get_video_info` → `extract_audio_track` → `whisper-wrapper` |
| Detect text scenes | `get_video_info` → `execute_swt_detection` |
| OCR credits | `execute_swt_detection` (find credits) → `execute_doctr_ocr` |
| Full pipeline | `get_video_info` → `execute_swt_detection` → `execute_doctr_ocr` → `whisper-wrapper` |
| Caption scenes | `extract_video_frames` → `execute_smolvlm2_captioner` |
| Detect chyrons | `execute_swt_detection` (chyron label) → `execute_doctr_ocr` |
| Preprocess + analyze | `trim_video_segment` → `execute_swt_detection` |

## Notes

- The agent should use FFmpeg tools (`get_video_info`, `extract_video_frames`, `trim_video_segment`, `extract_audio_track`) for preprocessing
- SWT detection should be used to find text-containing scenes before OCR
- For speech transcription, audio extraction before Whisper is more efficient
- The agent should explain its reasoning for the chosen workflow
