# Testing the Frame Review Feature

This document describes how to test the human-in-the-loop frame review functionality in the CLAMS Agent Prototype.

## Overview

The frame review feature allows users to review video frames extracted by the agent, providing feedback on detected text, entities, and other annotations. This creates a human-in-the-loop workflow for quality assurance and correction.

## Prerequisites

1. **Python environment**: Ensure you have activated the virtual environment:
   ```bash
   cd /Users/kelleylynch/clams/clams-agent-prototype
   source .venv/bin/activate
   ```

2. **FFmpeg**: Required for video processing:
   ```bash
   # Check if installed
   ffmpeg -version

   # Install on macOS
   brew install ffmpeg
   ```

3. **Test video file**: Have a video file available for testing. Example:
   ```
   /Users/kelleylynch/clams/chaptering_practice/cpb-aacip-225-009w0w1j.mp4
   ```

4. **Dependencies**: Install test requirements:
   ```bash
   pip install -r tests/requirements-test.txt
   ```

## Running Automated Tests

### Unit Tests

Run all tests:
```bash
python -m pytest tests/ -v
```

Run specific test files:
```bash
# FFmpeg tools tests
python -m pytest tests/test_ffmpeg_tools.py -v

# Frame review API tests
python -m pytest tests/test_frame_review_api.py -v
```

Run with coverage:
```bash
python -m pytest tests/ --cov=utils --cov-report=html
open htmlcov/index.html  # View coverage report
```

### Test Categories

| Test File | Coverage |
|-----------|----------|
| `test_ffmpeg_tools.py` | FFmpeg frame extraction, sprites, clips |
| `test_frame_review_api.py` | API endpoints, AG-UI events |
| `test_app_directory.py` | CLAMS app directory fetching |

## Manual Testing

### Step 1: Start the Backend Server

```bash
cd /Users/kelleylynch/clams/clams-agent-prototype
python app.py
```

The server starts on `http://localhost:5001`. You should see:
```
Starting CLAMS Agent Prototype server on port 5001
Debug mode: True
Agent available: True
AG-UI server available: True
```

### Step 2: Open the Web UI

Navigate to: `http://localhost:5001/chat`

### Step 3: Test Frame Extraction API

Extract a frame from a video:

```bash
curl -X POST http://localhost:5001/api/video/frame \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/path/to/your/video.mp4",
    "timestamp": 30.0
  }'
```

**Expected response:**
```json
{
  "success": true,
  "frame_id": "6c1a8627195e",
  "frame_path": "data/ffmpeg_output/frames/video_6c1a8627195e.jpg",
  "timestamp": 30.0,
  "timestamp_formatted": "00:00:30",
  "video_path": "/path/to/your/video.mp4"
}
```

### Step 4: View the Extracted Frame

Open in browser:
```
http://localhost:5001/api/video/frame/<frame_id>
```

Example: `http://localhost:5001/api/video/frame/6c1a8627195e`

### Step 5: Test the Frame Review Modal

#### Method A: Browser Console

Open the browser console (F12 → Console) and run:

```javascript
// Trigger a frame review with a real extracted frame
fetch('/api/test/frame-review', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    frame_id: '6c1a8627195e',  // Use your frame_id from step 3
    context: 'OCR detected text in this frame',
    detected_text: 'SAMPLE TEXT HERE',
    detected_entities: [
      { name: 'Example Corp', type: 'ORG', confidence: 0.95 },
      { name: 'John Doe', type: 'PERSON', confidence: 0.88 }
    ]
  })
}).then(r => r.json()).then(console.log);
```

#### Method B: curl Command

```bash
curl -X POST http://localhost:5001/api/test/frame-review \
  -H "Content-Type: application/json" \
  -d '{
    "frame_id": "6c1a8627195e",
    "context": "Test frame review",
    "detected_text": "SAMPLE TEXT",
    "detected_entities": [{"name": "Test", "type": "ORG", "confidence": 0.9}]
  }'
```

### Step 6: Test Full Integration (Chat Session)

1. Enter a task description: "Extract text from video frames"
2. Click "Start AI Assistant"
3. Once chat is active, the SSE connection is established
4. Frame reviews can be triggered during agent workflows

## Testing Individual Components

### FFmpeg Frame Extraction

```bash
# Test single frame extraction
curl -X POST http://localhost:5001/api/video/frame \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/path/to/video.mp4", "timestamp": 10.0}'

# Test with scaling
curl -X POST http://localhost:5001/api/video/frame \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/path/to/video.mp4", "timestamp": 10.0, "width": 640}'
```

### Thumbnail Sprite Generation

```bash
curl -X POST http://localhost:5001/api/video/sprite \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/path/to/video.mp4",
    "interval": 5.0,
    "thumb_width": 160,
    "thumb_height": 90
  }'
```

### Video Clip Extraction

```bash
curl -X POST http://localhost:5001/api/video/clip \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/path/to/video.mp4",
    "start_time": 10.0,
    "end_time": 20.0
  }'
```

## Frame Review Modal UI Testing

### Test Scenarios

| Scenario | Steps | Expected Result |
|----------|-------|-----------------|
| Open modal | Click Test Review button | Modal opens with frame image |
| Approve frame | Click "Approve" | Modal closes, feedback logged |
| Reject frame | Click "Reject" | Modal closes, feedback logged |
| Edit text | Click edit icon, modify text, click "Submit Correction" | Correction saved |
| Add comment | Type in comment field, approve | Comment included in feedback |
| Skip frame | Click "Skip" | Modal closes, skip action logged |
| Image load error | Use invalid frame URL | Error message displayed |

### Testing Edit Mode

1. Open frame review modal
2. Click the pencil icon next to "Detected Text"
3. Modify the text
4. Verify "Submit Correction" button appears
5. Click submit
6. Verify correction is included in feedback

### Testing Entity Display

1. Include multiple entities in the review request:
   ```json
   {
     "detected_entities": [
       { "name": "Mobil Oil", "type": "ORG", "confidence": 0.95 },
       { "name": "John Smith", "type": "PERSON", "confidence": 0.88 },
       { "name": "New York", "type": "LOCATION", "confidence": 0.92 }
     ]
   }
   ```
2. Verify each entity appears as a chip with:
   - Appropriate icon (building for ORG, person for PERSON)
   - Name and confidence percentage

## Troubleshooting

### Modal Not Opening

1. Check browser console for errors
2. Verify the test endpoint returns events:
   ```bash
   curl -X POST http://localhost:5001/api/test/frame-review -H "Content-Type: application/json" -d '{}'
   ```
3. Ensure frontend is rebuilt: `cd visualization && npm run build`

### Frame Not Loading

1. Verify the frame was extracted:
   ```bash
   ls data/ffmpeg_output/frames/
   ```
2. Check if the frame ID matches:
   ```bash
   curl http://localhost:5001/api/video/frame/<frame_id>
   ```
3. Verify FFmpeg is working:
   ```bash
   ffmpeg -version
   ```

### Server Errors

1. Check server logs in terminal
2. Verify all dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```
3. Check file permissions for output directories

## Test Data

### Sample Test Video Paths

```
/Users/kelleylynch/clams/chaptering_practice/cpb-aacip-225-009w0w1j.mp4
/Users/kelleylynch/clams/chaptering_practice/cpb-aacip-507-j678s4kf2w.mp4
```

### Sample Frame Review Request

```json
{
  "frame_id": "test_frame_001",
  "frame_url": "/api/video/frame/test_frame_001",
  "timestamp": 30.0,
  "timestamp_formatted": "00:00:30",
  "video_path": "/path/to/video.mp4",
  "context": "OCR analysis detected text in this frame",
  "detected_text": "MOBIL OIL COMPANY\nPresents",
  "detected_entities": [
    { "name": "Mobil Oil Company", "type": "ORG", "confidence": 0.95 }
  ],
  "review_type": "approval"
}
```

## Continuous Integration

Add to CI pipeline:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install FFmpeg
        run: sudo apt-get install -y ffmpeg
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r tests/requirements-test.txt
      - name: Run tests
        run: python -m pytest tests/ -v --cov=utils
```
