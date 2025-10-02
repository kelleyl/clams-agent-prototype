# CLAMS Agent Test Prompts & Expected Workflows

## Test Cases for Pipeline Recommendations

### 1. Text Extraction Tests

#### Test 1A: Basic OCR
**Prompt:** "I want to extract text from video"
**Expected Tools:** 
- Primary: `easyocr-wrapper` OR `tesseractocr-wrapper` OR `doctr-wrapper`
- Should clarify that only one OCR tool is needed
- May mention `swt-detection` for scene text detection first

#### Test 1B: Chyron/News Ticker Text
**Prompt:** "extract text from chyrons" 
**Expected Tools:**
- Primary: `chyron-detection` (specialized for news tickers)
- Alternative: `swt-detection` (general scene text detection) 
- Followed by: `easyocr-wrapper` OR `tesseractocr-wrapper` OR `doctr-wrapper`
- Should explain chyron-detection is specialized for news tickers

#### Test 1C: Scene Text Detection
**Prompt:** "find text overlays in video"
**Expected Tools:**
- Primary: `swt-detection` (scene with text detection)
- Followed by: `easyocr-wrapper` OR `tesseractocr-wrapper` OR `doctr-wrapper`
- Should explain the two-step process

### 2. Audio Processing Tests

#### Test 2A: Basic Speech Transcription
**Prompt:** "transcribe speech from video"
**Expected Tools:**
- Primary: `whisper-wrapper` OR `distil-whisper-wrapper`
- Should explain distil-whisper is faster but whisper may be more accurate
- Should mention only one is needed

#### Test 2B: Audio Segmentation
**Prompt:** "segment speech and music in audio"
**Expected Tools:**
- Primary: `inaspeechsegmenter-wrapper` (speech vs music segmentation)
- Optional: `whisper-wrapper` for transcription after segmentation

### 3. Video Analysis Tests

#### Test 3A: Scene Detection
**Prompt:** "detect scene changes in video"
**Expected Tools:**
- Primary: `pyscenedetect-wrapper` (scene boundary detection)
- Should be the main recommendation

#### Test 3B: Video Structure Analysis
**Prompt:** "analyze video structure and find title cards"
**Expected Tools:**
- `slatedetection` (for title cards/slates)
- `pyscenedetect-wrapper` (for scene boundaries)
- `barsdetection` (for black bars/letterboxing)

#### Test 3C: Content Analysis
**Prompt:** "describe what's happening in video frames"
**Expected Tools:**
- Primary: `llava-captioner` (image descriptions and captioning)
- May suggest `pyscenedetect-wrapper` first to get key frames

### 4. Text Analysis Tests

#### Test 4A: Named Entity Recognition
**Prompt:** "find people and places mentioned in transcripts"
**Expected Tools:**
- `whisper-wrapper` (to get transcripts first)
- `spacy-wrapper` (for named entity recognition)
- `dbpedia-spotlight-wrapper` (for entity linking)

#### Test 4B: Text Processing
**Prompt:** "analyze language and extract keywords from video speech"
**Expected Tools:**
- `whisper-wrapper` (transcription first)
- `spacy-wrapper` (language analysis)

### 5. Complex Workflow Tests

#### Test 5A: News Video Analysis
**Prompt:** "analyze a news video to extract chyrons, transcribe speech, and find named entities"
**Expected Tools (in order):**
1. `chyron-detection` (find news tickers)
2. `easyocr-wrapper` (read chyron text) 
3. `whisper-wrapper` (transcribe speech)
4. `spacy-wrapper` (extract named entities)
5. `dbpedia-spotlight-wrapper` (link entities)

#### Test 5B: General Video Content Analysis  
**Prompt:** "create a complete analysis of video content including scenes, text, and speech"
**Expected Tools:**
1. `pyscenedetect-wrapper` (scene detection)
2. `swt-detection` OR `chyron-detection` (text detection)
3. `easyocr-wrapper` (text reading)
4. `whisper-wrapper` (speech transcription)
5. `llava-captioner` (visual descriptions)

#### Test 5C: Archive Video Processing
**Prompt:** "process old archive footage to extract all available information"
**Expected Tools:**
1. `barsdetection` (check for letterboxing)
2. `slatedetection` (find title cards)
3. `pyscenedetect-wrapper` (scene detection)
4. `swt-detection` (text detection)
5. `easyocr-wrapper` (OCR)
6. `whisper-wrapper` (transcription)

### 6. Edge Cases & Clarification Tests

#### Test 6A: Ambiguous Request
**Prompt:** "help me process a video"
**Expected Response:** 
- Should ask clarifying questions about what type of processing
- Should mention main categories (text extraction, speech transcription, scene analysis, etc.)

#### Test 6B: Impossible Request
**Prompt:** "translate video to Spanish"
**Expected Response:**
- Should explain current tools don't do translation
- May suggest `whisper-wrapper` for transcription as first step
- Should be helpful about what IS possible

#### Test 6C: Tool Alternatives Request
**Prompt:** "what OCR tools are available?"
**Expected Response:**
- Should list: `easyocr-wrapper`, `tesseractocr-wrapper`, `doctr-wrapper`
- Should explain differences if any
- Should clarify only one is needed typically

## Validation Criteria

### Response Quality
1. **Natural Language**: Responses should be conversational, not technical
2. **Clear Recommendations**: Should clearly state which tools to use
3. **Alternatives Explained**: When multiple tools serve similar purposes, should explain
4. **Workflow Logic**: Should suggest tools in logical order
5. **User-Friendly**: Should avoid technical jargon

### Tool Recommendations
1. **Existing Tools Only**: Must only recommend tools that actually exist
2. **Appropriate Tools**: Tools should match the user's stated goal
3. **Complete Workflows**: Should include all necessary steps for complex tasks
4. **Clear Alternatives**: When multiple options exist, should present them clearly

### Clickable Tools
1. **Accurate Detection**: Tool names mentioned in text should become clickable
2. **Working Links**: Clicking should add tools to pipeline
3. **Visual Feedback**: Should be clearly visually distinguishable as clickable

## Running Tests

To run these tests:
1. Navigate to http://129.64.46.224:5000
2. For each test prompt, send the message and evaluate:
   - Does response quality meet criteria?
   - Are recommended tools appropriate?
   - Are tool names clickable?
   - Does clicking add tools to pipeline correctly?
3. Document any failures or unexpected behaviors
4. Test edge cases to ensure robustness

## Expected Conversation Flow Example

**User:** "extract text from chyrons"
**Expected Assistant Response:** 
"For extracting text from news chyrons (ticker text), I'd recommend using chyron-detection to specifically find those news ticker areas, followed by easyocr-wrapper to read the text. Alternatively, you could use swt-detection which finds any text in scenes, though chyron-detection is specialized for news tickers specifically."

**Expected Clickable Tools:** `chyron-detection`, `easyocr-wrapper`, `swt-detection`
**Expected Pipeline:** User can click to add chyron-detection → easyocr-wrapper