# Research: Human-in-the-Loop Video Frame Review Systems

This document logs related work and research relevant to the frame review feature in the CLAMS Agent Prototype.

## Overview

Human-in-the-loop (HITL) systems for video analysis combine automated processing with human verification and correction. This approach is particularly valuable for:

- Quality assurance of OCR/ASR outputs
- Ground truth generation for training data
- Handling edge cases that automated systems miss
- Building user trust in AI-generated results

## Related Systems and Tools

### 1. Label Studio

**URL**: https://labelstud.io/

**Description**: Open-source data labeling platform supporting video annotation with frame-by-frame review capabilities.

**Relevant Features**:
- Video timeline navigation
- Frame extraction and annotation
- Multi-annotator support with consensus building
- Export to various ML formats

**Differences from our approach**:
- Label Studio is a standalone annotation tool
- Our system integrates review into the agent workflow
- We focus on OCR/NER verification rather than general labeling

### 2. CVAT (Computer Vision Annotation Tool)

**URL**: https://github.com/opencv/cvat

**Description**: Intel's open-source video annotation tool for computer vision tasks.

**Relevant Features**:
- Frame interpolation for object tracking
- Semi-automatic annotation with AI assistance
- Review and quality control workflows

**Potential Integration**:
- Could export CLAMS agent results to CVAT format for detailed review
- CVAT annotations could feed back into CLAMS training data

### 3. Prodigy

**URL**: https://prodi.gy/

**Description**: Commercial annotation tool by Explosion (spaCy creators) with active learning integration.

**Relevant Features**:
- Active learning for efficient annotation
- Custom annotation interfaces
- NER and text classification review
- API-first design for integration

**Inspiration for our system**:
- Prodigy's "binary decision" interface (accept/reject) inspired our Approve/Reject flow
- The correction interface for NER entities informed our entity display

### 4. Amazon SageMaker Ground Truth

**URL**: https://aws.amazon.com/sagemaker/groundtruth/

**Description**: AWS service for building training datasets with human labelers.

**Relevant Features**:
- Video classification and object tracking
- Custom labeling workflows
- Quality control with consensus
- Integration with ML pipelines

**Applicable patterns**:
- Confidence-based routing (high confidence = auto-approve, low = human review)
- Batch review interfaces for efficiency

### 5. VIA (VGG Image Annotator)

**URL**: https://www.robots.ox.ac.uk/~vgg/software/via/

**Description**: Lightweight, browser-based annotation tool from Oxford's Visual Geometry Group.

**Relevant Features**:
- Runs entirely in browser (no server needed)
- Video timeline annotation
- Export to JSON/CSV

**Design influence**:
- Inspired our use of a modal dialog for review
- Simple, focused interface design

## Video Processing Libraries

### FFmpeg

**URL**: https://ffmpeg.org/

**Used in our system for**:
- Frame extraction at specific timestamps
- Thumbnail sprite generation
- Video clip extraction

**Key commands used**:
```bash
# Single frame extraction
ffmpeg -ss <timestamp> -i <video> -frames:v 1 -q:v 2 output.jpg

# Thumbnail sprite
ffmpeg -i <video> -vf "fps=1/5,scale=160:90,tile=10x10" -frames:v 1 sprite.jpg

# Clip extraction
ffmpeg -ss <start> -i <video> -t <duration> -c:v libx264 clip.mp4
```

### OpenCV

**URL**: https://opencv.org/

**Potential use**:
- Real-time frame capture from video streams
- Client-side frame extraction via JavaScript (opencv.js)
- Image preprocessing before OCR

### Vidstack

**URL**: https://vidstack.io/

**Description**: Modern video player library with thumbnail preview support.

**Features we could adopt**:
- Timeline thumbnail scrubbing using our generated sprites
- Keyboard navigation for frame-by-frame review
- Accessibility features

## Human-in-the-Loop Protocols

### AG-UI Protocol

**URL**: https://docs.ag-ui.org/ (hypothetical - based on our implementation)

**Our implementation adds**:
- `frame_review_request` - Agent requests human review of a frame
- `frame_review_response` - Human provides feedback
- `frame_review_batch_start/end` - Batch review session markers

**Event flow**:
```
Agent → frame_review_request → Frontend
Frontend → frame_review_response → Agent
Agent → (next frame or batch_end) → Frontend
```

### LangChain Human-in-the-Loop

**URL**: https://python.langchain.com/docs/modules/agents/how_to/human_in_the_loop

**Approach**:
- Tool-based human input requests
- Synchronous blocking for human response

**Our enhancement**:
- Asynchronous event-based communication
- Visual frame review rather than text-only
- Batch processing support

## OCR and Text Detection

### EasyOCR

**URL**: https://github.com/JaidedAI/EasyOCR

**CLAMS wrapper**: `easyocr-wrapper`

**Review considerations**:
- Often needs correction for handwritten text
- Language-specific accuracy varies
- Confidence scores help prioritize review

### Tesseract

**URL**: https://github.com/tesseract-ocr/tesseract

**CLAMS wrapper**: `tesseractocr-wrapper`

**Review considerations**:
- Better for printed text
- Layout analysis can affect accuracy
- Historical documents may need specialized models

### PaddleOCR

**URL**: https://github.com/PaddlePaddle/PaddleOCR

**Potential integration**:
- Strong performance on complex layouts
- Multi-language support
- Table structure recognition

## Named Entity Recognition

### spaCy

**URL**: https://spacy.io/

**CLAMS wrapper**: `spacy-wrapper`

**Review workflow**:
- Display detected entities with confidence
- Allow correction of entity boundaries
- Support entity type reassignment

### DBpedia Spotlight

**URL**: https://www.dbpedia-spotlight.org/

**CLAMS wrapper**: `dbpedia-spotlight-wrapper`

**Review considerations**:
- Entity linking to knowledge base
- Disambiguation verification needed
- Context-dependent accuracy

## Academic Research

### Active Learning for Video Annotation

**Key papers**:

1. "Active Learning for Video Description With Cluster-Regularized Ensemble Ranking" (TPAMI 2017)
   - Efficient selection of frames for annotation
   - Applicable to prioritizing which frames need human review

2. "Video Annotation and Tracking with Active Learning" (NeurIPS Workshop 2019)
   - Combining tracking with active learning
   - Reducing annotation effort through smart sampling

### Human-AI Collaboration

**Relevant work**:

1. "Human-in-the-Loop Machine Learning" (Morgan & Claypool 2021)
   - Survey of HITL paradigms
   - Quality control strategies

2. "Rethinking Human-AI Interaction for Large-Scale Machine Learning" (CHI 2020)
   - Interface design for efficient review
   - Cognitive load considerations

## Implementation Considerations

### Confidence-Based Routing

Frames with high OCR/NER confidence could be auto-approved, while low-confidence results get human review:

```python
def should_request_review(result):
    if result.confidence < 0.7:
        return True
    if result.contains_names and result.confidence < 0.9:
        return True
    return False
```

### Batch Review Efficiency

For reviewing many frames, consider:
- Keyboard shortcuts (A=approve, R=reject, S=skip)
- Quick navigation between frames
- Progress indicators
- Undo functionality

### Feedback Integration

Collected feedback could:
- Update the current session's results
- Generate training data for model improvement
- Build a correction dictionary for common errors

## Future Directions

1. **Active Learning Integration**: Automatically select which frames most need human review based on model uncertainty.

2. **Correction Propagation**: When a user corrects OCR text, apply similar corrections to visually similar frames.

3. **Multi-Annotator Support**: Allow multiple reviewers with consensus building for critical content.

4. **Annotation Export**: Export reviewed frames and corrections in standard formats (COCO, YOLO, etc.) for model retraining.

5. **Real-time Video Review**: Support frame review during live video processing, not just post-hoc.

## References

1. Settles, B. (2012). Active Learning. Morgan & Claypool Publishers.
2. Wang, Y., et al. (2017). Active Learning for Video Description. IEEE TPAMI.
3. Amershi, S., et al. (2014). Power to the People: The Role of Humans in Interactive Machine Learning. AI Magazine.
4. Label Studio Documentation: https://labelstud.io/guide/
5. CVAT Documentation: https://opencv.github.io/cvat/docs/
6. FFmpeg Documentation: https://ffmpeg.org/documentation.html

---

*Last updated: January 2025*
*Author: CLAMS Agent Development Team*
