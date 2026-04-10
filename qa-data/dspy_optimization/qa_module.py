"""DSPy module and Pydantic schemas for cross-modal QA generation.

Defines the signature, output schema, and module that GEPA optimizes.
"""
import dspy
import pydantic
from typing import Optional


# --- Pydantic Output Schema ---

class QuestionGrounding(pydantic.BaseModel):
    """Evidence citations grounding a question-answer pair."""
    speech_excerpt: Optional[str] = pydantic.Field(
        default=None,
        description="Exact quote from the speech transcript that supports the answer"
    )
    visual_reference: Optional[str] = pydantic.Field(
        default=None,
        description="Relevant part of the visual description"
    )
    ocr_reference: Optional[str] = pydantic.Field(
        default=None,
        description="Relevant on-screen text/graphics"
    )
    reasoning: str = pydantic.Field(
        description="Why multiple modalities are needed to answer this question"
    )


class QuestionOutput(pydantic.BaseModel):
    """A single cross-modal comprehension question with grounding."""
    question: str = pydantic.Field(
        description="The comprehension question text"
    )
    answer: str = pydantic.Field(
        description="The answer grounded in the segment evidence"
    )
    modalities_required: list[str] = pydantic.Field(
        description="Modalities needed to answer (e.g. speech, visual, ocr, entities)"
    )
    reasoning_type: str = pydantic.Field(
        description="One of: factual, inferential, comparative, causal, cross-modal"
    )
    grounding: QuestionGrounding = pydantic.Field(
        description="Evidence citations from the segment"
    )


class QAOutput(pydantic.BaseModel):
    """Container for generated questions."""
    questions: list[QuestionOutput] = pydantic.Field(
        description="Array of cross-modal comprehension questions with grounding"
    )


# --- DSPy Signature ---

class CrossModalQAGeneration(dspy.Signature):
    """Generate cross-modal comprehension questions from video segment evidence.

    Given multi-modal evidence from a broadcast news video segment (speech
    transcript, visual description, on-screen text, named entities, relations),
    generate questions that REQUIRE combining information from MULTIPLE
    modalities to answer correctly.

    Good questions:
    - Require 2+ modalities (removing either visual/OCR OR speech makes it unanswerable)
    - Have specific, verifiable answers grounded in cited evidence
    - Test understanding, not just recall
    - Sound natural — like a researcher, cataloger, or journalist would ask

    Question cognitive levels (aim for L2-L3):
    - L1 Identify: single-modality fact (avoid)
    - L2 Retrieve: locate info across sources (good)
    - L3 Integrate: combine info from multiple modalities (best)
    - L4 Analyze: reason about cross-modal relationships (good, but harder)

    Information need types to cover:
    - Cataloging: metadata extraction (who directed, when broadcast)
    - Factual: specific facts stated or shown
    - Subject: topics/events covered
    - Interpretive: framing, relationships, context
    """

    segment_evidence: str = dspy.InputField(
        desc="Multi-modal evidence from a video segment: speech transcript, "
             "visual description, on-screen text/OCR, named entities, "
             "relations, speaker identity, and entity knowledge"
    )
    broadcast_context: str = dspy.InputField(
        desc="Video ID and broadcast date for context"
    )
    questions: list[QuestionOutput] = dspy.OutputField(
        desc="3-5 cross-modal comprehension questions with grounding citations"
    )


# --- DSPy Module ---

class QAGeneratorModule(dspy.Module):
    """DSPy module for cross-modal QA generation, using Pydantic for structured output."""

    def __init__(self, use_cot: bool = True):
        super().__init__()
        if use_cot:
            self.generator = dspy.ChainOfThought(CrossModalQAGeneration)
        else:
            self.generator = dspy.Predict(CrossModalQAGeneration)

    def forward(self, segment_evidence: str, broadcast_context: str):
        result = self.generator(
            segment_evidence=segment_evidence,
            broadcast_context=broadcast_context,
        )

        questions_raw = result.questions

        # Normalize: DSPy may return list[QuestionOutput], list[dict], or str
        questions_list = []
        if isinstance(questions_raw, list):
            for item in questions_raw:
                if isinstance(item, QuestionOutput):
                    questions_list.append(item)
                elif isinstance(item, dict):
                    try:
                        questions_list.append(QuestionOutput.model_validate(item))
                    except Exception:
                        pass
        elif isinstance(questions_raw, QAOutput):
            questions_list = questions_raw.questions

        return dspy.Prediction(
            questions=questions_list,
            reasoning=getattr(result, "reasoning", ""),
        )
