"""
Smart Examiner — OCR Evaluator

Computes Character Error Rate (CER) and Word Error Rate (WER)
to measure OCR accuracy against ground truth text.
"""

import logging
from dataclasses import dataclass
import logging
from dataclasses import dataclass
import jiwer
from jiwer import cer as compute_cer, wer as compute_wer

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Accuracy metrics for OCR output vs ground truth."""
    cer: float              # Character Error Rate (0.0 = perfect)
    wer: float              # Word Error Rate (0.0 = perfect)
    accuracy: float         # 1 - CER, as a percentage
    added_words: int
    missing_words: int
    substituted_words: int
    extracted_text: str
    ground_truth: str


def evaluate(extracted_text: str, ground_truth: str) -> EvaluationResult:
    """Compare extracted OCR text against ground truth.

    Args:
        extracted_text: Text produced by the OCR pipeline.
        ground_truth: The correct reference text.

    Returns:
        EvaluationResult with CER, WER, accuracy, and word-level confusion stats.
    """
    if not extracted_text.strip() or not ground_truth.strip():
        logger.warning("Empty text in evaluation — returning 0%% accuracy.")
        return EvaluationResult(
            cer=1.0,
            wer=1.0,
            accuracy=0.0,
            added_words=len(extracted_text.split()) if extracted_text else 0,
            missing_words=len(ground_truth.split()) if ground_truth else 0,
            substituted_words=0,
            extracted_text=extracted_text,
            ground_truth=ground_truth,
        )

    cer_score = compute_cer(ground_truth, extracted_text)
    wer_score = compute_wer(ground_truth, extracted_text)
    accuracy = max(0.0, (1.0 - cer_score)) * 100
    
    # Compute detailed word-level statistics using jiwer
    added = 0
    missing = 0
    substituted = 0
    
    try:
        word_out = jiwer.process_words(ground_truth, extracted_text)
        # Handle insertions (added), deletions (missing), substitutions (corrected/hallucinated)
        if hasattr(word_out, 'alignments') and word_out.alignments:
            for chunk in word_out.alignments[0]:
                if chunk.type == 'insert':
                    added += (chunk.hyp_end_idx - chunk.hyp_start_idx)
                elif chunk.type == 'delete':
                    missing += (chunk.ref_end_idx - chunk.ref_start_idx)
                elif chunk.type == 'substitute':
                    # A substitution implies 1 ref word replaced by 1 hyp word, but could be multi
                    substituted += max((chunk.hyp_end_idx - chunk.hyp_start_idx), (chunk.ref_end_idx - chunk.ref_start_idx))
    except Exception as e:
        logger.error(f"Failed to process word alignments: {e}")

    logger.info(
        "Evaluation — CER: %.2f%%, WER: %.2f%%, Accuracy: %.2f%% [Added:%d Missing:%d Substituted:%d]",
        cer_score * 100, wer_score * 100, accuracy, added, missing, substituted
    )

    return EvaluationResult(
        cer=cer_score,
        wer=wer_score,
        accuracy=accuracy,
        added_words=added,
        missing_words=missing,
        substituted_words=substituted,
        extracted_text=extracted_text,
        ground_truth=ground_truth,
    )
