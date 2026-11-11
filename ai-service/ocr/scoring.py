"""
Smart Examiner — Deterministic Scoring Engine (v1.0)

Pure Python. ZERO LLM calls. Authoritative source for all final marks.

Maps evidence states to awarded marks using the frozen blueprint's grading policy.

Key invariants:
  - Python is the sole authority for awarded_marks.
  - Evidence states come from matching; marks come from this module.
  - All marks respect GRADING_INCREMENT and max_marks.
  - Choose-any-N uses Python-side concept counting (LLM valid_count NOT trusted).
  - Uncertain evidence is flagged for teacher review.
"""

import math
import logging
from typing import Any

import config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Mark helpers (pure Python, deterministic)
# ═══════════════════════════════════════════════════════════════════════════════

def _snap_to_increment(value: float, increment: float) -> float:
    """Round to nearest increment using round-half-up."""
    if increment <= 0:
        return round(value, 2)
    return round(math.floor(value / increment + 0.5) * increment, 2)


def _is_multiple_of(value: float, increment: float) -> bool:
    """Check if value is a valid multiple of increment (within tolerance)."""
    if increment <= 0:
        return True
    remainder = round(value % increment, 4)
    return remainder < 0.001 or abs(remainder - increment) < 0.001


# ═══════════════════════════════════════════════════════════════════════════════
# Requirement-level scoring
# ═══════════════════════════════════════════════════════════════════════════════

def score_requirement(
    evidence_state: str,
    grading_policy: dict[str, float],
    max_marks: float,
    increment: float,
) -> tuple[float, bool]:
    """Score a single standard requirement using deterministic policy lookup.

    Args:
        evidence_state: One of present/partial/absent/incorrect/contradicted/uncertain
        grading_policy: Maps evidence_state → concrete_marks (from blueprint)
        max_marks: Maximum marks for this requirement
        increment: Grading increment (e.g. 0.5)

    Returns:
        (awarded_marks, review_required)
    """
    review_required = False

    # Normalize state
    state = evidence_state.lower().strip() if evidence_state else "absent"

    # Map old states for compatibility
    if state == "satisfied":
        state = "present"
    elif state == "missing":
        state = "absent"

    if state == "uncertain":
        review_required = True
        # Provisional 0 — teacher must review
        awarded = grading_policy.get("uncertain", 0.0)
    elif state in grading_policy:
        awarded = grading_policy[state]
    else:
        # Unknown state — treat as absent, flag for review
        logger.warning(f"Unknown evidence state '{state}', treating as absent")
        awarded = 0.0
        review_required = True

    # Enforce bounds
    awarded = max(0.0, min(awarded, max_marks))
    awarded = _snap_to_increment(awarded, increment)

    # Hard clamp after snap
    if awarded > max_marks:
        awarded = max_marks

    return awarded, review_required


def score_choose_any_n(
    candidate_concepts: list[dict[str, Any]],
    required_count: int,
    max_marks: float,
    increment: float,
    allocation_mode: str = "fixed",
    per_item_marks: float = 0.0,
) -> tuple[float, int, bool]:
    """Score a choose-any-N or per-item requirement by counting distinct valid concepts.

    Python counts concepts itself — LLM's valid_count is NOT trusted.

    Args:
        candidate_concepts: [{concept, valid, evidence}, ...] from matching LLM
        required_count: How many items were required
        max_marks: Maximum marks for this requirement
        increment: Grading increment
        allocation_mode: "fixed" or "per_item"
        per_item_marks: Marks awarded per valid item (if mode is per_item)

    Returns:
        (awarded_marks, valid_count, review_required)
    """
    if not candidate_concepts:
        return 0.0, 0, False

    # Python-side valid counting (deduplicated by concept identity)
    seen_concepts = set()
    valid_count = 0
    weighted_items = 0.0

    for c in candidate_concepts:
        if isinstance(c, dict):
            is_valid = c.get("valid", False)
            quality = str(c.get("quality", c.get("state", ""))).lower().strip()
            concept_id = str(c.get("concept", "")).lower().strip()

            if concept_id and concept_id not in seen_concepts:
                seen_concepts.add(concept_id)
                if is_valid is True or quality == "present":
                    if quality == "partial":
                        weighted_items += 0.5
                    else:
                        weighted_items += 1.0
                    valid_count += 1
                elif is_valid == "partial" or quality == "partial":
                    weighted_items += 0.5
                    valid_count += 1

    if required_count > 0:
        capped_items = min(weighted_items, float(required_count))
    else:
        capped_items = weighted_items

    if allocation_mode == "per_item" and per_item_marks > 0:
        # Per-item scoring: simply multiply valid weighted items by item marks
        raw_marks = capped_items * per_item_marks
    else:
        # Fixed choose-any-N scoring: proportional fraction of total max_marks
        if required_count <= 0:
            return 0.0, valid_count, False
        raw_marks = (capped_items / float(required_count)) * max_marks

    awarded = _snap_to_increment(raw_marks, increment)
    awarded = max(0.0, min(awarded, max_marks))

    return awarded, valid_count, False


# ═══════════════════════════════════════════════════════════════════════════════
# Question-level scoring
# ═══════════════════════════════════════════════════════════════════════════════

def score_question(
    requirement_evidence: list[dict[str, Any]],
    blueprint_question: dict[str, Any],
    increment: float,
) -> dict[str, Any]:
    """Score all requirements for one question using deterministic policy.

    Args:
        requirement_evidence: Evidence results from matching for this question
        blueprint_question: Blueprint question with requirements and grading policies
        increment: Grading increment

    Returns:
        {requirements, question_marks, total_marks, warnings, review_required}
    """
    total_marks = float(blueprint_question.get("total_marks", 0))
    bp_requirements = blueprint_question.get("requirements", [])
    warnings: list[str] = []
    scored_requirements: list[dict] = []
    question_review = False

    # Build evidence lookup by requirement ID
    evidence_by_id: dict[str, dict] = {}
    for ev in requirement_evidence:
        rid = ev.get("requirement_id")
        if rid:
            evidence_by_id[rid] = ev

    for bp_req in bp_requirements:
        req_id = bp_req["id"]
        req_type = bp_req.get("type", "default")
        max_m = float(bp_req.get("max_marks", 0))
        grading_policy = bp_req.get("grading_policy", {})

        # Get evidence for this requirement
        ev = evidence_by_id.get(req_id, {})
        evidence_state = ev.get("evidence_state", "absent")

        # Score based on requirement type
        alloc_mode = bp_req.get("allocation_mode", "fixed")
        if req_type == "choose_any_n" or alloc_mode == "per_item":
            candidates = ev.get("candidate_concepts", [])
            required_count = bp_req.get("count", 1)
            per_item_marks = bp_req.get("per_item_marks", 0.0)
            
            awarded, valid_count, review = score_choose_any_n(
                candidates, required_count, max_m, increment, alloc_mode, per_item_marks
            )
            scored_requirements.append({
                "requirement_id": req_id,
                "criterion_id": bp_req.get("criterion_id"),
                "evidence_state": evidence_state,
                "allocation_mode": alloc_mode,
                "max_marks": max_m,
                "per_item_marks": per_item_marks,
                "awarded_marks": awarded,
                "valid_count": valid_count,
                "required_count": required_count,
                "candidate_count": len(candidates),
                "evidence": ev.get("evidence"),
                "candidate_concepts": candidates,
                "reason": ev.get("reason", ""),
                "review_required": review,
                "matching_was_uncertain": ev.get("matching_was_uncertain", False),
                "source": "canonical_matching",
                "scoring_engine_version": "v3",
            })
        else:
            awarded, review = score_requirement(
                evidence_state, grading_policy, max_m, increment
            )
            scored_requirements.append({
                "requirement_id": req_id,
                "criterion_id": bp_req.get("criterion_id"),
                "evidence_state": evidence_state,
                "allocation_mode": alloc_mode,
                "max_marks": max_m,
                "per_item_marks": 0.0,
                "awarded_marks": awarded,
                "evidence": ev.get("evidence"),
                "reason": ev.get("reason", ""),
                "review_required": review,
                "matching_was_uncertain": ev.get("matching_was_uncertain", False),
                "source": "canonical_matching",
                "scoring_engine_version": "v3",
            })

        if review:
            question_review = True

    # Calculate question total
    question_marks = sum(r["awarded_marks"] for r in scored_requirements)

    # Clamp to total_marks
    if question_marks > total_marks:
        warnings.append(
            f"Sum of awarded ({question_marks}) exceeds total_marks ({total_marks}), clamped"
        )
        question_marks = total_marks

    question_marks = _snap_to_increment(question_marks, increment)

    return {
        "requirements": scored_requirements,
        "question_marks": question_marks,
        "total_marks": total_marks,
        "warnings": warnings,
        "review_required": question_review,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Student-level scoring
# ═══════════════════════════════════════════════════════════════════════════════

def score_student(
    all_evidence: dict[str, list[dict]],
    blueprint: dict[str, Any],
    increment: float | None = None,
) -> dict[str, Any]:
    """Score all questions for one student using deterministic scoring.

    Args:
        all_evidence: {norm_id -> [requirement_evidence, ...]} from matching
        blueprint: Full assessment blueprint
        increment: Grading increment (defaults to config.GRADING_INCREMENT)

    Returns:
        {questions: [...], total_marks_available, total_marks_awarded, warnings, review_required}
    """
    if increment is None:
        increment = config.GRADING_INCREMENT

    bp_questions = blueprint.get("questions", {})
    scored_questions: list[dict] = []
    total_available = 0.0
    total_awarded = 0.0
    all_warnings: list[str] = []
    student_review = False

    for norm_id, bp_q in bp_questions.items():
        total_available += float(bp_q.get("total_marks", 0))
        evidence = all_evidence.get(norm_id, [])

        result = score_question(evidence, bp_q, increment)

        scored_questions.append({
            "normalized_question_id": norm_id,
            **result,
        })

        total_awarded += result["question_marks"]
        all_warnings.extend([f"Q{norm_id}: {w}" for w in result["warnings"]])
        if result["review_required"]:
            student_review = True

    return {
        "questions": scored_questions,
        "total_marks_available": total_available,
        "total_marks_awarded": total_awarded,
        "warnings": all_warnings,
        "review_required": student_review,
    }
