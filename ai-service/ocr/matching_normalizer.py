"""
Smart Examiner — Canonical Matching Normalizer

Ensures matching results are standardized into a single, predictable format before scoring.
Validates blueprint contract and merges choose_any_n_details to prevent evidence loss.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

def normalize_matching_result(
    raw_assessment: dict[str, Any],
    blueprint_requirements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Normalize and validate raw LLM matching output into canonical evidence records.

    Merges 'requirements' and 'choose_any_n_details'.
    Calculates valid_count locally (distrusts LLM math).
    Detects duplicated concepts across overlapping criteria to prevent double-counting.
    Ensures every blueprint requirement has exactly one canonical record.

    Args:
        raw_assessment: LLM output for one question
        blueprint_requirements: Frozen requirements from the blueprint

    Returns:
        Canonical assessment dict with a fully populated 'requirements' array.
    """
    overall_status = raw_assessment.get("overall_status", "uncertain")

    # Fast-path for non-responses
    if overall_status in ("unanswered", "error"):
        return _build_fallback_assessment(raw_assessment, blueprint_requirements, overall_status)

    raw_reqs = raw_assessment.get("requirements", [])
    raw_choose = raw_assessment.get("choose_any_n_details", {})

    # Convert choose_any_n_details into lookup (handle dict or list format)
    choose_lookup = {}
    if isinstance(raw_choose, list):
        for item in raw_choose:
            rid = item.get("requirement_id")
            if rid:
                choose_lookup[rid] = item
    elif isinstance(raw_choose, dict):
        for rid, item in raw_choose.items():
            if isinstance(item, dict):
                choose_lookup[rid] = item

    # Convert raw requirements into lookup
    req_lookup = {}
    if isinstance(raw_reqs, list):
        for item in raw_reqs:
            rid = item.get("requirement_id")
            if rid:
                req_lookup[rid] = item

    canonical_requirements = []
    seen_concept_identities = set()
    review_required = False

    for bp_req in blueprint_requirements:
        req_id = bp_req["id"]
        req_type = bp_req.get("type", "mandatory")
        alloc_mode = bp_req.get("allocation_mode", "fixed")

        is_choose_any = req_type == "choose_any_n" or alloc_mode == "per_item"

        raw_r = req_lookup.get(req_id, {})
        raw_c = choose_lookup.get(req_id, {})

        # 1. Base Evidence State
        if raw_r:
            evidence_state = raw_r.get("evidence_state", raw_r.get("status", "absent"))
            evidence_text = raw_r.get("evidence")
        elif raw_c:
            # Bug fix: if choose_any_n_details exists but requirements doesn't
            evidence_state = "present" if any(c.get("valid") for c in raw_c.get("candidate_concepts", [])) else "absent"
            evidence_text = "See candidate concepts"
        else:
            evidence_state = "absent"
            evidence_text = None

        canonical_req = {
            "requirement_id": req_id,
            "evidence_state": evidence_state,
            "evidence": evidence_text,
            "reason": raw_r.get("reason", raw_c.get("reason", "")),
            "matching_was_uncertain": raw_r.get("matching_was_uncertain", False),
        }

        # 2. Concept Normalization (for choose-any/per-item)
        candidate_concepts = raw_c.get("candidate_concepts", raw_r.get("candidate_concepts", []))
        if is_choose_any and candidate_concepts:
            valid_count = 0
            clean_concepts = []

            for concept in candidate_concepts:
                if not isinstance(concept, dict):
                    continue

                is_valid = concept.get("valid", False)
                concept_name = str(concept.get("concept", "")).strip()

                if not concept_name:
                    continue

                # Deduplication across requirements
                if is_valid:
                    identity = concept_name.lower()
                    if identity in seen_concept_identities:
                        logger.warning(f"Overlap detected: concept '{concept_name}' appears in multiple requirements.")
                        # Disable double counting by marking it invalid for this secondary requirement
                        is_valid = False
                        review_required = True
                        canonical_req["reason"] = f"Double counting prevented: '{concept_name}' already scored."
                    else:
                        seen_concept_identities.add(identity)
                        valid_count += 1

                clean_concepts.append({
                    "concept": concept_name,
                    "valid": is_valid,
                    "evidence": concept.get("evidence", ""),
                })

            canonical_req["candidate_concepts"] = clean_concepts
            canonical_req["valid_count"] = valid_count
            canonical_req["candidate_count"] = len(clean_concepts)
        elif is_choose_any:
            # Should have concepts but none provided
            canonical_req["candidate_concepts"] = []
            canonical_req["valid_count"] = 0
            canonical_req["candidate_count"] = 0

        canonical_requirements.append(canonical_req)

    return {
        "overall_status": overall_status,
        "confidence": raw_assessment.get("confidence", 0.0),
        "conceptual_errors": raw_assessment.get("conceptual_errors", []),
        "contradictions": raw_assessment.get("contradictions", []),
        "review_required": review_required,
        "requirements": canonical_requirements,
        "normalized": True,
    }

def _build_fallback_assessment(
    raw_assessment: dict[str, Any],
    blueprint_requirements: list[dict[str, Any]],
    overall_status: str,
) -> dict[str, Any]:
    """Build a deterministic canonical assessment for empty/error states."""
    reqs = []
    for r in blueprint_requirements:
        reqs.append({
            "requirement_id": r["id"],
            "evidence_state": "absent",
            "evidence": None,
            "reason": raw_assessment.get("error", {}).get("message", "No answer provided"),
            "matching_was_uncertain": False,
            "candidate_concepts": [],
            "valid_count": 0,
        })
    return {
        "overall_status": overall_status,
        "confidence": 1.0,
        "conceptual_errors": [],
        "contradictions": [],
        "review_required": False,
        "requirements": reqs,
        "normalized": True,
        "error": raw_assessment.get("error"),
    }
