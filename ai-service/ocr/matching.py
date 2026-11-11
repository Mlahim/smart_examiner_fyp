"""
Smart Examiner — Student ↔ Teacher Matching Module (v2: Blueprint-Aware)

Responsibilities:
  1. Deterministic question mapping via normalized IDs
  2. Student answer evidence analysis against FROZEN blueprint requirements
  3. Return evidence states (present/partial/absent/incorrect/contradicted/uncertain)

Key invariants:
  - Teacher requirements come from the FROZEN blueprint (never regenerated here)
  - Evidence states are classified, but marks are NEVER assigned
  - Choose-any-N returns candidate concepts for Python counting
  - Empty answers are handled deterministically (no LLM call)

Does NOT:
  - Generate teacher requirements (moved to blueprint.py)
  - Assign marks or scores
  - Modify requirement IDs
  - Create new requirements
  - Import google.genai directly (uses LLMProvider abstraction)
"""

import json
import logging
from typing import Any

import config
from ocr.provider_factory import get_llm_provider
from ocr.matching_normalizer import normalize_matching_result
from ocr.matching_cache import get_cached_matching, set_cached_matching

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# STEP 1: Deterministic Question Mapping
# ---------------------------------------------------------------------------

def match_questions(
    student_questions: list[dict[str, Any]],
    teacher_questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map each student question to the corresponding teacher question deterministically."""
    teacher_by_norm_id = {}
    for tq in teacher_questions:
        norm_id = tq.get("normalized_question_id", "")
        if norm_id:
            teacher_by_norm_id[norm_id] = tq

    mappings = []
    for sq in student_questions:
        s_norm_id = sq.get("normalized_question_id", "")
        s_q_id = sq.get("question_id", "")

        if s_norm_id and s_norm_id in teacher_by_norm_id:
            tq = teacher_by_norm_id[s_norm_id]
            mappings.append({
                "student_question_id": s_q_id,
                "student_normalized_id": s_norm_id,
                "teacher_question_id": tq.get("question_id", ""),
                "teacher_normalized_id": tq.get("normalized_question_id", ""),
                "mapping_status": "matched",
                "mapping_confidence": 1.0,
                "student_question": sq,
                "teacher_question": tq,
            })
        else:
            mappings.append({
                "student_question_id": s_q_id,
                "student_normalized_id": s_norm_id,
                "teacher_question_id": None,
                "teacher_normalized_id": None,
                "mapping_status": "uncertain",
                "mapping_confidence": 0.0,
                "student_question": sq,
                "teacher_question": None,
            })
            logger.warning(f"No teacher match for student question '{s_q_id}'")

    return mappings


# ---------------------------------------------------------------------------
# STEP 2: Student Answer Evidence Analysis (uses frozen blueprint)
# ---------------------------------------------------------------------------

def batch_assess_student_answers(
    mappings: list[dict[str, Any]],
    blueprint: dict[str, Any],
) -> dict[str, dict]:
    """Assess student answers against frozen blueprint requirements.

    Args:
        mappings: Question mappings from match_questions()
        blueprint: Frozen assessment blueprint from blueprint.py

    Returns:
        {norm_id -> assessment_dict} with evidence_state per requirement
    """
    prompt_path = config.BASE_DIR / "prompts" / "matching_assess_prompt.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_instruction = f.read().strip()

    blueprint_questions = blueprint.get("questions", {})
    assessments_by_norm_id: dict[str, dict] = {}
    batch_items = []

    # Sort mappings for deterministic ordering
    sorted_mappings = sorted(
        mappings,
        key=lambda m: m.get("student_normalized_id", "")
    )

    for mapping in sorted_mappings:
        if mapping["mapping_status"] != "matched":
            continue

        teacher_q = mapping["teacher_question"]
        student_q = mapping["student_question"]
        norm_id = mapping["student_normalized_id"]
        teacher_norm_id = mapping["teacher_normalized_id"]

        student_answer = student_q.get("answer", "")
        bp_q = blueprint_questions.get(teacher_norm_id, {})
        bp_reqs = bp_q.get("requirements", [])
        bp_hash = blueprint.get("teacher_input_hash", "")

        # --- EMPTY ANSWER: deterministic, no LLM ---
        if not student_answer.strip():
            raw_eval = {"overall_status": "unanswered"}
            assessments_by_norm_id[norm_id] = normalize_matching_result(raw_eval, bp_reqs)
            continue

        # --- CACHE CHECK ---
        cached_result = get_cached_matching(norm_id, student_answer, bp_hash)
        if cached_result:
            assessments_by_norm_id[norm_id] = cached_result
            continue

        # Convert blueprint requirements to prompt-friendly format
        teacher_requirements = [
            {
                "id": r["id"],
                "description": r.get("requirement", ""), # Note: using "requirement" as that's the field created in blueprint.py
                "type": r.get("type", "mandatory"),
                "count": r.get("count", 1),
            }
            for r in bp_reqs
        ]

        batch_items.append({
            "norm_id": norm_id,
            "student_answer": student_answer,
            "bp_hash": bp_hash,
            "bp_reqs": bp_reqs,
            "payload": {
                "normalized_question_id": norm_id,
                "teacher_requirements": teacher_requirements,
                "teacher_solution": teacher_q.get("solution", ""),
                "teacher_question_text": teacher_q.get("question_text"),
                "student_answer": student_answer,
                "student_subquestions": student_q.get("subquestions", []),
            },
        })

    # --- Batch LLM call ---
    if batch_items:
        batch_payload = [item["payload"] for item in batch_items]
        input_json = json.dumps(batch_payload, indent=2)

        try:
            llm = get_llm_provider()
            raw_response, _ = llm.generate_json(
                model_name=config.MATCHING_MODEL_NAME,
                system_instruction=system_instruction,
                prompt_parts=[input_json],
            )

            results = json.loads(raw_response)

            if isinstance(results, list):
                result_by_id = {
                    item.get("normalized_question_id"): item
                    for item in results
                }

                for batch_item in batch_items:
                    norm_id = batch_item["norm_id"]
                    bp_reqs = batch_item["bp_reqs"]
                    raw_result = result_by_id.get(norm_id)

                    if raw_result:
                        normalized = normalize_matching_result(raw_result, bp_reqs)
                    else:
                        normalized = normalize_matching_result({
                            "overall_status": "error",
                            "error": {"message": "LLM did not return assessment for this question"}
                        }, bp_reqs)

                    assessments_by_norm_id[norm_id] = normalized
                    set_cached_matching(
                        norm_id,
                        batch_item["student_answer"],
                        batch_item["bp_hash"],
                        normalized
                    )

        except Exception as e:
            logger.exception("Student assessment LLM call failed")
            for batch_item in batch_items:
                norm_id = batch_item["norm_id"]
                bp_reqs = batch_item["bp_reqs"]
                normalized = normalize_matching_result({
                    "overall_status": "error",
                    "error": {"message": str(e)}
                }, bp_reqs)
                assessments_by_norm_id[norm_id] = normalized

    return assessments_by_norm_id, bool(batch_items)


# ---------------------------------------------------------------------------
# Full Pipeline: Match all students against one teacher solution
# ---------------------------------------------------------------------------

def run_matching(
    students_data: list[dict[str, Any]],
    teacher_data: dict[str, Any],
    blueprint: dict[str, Any],
) -> dict[str, Any]:
    """Run the full matching pipeline using a frozen blueprint.

    Args:
        students_data: List of student dicts from OCR extraction
        teacher_data: Teacher data from OCR extraction
        blueprint: Frozen assessment blueprint from blueprint.py
    """
    teacher_questions = teacher_data.get("questions", [])
    blueprint_questions = blueprint.get("questions", {})

    all_results = []
    api_call_count = 0

    for student in students_data:
        student_index = student.get("student_index", 0)
        student_identity = student.get("identity", {})
        student_questions = student.get("questions", [])

        logger.info(f"Processing student {student_index}: {student_identity.get('name', 'Unknown')}")

        # Step 1: Map questions
        mappings = match_questions(student_questions, teacher_questions)

        # Step 2: Assess student answers against blueprint
        assessments_by_norm_id, cache_miss = batch_assess_student_answers(mappings, blueprint)

        # Count API call only if there was a cache miss (actual LLM call)
        if cache_miss:
            api_call_count += 1

        matches = []
        for mapping in mappings:
            norm_id = mapping["student_normalized_id"]
            teacher_norm_id = mapping.get("teacher_normalized_id")

            match_entry = {
                "question_mapping": {
                    "student_question_id": mapping["student_question_id"],
                    "student_normalized_id": mapping["student_normalized_id"],
                    "teacher_question_id": mapping["teacher_question_id"],
                    "teacher_normalized_id": mapping["teacher_normalized_id"],
                    "mapping_status": mapping["mapping_status"],
                    "mapping_confidence": mapping["mapping_confidence"],
                },
            }

            if mapping["mapping_status"] == "matched" and mapping["teacher_question"]:
                # Get blueprint requirements for this question
                bp_q = blueprint_questions.get(teacher_norm_id, {})

                match_entry["question_understanding"] = {
                    "normalized_question_id": teacher_norm_id,
                    "requirements": bp_q.get("requirements", []),
                }

                # Look up assessment
                assessment = assessments_by_norm_id.get(norm_id)
                if not assessment:
                    assessment = {
                        "overall_status": "error",
                        "confidence": 0.0,
                        "error": {
                            "type": "model_unavailable",
                            "message": "Assessment not found for this question",
                        },
                    }
                match_entry["student_assessment"] = assessment
            else:
                match_entry["question_understanding"] = None
                match_entry["student_assessment"] = {
                    "overall_status": "uncertain",
                    "confidence": 0.0,
                }

            matches.append(match_entry)

        all_results.append({
            "student_index": student_index,
            "student_identity": student_identity,
            "matches": matches,
        })

    return {
        "api_call_count": api_call_count,
        "matching_results": all_results,
    }
