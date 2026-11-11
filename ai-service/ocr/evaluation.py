"""
Smart Examiner — Evaluation / Marking Orchestrator (v3: Deterministic Scoring)

Orchestrates the full evaluation pipeline:
  1. Build/load frozen assessment blueprint (once per teacher paper)
  2. Run matching/evidence analysis (per student)
  3. Run deterministic Python scoring engine (per student)

Key invariants:
  - The LLM NEVER proposes final awarded_marks
  - Python (scoring.py) is the sole authority for awarded marks
  - Blueprint is frozen before any student is evaluated
  - Empty answers are handled deterministically (no LLM)
  - Same blueprint + same evidence = same marks

Does NOT:
  - Re-run OCR or extraction
  - Remap questions
  - Call an evaluation LLM for mark proposals
  - Import google.genai directly (uses LLMProvider abstraction)
"""

import logging
from typing import Any

import config
from ocr.blueprint import build_assessment_blueprint
from ocr.scoring import score_question

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Student Evaluation (uses blueprint + matching evidence + scoring engine)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_student(
    student_result: dict[str, Any],
    teacher_questions_map: dict[str, dict],
    student_questions_map: dict[str, dict],
    blueprint: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate all questions for one student using deterministic scoring.

    Args:
        student_result: One entry from matching_results
        teacher_questions_map: {norm_id -> teacher_question_dict}
        student_questions_map: {norm_id -> student_question_dict}
        blueprint: Frozen assessment blueprint

    Returns:
        {questions, total_marks_available, total_marks_awarded, validation_warnings, review_flags}
    """
    matches = student_result.get("matches", [])
    increment = config.GRADING_INCREMENT
    blueprint_questions = blueprint.get("questions", {})
    evaluated_questions = []
    total_available = 0.0
    all_warnings: list[str] = []
    review_flags: list[dict] = []

    for match_entry in matches:
        mapping = match_entry.get("question_mapping", {})
        assessment = match_entry.get("student_assessment", {})
        norm_id = mapping.get("student_normalized_id", "")
        teacher_norm_id = mapping.get("teacher_normalized_id", "")
        teacher_q = teacher_questions_map.get(teacher_norm_id)

        q_total_marks = float(teacher_q.get("total_marks", 0) or 0) if teacher_q else 0.0
        total_available += q_total_marks

        # Get blueprint question
        bp_q = blueprint_questions.get(teacher_norm_id)

        # --- NO BLUEPRINT: allocation error ---
        if not bp_q or bp_q.get("allocation_status") == "error":
            evaluated_questions.append({
                "student_normalized_question_id": norm_id,
                "teacher_normalized_question_id": teacher_norm_id,
                "evaluation_status": "allocation_error",
                "total_marks": q_total_marks,
                "requirements": [],
                "question_marks": 0,
                "review_required": False,
                "error": {
                    "type": "allocation_error",
                    "message": bp_q.get("error", "No valid blueprint") if bp_q else "Blueprint missing",
                },
            })
            continue

        bp_reqs = bp_q.get("requirements", [])
        allocation_source = bp_q.get("allocation_source", "unknown")
        overall_status = assessment.get("overall_status", "")

        # --- UNANSWERED: deterministic, no LLM, all absent ---
        if overall_status == "unanswered":
            evaluated_questions.append({
                "student_normalized_question_id": norm_id,
                "teacher_normalized_question_id": teacher_norm_id,
                "evaluation_status": "evaluated",
                "total_marks": q_total_marks,
                "requirements": [
                    {
                        "requirement_id": r["id"],
                        "evidence_state": "absent",
                        "max_marks": r["max_marks"],
                        "awarded_marks": 0,
                        "allocation_source": allocation_source,
                        "reason": "Student did not answer this question",
                        "evidence": None,
                        "review_required": False,
                        "matching_was_uncertain": False,
                    }
                    for r in bp_reqs
                ],
                "question_marks": 0,
                "review_required": False,
            })
            continue

        # --- ERROR: propagate upstream errors ---
        if overall_status == "error":
            evaluated_questions.append({
                "student_normalized_question_id": norm_id,
                "teacher_normalized_question_id": teacher_norm_id,
                "evaluation_status": "error",
                "total_marks": q_total_marks,
                "requirements": [],
                "question_marks": 0,
                "review_required": False,
                "error": assessment.get("error", {
                    "type": "upstream_error",
                    "message": "Matching produced an error for this question",
                }),
            })
            continue

        # --- UNMATCHED ---
        if mapping.get("mapping_status") != "matched" or not teacher_q:
            evaluated_questions.append({
                "student_normalized_question_id": norm_id,
                "teacher_normalized_question_id": teacher_norm_id,
                "evaluation_status": "unmatched",
                "total_marks": q_total_marks,
                "requirements": [],
                "question_marks": 0,
                "review_required": False,
            })
            continue

        # --- DETERMINISTIC SCORING ---
        # Evidence is now strictly canonical from matching_normalizer
        matching_reqs = assessment.get("requirements", [])
        requirement_evidence = []

        for req in matching_reqs:
            requirement_evidence.append({
                "requirement_id": req.get("requirement_id"),
                "evidence_state": req.get("evidence_state", "absent"),
                "evidence": req.get("evidence"),
                "reason": req.get("reason", ""),
                "matching_was_uncertain": req.get("matching_was_uncertain", False),
                "candidate_concepts": req.get("candidate_concepts", []),
                "valid_count": req.get("valid_count", 0),
            })

        # Call deterministic scoring engine
        score_result = score_question(requirement_evidence, bp_q, increment)

        # Annotate each requirement with allocation source
        for sr in score_result["requirements"]:
            sr["allocation_source"] = allocation_source

        # Track review flags
        if score_result["review_required"]:
            review_flags.append({
                "student_normalized_question_id": norm_id,
                "reason": "One or more requirements have uncertain evidence",
            })

        if score_result["warnings"]:
            all_warnings.extend([f"Q{norm_id}: {w}" for w in score_result["warnings"]])

        evaluated_questions.append({
            "student_normalized_question_id": norm_id,
            "teacher_normalized_question_id": teacher_norm_id,
            "evaluation_status": "evaluated",
            "total_marks": q_total_marks,
            "requirements": score_result["requirements"],
            "question_marks": score_result["question_marks"],
            "review_required": score_result["review_required"],
        })

    total_awarded = sum(
        q.get("question_marks", 0) for q in evaluated_questions
        if q.get("evaluation_status") == "evaluated"
    )

    return {
        "questions": evaluated_questions,
        "total_marks_available": total_available,
        "total_marks_awarded": total_awarded,
        "validation_warnings": all_warnings,
        "review_flags": review_flags,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Top-level entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    matching_results: list[dict[str, Any]],
    teacher_data: dict[str, Any],
    students_data: list[dict[str, Any]],
    blueprint: dict[str, Any],
) -> dict[str, Any]:
    """Run the full evaluation pipeline using deterministic scoring.

    Blueprint has already been built. This function:
      1. Iterates students
      2. For each student, uses matching evidence + blueprint + scoring engine
      3. Returns deterministic final marks

    Args:
        matching_results: From run_matching()
        teacher_data: Teacher data from OCR
        students_data: Student data from OCR
        blueprint: Frozen assessment blueprint from blueprint.py
    """
    # Build teacher questions lookup
    teacher_questions_map: dict[str, dict] = {}
    for tq in teacher_data.get("questions", []):
        norm_id = tq.get("normalized_question_id", "")
        if norm_id:
            teacher_questions_map[norm_id] = tq

    # Build student questions lookup
    students_questions_by_index: dict[int, dict[str, dict]] = {}
    for student in students_data:
        s_idx = student.get("student_index", 0)
        sq_map: dict[str, dict] = {}
        for sq in student.get("questions", []):
            nid = sq.get("normalized_question_id", "")
            if nid:
                sq_map[nid] = sq
        students_questions_by_index[s_idx] = sq_map

    # Build mark_allocations output (from blueprint, for API compatibility)
    alloc_output = []
    for norm_id, bp_q in blueprint.get("questions", {}).items():
        alloc_output.append({
            "normalized_question_id": bp_q.get("normalized_question_id", norm_id),
            "total_marks": bp_q.get("total_marks"),
            "allocation_source": bp_q.get("allocation_source"),
            "allocation_status": bp_q.get("allocation_status", "unknown"),
            "requirements": [
                {
                    "requirement_id": r["id"],
                    "max_marks": r["max_marks"],
                    "grading_policy": r.get("grading_policy", {}),
                }
                for r in bp_q.get("requirements", [])
            ],
        })

    # Evaluate each student
    logger.info("Evaluating students with deterministic scoring engine...")
    evaluation_results = []
    all_review_flags: list[dict] = []

    for student_result in matching_results:
        student_index = student_result.get("student_index", 0)
        student_identity = student_result.get("student_identity", {})
        student_questions_map = students_questions_by_index.get(student_index, {})

        logger.info(
            f"  Evaluating student {student_index}: "
            f"{student_identity.get('name', 'Unknown')}"
        )

        result = evaluate_student(
            student_result, teacher_questions_map, student_questions_map,
            blueprint,
        )

        all_review_flags.extend(result.get("review_flags", []))

        evaluation_results.append({
            "student_index": student_index,
            "student_identity": student_identity,
            "questions": result["questions"],
            "total_marks_available": result["total_marks_available"],
            "total_marks_awarded": result["total_marks_awarded"],
            "validation_warnings": result.get("validation_warnings", []),
        })

    return {
        "mark_allocations": alloc_output,
        "evaluation_results": evaluation_results,
        "metadata": {
            "model_name": config.MATCHING_MODEL_NAME,
            "blueprint_model": config.BLUEPRINT_MODEL_NAME,
            "provider": "gemini",
            "blueprint_version": config.BLUEPRINT_VERSION,
            "scoring_engine_version": config.SCORING_ENGINE_VERSION,
            "grading_policy_version": config.GRADING_POLICY_VERSION,
            "grading_increment": config.GRADING_INCREMENT,
            "matching_prompt_version": config.MATCHING_PROMPT_VERSION,
            "teacher_input_hash": blueprint.get("teacher_input_hash"),
            "blueprint_api_calls": blueprint.get("api_calls", 0),
            "evaluation_api_calls": 0,  # No evaluation LLM calls in new architecture
            "api_call_count": blueprint.get("api_calls", 0),
        },
        "blueprint_warnings": blueprint.get("warnings", []),
        "review_flags": all_review_flags,
    }
