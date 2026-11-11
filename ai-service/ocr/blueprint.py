"""
Smart Examiner — Canonical Assessment Blueprint Builder (v2.0)

Creates and caches the FROZEN assessment blueprint per teacher paper.

v2.0 changes:
  - Integrates rubric_parser for embedded rubric extraction
  - Criterion-level requirements (one requirement per teacher criterion)
  - Separate rubric_source and allocation_source fields
  - Never silently falls back when explicit rubric is detected but invalid
  - Per-item rubric support ("Each term (1)")

Responsibilities:
  - Normalize structured/embedded rubric into canonical criteria
  - Generate criterion-level requirements for rubric questions
  - Generate fine-grained requirements for non-rubric questions (LLM)
  - Build/infer mark allocation only when no explicit rubric exists
  - Compute deterministic grading policy per requirement
  - Validate everything, freeze, version
  - Cache by teacher input fingerprint

Key invariants:
  - Blueprint is FROZEN after creation
  - Teacher-provided marks are NEVER replaced with inferred marks
  - Same teacher input hash → same blueprint
  - Criterion IDs and requirement IDs are deterministic

Does NOT:
  - Evaluate students
  - Assign awarded marks
  - Import google.genai directly (uses LLMProvider abstraction)
"""

import hashlib
import json
import logging
import re
import time
from typing import Any

import config
from ocr.provider_factory import get_llm_provider
from ocr.scoring import _snap_to_increment, _is_multiple_of
from ocr.rubric_parser import resolve_rubric

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Blueprint cache: keyed by teacher_input_hash
# ---------------------------------------------------------------------------
_blueprint_cache: dict[str, dict] = {}


def clear_blueprint_cache():
    """Clear the cached blueprints."""
    global _blueprint_cache
    _blueprint_cache = {}
    logger.info("Blueprint cache cleared.")


# ═══════════════════════════════════════════════════════════════════════════════
# Teacher Input Fingerprint
# ═══════════════════════════════════════════════════════════════════════════════

def _build_teacher_input_hash(teacher_data: dict[str, Any]) -> str:
    """Build a deterministic hash of teacher inputs for blueprint caching."""
    questions = teacher_data.get("questions", [])
    sorted_qs = sorted(questions, key=lambda q: q.get("normalized_question_id", ""))

    hash_parts = []
    for q in sorted_qs:
        hash_parts.append(json.dumps({
            "normalized_question_id": q.get("normalized_question_id"),
            "question_text": q.get("question_text"),
            "solution": q.get("solution"),
            "total_marks": q.get("total_marks"),
            "rubric": q.get("rubric"),
            "blueprint_version": config.BLUEPRINT_VERSION,
            "grading_policy_version": config.GRADING_POLICY_VERSION,
        }, sort_keys=True))

    combined = "|".join(hash_parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════════
# Teacher Requirement Generation (LLM) — for non-rubric questions only
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_teacher_requirements(
    teacher_questions: list[dict[str, Any]],
) -> tuple[dict[str, dict], int]:
    """Generate canonical teacher requirements via LLM.

    Only called for questions that do NOT have explicit detailed rubric.

    Returns:
        (requirements_by_norm_id, api_call_count)
    """
    if not teacher_questions:
        return {}, 0

    prompt_path = config.BASE_DIR / "prompts" / "matching_understand_prompt.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_instruction = f.read().strip()

    sorted_questions = sorted(
        teacher_questions,
        key=lambda q: q.get("normalized_question_id", "")
    )

    payload = []
    for tq in sorted_questions:
        norm_id = tq.get("normalized_question_id", "")
        payload.append({
            "question_id": tq.get("question_id", ""),
            "normalized_question_id": norm_id,
            "question_text": tq.get("question_text", ""),
            "solution": tq.get("solution", ""),
            "rubric": tq.get("rubric"),
            "total_marks": tq.get("total_marks"),
            "subquestions": tq.get("subquestions"),
        })

    input_json = json.dumps(payload, indent=2)

    llm = get_llm_provider()
    raw_response, _ = llm.generate_json(
        model_name=config.BLUEPRINT_MODEL_NAME,
        system_instruction=system_instruction,
        prompt_parts=[input_json],
    )

    results = json.loads(raw_response)
    requirements_by_id: dict[str, dict] = {}

    if isinstance(results, list):
        for item in results:
            norm_id = item.get("normalized_question_id", "")
            reqs = item.get("requirements", [])
            reqs.sort(key=lambda r: r.get("id", ""))
            requirements_by_id[norm_id] = {
                "normalized_question_id": norm_id,
                "requirements": reqs,
            }

    return requirements_by_id, 1


# ═══════════════════════════════════════════════════════════════════════════════
# Criterion-Based Requirement Builder (for rubric questions — NO LLM)
# ═══════════════════════════════════════════════════════════════════════════════

def _segment_solution_for_criteria(solution_text: str, criteria: list[dict]) -> list[str]:
    """Segment solution text to match structured criteria that lack explicit content boundaries."""
    if len(criteria) <= 1 or not solution_text.strip():
        return [solution_text.strip()] * len(criteria)

    split_patterns = [
        r'(?:\n|\A)(?=Basic parts and functions:)',
        r'(?:\n|\A)(?=Example(?:\s*\d+)?:)',
        r'(?:\n|\A)(?=Balanced chemical equation:)',
        r'(?:\n|\A)(?=Chemical equation:)',
        r'(?:\n|\A)(?=Equation:)',
        r'(?:\n\s*\n)',
    ]

    for pat in split_patterns:
        parts = [p.strip() for p in re.split(pat, solution_text.strip(), flags=re.IGNORECASE) if p.strip()]
        if len(parts) == len(criteria):
            return parts
        elif len(parts) > len(criteria):
            merged = parts[:len(criteria) - 1]
            merged.append('\n\n'.join(parts[len(criteria) - 1:]))
            return merged

    lines = [l for l in solution_text.strip().split('\n') if l.strip()]
    if len(lines) >= len(criteria):
        per = len(lines) // len(criteria)
        result = []
        for i in range(len(criteria)):
            if i == len(criteria) - 1:
                result.append('\n'.join(lines[i * per:]))
            else:
                result.append('\n'.join(lines[i * per:(i + 1) * per]))
        return result

    return [solution_text.strip()] * len(criteria)


def _build_criterion_requirements(
    criteria: list[dict],
    norm_id: str,
    teacher_question: dict,
) -> list[dict]:
    """Build one requirement per teacher criterion. Pure Python, no LLM.

    Each criterion becomes exactly one requirement with:
      - id: deterministic from criterion
      - description: teacher content under that criterion heading
      - max_marks: from teacher's explicit allocation
      - criterion_id: links back to the criterion

    For per-item criteria, creates a choose_any_n requirement.
    """
    requirements = []
    solution_text = teacher_question.get("solution", "") or ""
    question_text = teacher_question.get("question_text", "") or ""

    # Pre-segment solution if criteria do not have content boundaries
    segmented_contents = None
    if any(not crit.get("content") for crit in criteria):
        segmented_contents = _segment_solution_for_criteria(solution_text, criteria)

    for i, crit in enumerate(criteria):
        criterion_id = crit["criterion_id"]
        label = crit["label"]
        marks = crit["marks"]
        allocation_mode = crit.get("allocation_mode", "fixed")
        content = crit.get("content", "")

        # Build a meaningful requirement description from criterion content
        if content:
            description = content.strip()
        elif allocation_mode == "per_item":
            item_count = crit.get("item_count", 1)
            description = f"{label} ({item_count} items required). Expected content from solution:\n{solution_text.strip()}"
        elif segmented_contents and i < len(segmented_contents):
            seg = segmented_contents[i].strip()
            description = f"{label}: {seg}"
        elif solution_text:
            description = f"{label}: {solution_text.strip()}"
        else:
            description = label

        req_id = f"req_q{norm_id}_{label.lower().replace(' ', '_')}"

        if allocation_mode == "per_item":
            item_count = crit.get("item_count")
            requirements.append({
                "id": req_id,
                "requirement": description,
                "type": "choose_any_n",
                "count": item_count if item_count else 1,
                "criterion_id": criterion_id,
                "max_marks": marks * (item_count if item_count else 1),
                "allocation_mode": "per_item",
                "per_item_marks": marks,
                "item_count": item_count,
            })
        else:
            # Map label and description to an appropriate requirement type
            label_lower = label.lower()
            if "equation" in label_lower or "formula" in label_lower:
                req_type = "mandatory"
            elif "example" in label_lower:
                req_type = "example"
            elif "definition" in label_lower:
                req_type = "definition"
            elif "comparison" in label_lower or "difference" in label_lower:
                req_type = "comparison"
            elif "diagram" in label_lower:
                req_type = "mandatory"
            elif "explanation" in label_lower:
                req_type = "explanation"
            else:
                req_type = "mandatory"

            requirements.append({
                "id": req_id,
                "requirement": description,
                "type": req_type,
                "count": 1,
                "criterion_id": criterion_id,
                "max_marks": marks,
                "allocation_mode": "fixed",
            })

    return requirements


# ═══════════════════════════════════════════════════════════════════════════════
# Allocation Validation & LLM (for non-rubric questions only)
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_allocation(
    allocation: dict,
    total_marks: float,
    increment: float,
) -> tuple[bool, list[str]]:
    """Strictly validate a mark allocation. No silent adjustments."""
    errors = []
    reqs = allocation.get("requirements", [])

    if not reqs:
        errors.append("No requirements in allocation")
        return False, errors

    for req in reqs:
        req_id = req.get("requirement_id", "unknown")
        max_m = req.get("max_marks")

        try:
            max_m = float(max_m)
        except (TypeError, ValueError):
            errors.append(f"{req_id}: max_marks is not numeric ({max_m})")
            continue

        if max_m < 0:
            errors.append(f"{req_id}: max_marks ({max_m}) is negative")
        if not _is_multiple_of(max_m, increment):
            errors.append(f"{req_id}: max_marks ({max_m}) is not a multiple of {increment}")

    try:
        alloc_sum = sum(float(r.get("max_marks", 0)) for r in reqs)
    except (TypeError, ValueError):
        errors.append("Cannot compute allocation sum — non-numeric max_marks")
        return False, errors

    if abs(alloc_sum - total_marks) > 0.01:
        errors.append(f"Allocation sum ({alloc_sum}) != total_marks ({total_marks})")

    return len(errors) == 0, errors


def _call_allocation_llm(input_payload: str, error_context: str = "") -> str:
    """Call the allocation LLM for mark inference."""
    prompt_path = config.BASE_DIR / "prompts" / "allocation_prompt.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_instruction = f.read().strip()

    prompt_parts = []
    if error_context:
        prompt_parts.append(
            f"IMPORTANT — YOUR PREVIOUS RESPONSE WAS REJECTED.\n"
            f"The following validation errors were found:\n\n"
            f"{error_context}\n\n"
            f"GRADING_INCREMENT = {config.GRADING_INCREMENT}\n"
            f"Every max_marks MUST be a multiple of {config.GRADING_INCREMENT}.\n"
            f"Allowed values: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, ...\n"
            f"Forbidden values: 0.25, 0.33, 0.67, 0.75, 0.8, 1.25, 1.33, etc.\n\n"
            f"Fix the allocation below and regenerate ONLY the corrected JSON.\n\n"
        )
    prompt_parts.append(input_payload)

    llm = get_llm_provider()
    text, _ = llm.generate_json(
        model_name=config.BLUEPRINT_MODEL_NAME,
        system_instruction=system_instruction,
        prompt_parts=prompt_parts,
    )
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Grading Policy Computation
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_grading_policy(
    max_marks: float,
    requirement_type: str,
    increment: float,
) -> dict[str, float]:
    """Compute concrete grading policy for one requirement."""
    norm_type = requirement_type.lower().strip() if requirement_type else "default"

    type_mapping = {
        "explanation_required": "explanation",
        "definition_required": "definition",
        "example_required": "example",
        "comparison_required": "comparison",
    }
    norm_type = type_mapping.get(norm_type, norm_type)

    fractions = config.DEFAULT_GRADING_POLICY.get(
        norm_type, config.DEFAULT_GRADING_POLICY["default"]
    )

    policy = {}
    for state, fraction in fractions.items():
        concrete = fraction * max_marks
        concrete = _snap_to_increment(concrete, increment)
        concrete = max(0.0, min(concrete, max_marks))
        policy[state] = concrete

    return policy


# ═══════════════════════════════════════════════════════════════════════════════
# Main Blueprint Builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_assessment_blueprint(
    teacher_data: dict[str, Any],
) -> dict[str, Any]:
    """Build a frozen canonical assessment blueprint for a teacher paper.

    Pipeline:
      1. Check cache
      2. Resolve rubric per question (parser — NO LLM)
      3. For rubric questions: build criterion-level requirements (NO LLM)
      4. For non-rubric questions: generate requirements via LLM
      5. For non-rubric questions: infer allocation via LLM
      6. Compute grading policy, validate, freeze, cache
    """
    global _blueprint_cache

    teacher_input_hash = _build_teacher_input_hash(teacher_data)

    if teacher_input_hash in _blueprint_cache:
        logger.info(f"Blueprint cache hit (hash={teacher_input_hash})")
        return _blueprint_cache[teacher_input_hash]

    logger.info(f"Building new blueprint (hash={teacher_input_hash})")

    teacher_questions = teacher_data.get("questions", [])
    increment = config.GRADING_INCREMENT
    api_calls = 0
    allocation_api_calls = 0
    requirement_api_calls = 0
    warnings: list[str] = []

    teacher_questions_map: dict[str, dict] = {}
    for tq in teacher_questions:
        norm_id = tq.get("normalized_question_id", "")
        if norm_id:
            teacher_questions_map[norm_id] = tq

    # ─── Step 1: Resolve rubric per question (pure Python) ───────────
    rubric_results: dict[str, dict] = {}
    rubric_questions: dict[str, dict] = {}      # Questions with explicit detailed rubric
    non_rubric_questions: dict[str, dict] = {}   # Questions needing LLM

    for norm_id in sorted(teacher_questions_map.keys()):
        teacher_q = teacher_questions_map[norm_id]
        rubric_result = resolve_rubric(teacher_q, increment)
        rubric_results[norm_id] = rubric_result

        if rubric_result["detected"]:
            rubric_questions[norm_id] = teacher_q
            if not rubric_result["valid"]:
                warnings.append(
                    f"Q{norm_id}: Rubric detected ({rubric_result['rubric_source']}) "
                    f"but invalid: {rubric_result['reason']}"
                )
        else:
            non_rubric_questions[norm_id] = teacher_q

    logger.info(
        f"Rubric resolution: {len(rubric_questions)} rubric questions, "
        f"{len(non_rubric_questions)} non-rubric questions"
    )

    # ─── Step 2: Build criterion-level requirements for rubric questions ──
    rubric_requirements: dict[str, list[dict]] = {}

    for norm_id, teacher_q in rubric_questions.items():
        rubric_result = rubric_results[norm_id]
        criteria = rubric_result["criteria"]
        reqs = _build_criterion_requirements(criteria, norm_id, teacher_q)
        rubric_requirements[norm_id] = reqs
        logger.info(
            f"Q{norm_id}: Built {len(reqs)} criterion-level requirements "
            f"(rubric_source={rubric_result['rubric_source']})"
        )

    # ─── Step 3: Generate requirements for non-rubric questions (LLM) ────
    llm_requirements: dict[str, list[dict]] = {}

    if non_rubric_questions:
        non_rubric_list = [
            non_rubric_questions[nid]
            for nid in sorted(non_rubric_questions.keys())
        ]
        try:
            req_results, req_calls = _generate_teacher_requirements(non_rubric_list)
            requirement_api_calls += req_calls
            api_calls += req_calls
            for nid in non_rubric_questions:
                understanding = req_results.get(nid, {})
                llm_requirements[nid] = understanding.get("requirements", [])
        except Exception as e:
            logger.exception("LLM requirement generation failed")
            for nid in non_rubric_questions:
                llm_requirements[nid] = []
                warnings.append(f"Q{nid}: Requirement generation failed: {e}")

    # ─── Step 4: Infer allocation for non-rubric questions (LLM) ─────────
    allocation_cache: dict[str, dict] = {}

    # First, handle rubric questions (NO LLM — direct from criteria)
    for norm_id in rubric_questions:
        rubric_result = rubric_results[norm_id]
        criteria = rubric_result["criteria"]
        total_marks = float(teacher_questions_map[norm_id].get("total_marks", 0) or 0)
        reqs = rubric_requirements[norm_id]

        # Build allocation directly from criteria
        alloc_reqs = []
        for req in reqs:
            alloc_reqs.append({
                "requirement_id": req["id"],
                "max_marks": req["max_marks"],
            })

        allocation_cache[norm_id] = {
            "normalized_question_id": norm_id,
            "total_marks": total_marks,
            "allocation_source": "rubric_detailed",
            "rubric_source": rubric_result["rubric_source"],
            "rubric_valid": rubric_result["valid"],
            "review_required": not rubric_result["valid"],
            "allocation_status": "valid" if rubric_result["valid"] else "needs_review",
            "requirements": alloc_reqs,
        }

    # Handle non-rubric questions via LLM
    needs_llm: list[dict] = []
    for norm_id in sorted(non_rubric_questions.keys()):
        teacher_q = non_rubric_questions[norm_id]
        total_marks = float(teacher_q.get("total_marks", 0) or 0)
        requirements = llm_requirements.get(norm_id, [])

        if not requirements:
            warnings.append(f"Q{norm_id}: No requirements generated, skipping allocation")
            allocation_cache[norm_id] = {
                "normalized_question_id": norm_id,
                "total_marks": total_marks,
                "allocation_source": "error",
                "rubric_source": rubric_results[norm_id]["rubric_source"],
                "allocation_status": "error",
                "error": "No requirements generated",
                "requirements": [],
            }
            continue

        rubric_source = rubric_results[norm_id]["rubric_source"]
        needs_llm.append({
            "normalized_question_id": norm_id,
            "question_text": teacher_q.get("question_text"),
            "solution": teacher_q.get("solution"),
            "total_marks": total_marks,
            "requirements": requirements,
            "rubric_type": "total_only" if rubric_source == "total_only" else "none",
        })

    if needs_llm:
        MAX_ALLOCATION_ATTEMPTS = 3
        attempt = 0
        remaining = list(needs_llm)
        prev_errors_by_id: dict[str, list[str]] = {}

        while remaining and attempt < MAX_ALLOCATION_ATTEMPTS:
            attempt += 1

            error_context = ""
            if attempt > 1 and prev_errors_by_id:
                error_lines = []
                for nid, errs in prev_errors_by_id.items():
                    for e in errs:
                        error_lines.append(f"  Q{nid}: {e}")
                error_context = "\n".join(error_lines)

            if attempt > 1:
                time.sleep(2)

            input_json = json.dumps(remaining, indent=2)

            try:
                raw_response = _call_allocation_llm(input_json, error_context=error_context)
                allocation_api_calls += 1
                api_calls += 1
                llm_results = json.loads(raw_response)

                if not isinstance(llm_results, list):
                    warnings.append(f"Allocation attempt {attempt}: LLM returned non-array")
                    continue

                llm_by_id = {}
                for item in llm_results:
                    nid = item.get("normalized_question_id")
                    if nid:
                        llm_by_id[nid] = item

                still_remaining = []
                prev_errors_by_id = {}

                for q_input in remaining:
                    nid = q_input["normalized_question_id"]
                    total = q_input["total_marks"]
                    llm_alloc = llm_by_id.get(nid)

                    if not llm_alloc:
                        still_remaining.append(q_input)
                        prev_errors_by_id[nid] = ["LLM did not return allocation"]
                        continue

                    rubric_source = rubric_results[nid]["rubric_source"]
                    alloc = {
                        "normalized_question_id": nid,
                        "total_marks": total,
                        "allocation_source": (
                            "inferred_from_total" if q_input["rubric_type"] == "total_only"
                            else "inferred"
                        ),
                        "rubric_source": rubric_source,
                        "requirements": llm_alloc.get("requirements", []),
                    }

                    is_valid, errs = _validate_allocation(alloc, total, increment)

                    if is_valid:
                        alloc["allocation_status"] = "valid"
                        allocation_cache[nid] = alloc
                        logger.info(f"Q{nid}: {alloc['allocation_source']} allocation valid (attempt {attempt})")
                    else:
                        warnings.extend([f"Q{nid} attempt {attempt}: {e}" for e in errs])
                        still_remaining.append(q_input)
                        prev_errors_by_id[nid] = errs

                remaining = still_remaining

            except Exception as e:
                allocation_api_calls += 1
                api_calls += 1
                warnings.append(f"Allocation LLM call failed (attempt {attempt}): {str(e)}")
                logger.exception(f"Allocation LLM call failed (attempt {attempt})")

        for q_input in remaining:
            nid = q_input["normalized_question_id"]
            allocation_cache[nid] = {
                "normalized_question_id": nid,
                "total_marks": q_input["total_marks"],
                "allocation_source": "error",
                "rubric_source": rubric_results[nid]["rubric_source"],
                "allocation_status": "error",
                "error": f"Failed after {MAX_ALLOCATION_ATTEMPTS} attempts",
                "requirements": [],
            }
            warnings.append(f"Q{nid}: allocation FAILED after {MAX_ALLOCATION_ATTEMPTS} attempts")

    # ─── Step 5: Merge requirements + allocation + grading policy ────────
    blueprint_questions: dict[str, dict] = {}

    for norm_id in sorted(teacher_questions_map.keys()):
        teacher_q = teacher_questions_map[norm_id]
        total_marks = float(teacher_q.get("total_marks", 0) or 0)
        rubric_result = rubric_results.get(norm_id, {})
        allocation = allocation_cache.get(norm_id, {})

        # Determine requirements source
        if norm_id in rubric_requirements:
            canonical_reqs = rubric_requirements[norm_id]
        elif norm_id in llm_requirements:
            canonical_reqs = llm_requirements[norm_id]
        else:
            canonical_reqs = []

        # Build allocation lookup
        alloc_reqs = allocation.get("requirements", [])
        alloc_by_id = {r.get("requirement_id"): r for r in alloc_reqs}

        merged_requirements = []
        for req in canonical_reqs:
            req_id = req.get("id", "")
            req_type = req.get("type", "mandatory")

            # Get max_marks: from allocation if available, else from requirement itself
            alloc_entry = alloc_by_id.get(req_id, {})
            max_marks = float(
                alloc_entry.get("max_marks", req.get("max_marks", 0))
            )

            grading_policy = _compute_grading_policy(max_marks, req_type, increment)

            merged_req = {
                "id": req_id,
                "type": req_type,
                "description": req.get("requirement", req.get("description", "")),
                "count": req.get("count", 1),
                "max_marks": max_marks,
                "grading_policy": grading_policy,
            }

            # Add criterion metadata if present
            if "criterion_id" in req:
                merged_req["criterion_id"] = req["criterion_id"]
            if "allocation_mode" in req:
                merged_req["allocation_mode"] = req["allocation_mode"]
            if "per_item_marks" in req:
                merged_req["per_item_marks"] = req["per_item_marks"]
            if "item_count" in req:
                merged_req["item_count"] = req["item_count"]

            merged_requirements.append(merged_req)

        # Build criteria metadata for the blueprint question
        criteria_output = None
        if rubric_result.get("detected"):
            criteria_output = [
                {
                    "criterion_id": c["criterion_id"],
                    "label": c["label"],
                    "marks": c["marks"],
                    "allocation_mode": c.get("allocation_mode", "fixed"),
                    "source": c.get("source", "unknown"),
                }
                for c in rubric_result.get("criteria", [])
            ]
            # Add per-item metadata
            for c_out in criteria_output:
                source_crit = next(
                    (c for c in rubric_result.get("criteria", [])
                     if c["criterion_id"] == c_out["criterion_id"]),
                    None
                )
                if source_crit and source_crit.get("allocation_mode") == "per_item":
                    c_out["item_count"] = source_crit.get("item_count")
                    c_out["per_item_marks"] = source_crit.get("marks")

        blueprint_questions[norm_id] = {
            "normalized_question_id": norm_id,
            "question_text": teacher_q.get("question_text"),
            "teacher_solution": teacher_q.get("solution"),
            "total_marks": total_marks,
            "rubric_source": rubric_result.get("rubric_source", "none"),
            "allocation_source": allocation.get("allocation_source", "unknown"),
            "allocation_status": allocation.get("allocation_status", "unknown"),
            "rubric_valid": rubric_result.get("valid", True),
            "review_required": allocation.get("review_required", False),
            "criteria": criteria_output,
            "requirements": merged_requirements,
        }

    blueprint = {
        "teacher_input_hash": teacher_input_hash,
        "blueprint_version": config.BLUEPRINT_VERSION,
        "grading_policy_version": config.GRADING_POLICY_VERSION,
        "scoring_engine_version": config.SCORING_ENGINE_VERSION,
        "grading_increment": increment,
        "questions": blueprint_questions,
        "api_calls": api_calls,
        "requirement_api_calls": requirement_api_calls,
        "allocation_api_calls": allocation_api_calls,
        "warnings": warnings,
        "status": "valid",
    }

    _blueprint_cache[teacher_input_hash] = blueprint
    logger.info(
        f"Blueprint created and cached (hash={teacher_input_hash}, "
        f"api_calls={api_calls}, requirement_calls={requirement_api_calls}, "
        f"allocation_calls={allocation_api_calls})"
    )

    return blueprint
