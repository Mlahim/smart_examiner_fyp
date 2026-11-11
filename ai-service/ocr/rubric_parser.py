"""
Smart Examiner — Rubric Normalizer & Embedded Rubric Parser

Pure Python. ZERO LLM calls.

Responsibilities:
  1. Normalize structured rubric entries (dict or string) into canonical format
  2. Extract embedded rubric criteria from teacher solution text
  3. Handle per-item rubrics ("Each term (1)")
  4. Validate rubric criteria against total_marks
  5. Provide structured validation results (never silently fall back)

Key invariants:
  - Teacher-provided marks are NEVER replaced with inferred marks
  - Invalid rubrics are flagged, not silently discarded
  - Criterion IDs are deterministic
  - OCR variations in labels are tolerated
  - False positives (e.g., "plant cells (2 types)") are rejected
"""

import re
import logging
from typing import Any

from ocr.scoring import _is_multiple_of

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Label Canonicalization (OCR-tolerant)
# ═══════════════════════════════════════════════════════════════════════════════

# Common OCR misspellings → canonical label
_LABEL_CORRECTIONS: dict[str, str] = {
    "explantion": "Explanation",
    "explnation": "Explanation",
    "explaination": "Explanation",
    "explination": "Explanation",
    "explanaton": "Explanation",
    "defination": "Definition",
    "defnition": "Definition",
    "definiton": "Definition",
    "defintion": "Definition",
    "equaton": "Equation",
    "eqution": "Equation",
    "eqation": "Equation",
    "exmple": "Example",
    "exampl": "Example",
    "exampel": "Example",
    "comparsion": "Comparison",
    "comparision": "Comparison",
    "comparson": "Comparison",
}

# Known canonical labels for title-casing
_KNOWN_LABELS = {
    "explanation", "example", "definition", "equation",
    "comparison", "diagram", "formula", "calculation",
    "description", "each term", "each point", "each difference",
    "each example", "each part", "each component",
}


def _canonicalize_label(raw_label: str) -> str:
    """Normalize an OCR-noisy label to a canonical form.

    Preserves the readable label. Does NOT alter semantic meaning.
    """
    if not raw_label:
        return raw_label

    stripped = raw_label.strip()
    lower = stripped.lower()

    # Check OCR corrections first
    if lower in _LABEL_CORRECTIONS:
        return _LABEL_CORRECTIONS[lower]

    # Check known labels for proper casing
    if lower in _KNOWN_LABELS:
        return stripped.title()

    # For "each X" patterns, title-case
    if lower.startswith("each "):
        return stripped.title()

    # Return as-is with first letter capitalized
    if stripped:
        return stripped[0].upper() + stripped[1:]
    return stripped


def _make_criterion_id(norm_id: str, label: str) -> str:
    """Create a deterministic criterion ID from question ID and label.

    Example: norm_id="3", label="Explanation" → "q3_explanation"
    """
    # Normalize: lowercase, replace spaces/special chars with underscore
    safe_label = re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')
    return f"q{norm_id}_{safe_label}"


# ═══════════════════════════════════════════════════════════════════════════════
# Structured Rubric Normalization (dict or string entries)
# ═══════════════════════════════════════════════════════════════════════════════

# Pattern to extract marks from various formats:
# "Explanation (2)"
# "Explanation [2]"
# "Explanation: 2"
# "Explanation - 2"
# "Explanation 2 marks"
_STRING_RUBRIC_PATTERN = re.compile(
    r'^(.+?)\s*(?:\(|\[|:\s*|-\s*)?(\d+(?:\.\d+)?)(?:\)|\]|\s*marks?)?\s*$',
    re.IGNORECASE
)

def _normalize_rubric_entry(
    entry: Any,
    norm_id: str,
    source: str = "structured_rubric",
) -> dict | None:
    """Normalize a single rubric entry (dict or string) into canonical format.

    Returns None if the entry cannot be parsed as a valid rubric criterion.
    """
    if isinstance(entry, dict):
        label = entry.get("label", entry.get("text", entry.get("criterion", "")))
        marks = entry.get("marks")
        if marks is not None and label:
            try:
                marks_float = float(marks)
            except (TypeError, ValueError):
                return None
            canonical_label = _canonicalize_label(str(label))
            return {
                "criterion_id": _make_criterion_id(norm_id, canonical_label),
                "label": canonical_label,
                "marks": marks_float,
                "source": source,
                "allocation_mode": "fixed",
            }

    if isinstance(entry, str):
        match = _STRING_RUBRIC_PATTERN.match(entry.strip())
        if match:
            raw_label = match.group(1).strip()
            marks_str = match.group(2)
            try:
                marks_float = float(marks_str)
            except ValueError:
                return None
            canonical_label = _canonicalize_label(raw_label)
            return {
                "criterion_id": _make_criterion_id(norm_id, canonical_label),
                "label": canonical_label,
                "marks": marks_float,
                "source": source,
                "allocation_mode": "fixed",
            }

    return None


def normalize_structured_rubric(
    rubric: list,
    norm_id: str,
    total_marks: float,
    increment: float,
) -> dict:
    """Normalize a structured rubric array (from teacher grouping) into canonical criteria.

    Accepts both dict and string entries. Returns a validation result.

    Returns:
        {
            "detected": bool,
            "valid": bool,
            "reason": str | None,
            "criteria": list[dict],
            "source": "structured_rubric"
        }
    """
    if not rubric or not isinstance(rubric, list):
        return {
            "detected": False,
            "valid": False,
            "reason": "No rubric entries",
            "criteria": [],
            "source": "structured_rubric",
        }

    # Filter out multiplication/summary annotations like "(4 x 1 = 4)", "4 × 1 = 4"
    # These are NOT actual criteria — they describe how per-item marks add up
    _MULT_ANNOTATION_PATTERN = re.compile(
        r'^\s*\(?\s*\d+\s*[xX×*]\s*\d+(?:\.\d+)?\s*(?:=\s*\d+(?:\.\d+)?)?\s*\)?\s*$'
    )

    filtered_rubric = []
    mult_info = None  # Store parsed multiplication info for per-item linking
    for entry in rubric:
        entry_str = str(entry).strip() if not isinstance(entry, dict) else ""
        if entry_str and _MULT_ANNOTATION_PATTERN.match(entry_str):
            # Parse the multiplication to extract item_count and per_item_marks
            nums = re.findall(r'\d+(?:\.\d+)?', entry_str)
            if len(nums) >= 2:
                mult_info = {
                    "item_count": int(float(nums[0])),
                    "per_item_marks": float(nums[1]),
                }
            logger.info(f"Q{norm_id}: Filtered multiplication annotation: '{entry_str}' (mult_info={mult_info})")
            continue
        filtered_rubric.append(entry)

    criteria = []
    for entry in filtered_rubric:
        normalized = _normalize_rubric_entry(entry, norm_id, source="structured_rubric")
        if normalized:
            criteria.append(normalized)

    if not criteria:
        return {
            "detected": False,
            "valid": False,
            "reason": "No parseable rubric entries",
            "criteria": [],
            "source": "structured_rubric",
        }

    # Detect per-item criteria
    _detect_per_item_criteria(criteria, total_marks)

    # If we extracted multiplication info, link it to per-item criteria
    if mult_info:
        for crit in criteria:
            if crit.get("allocation_mode") == "per_item":
                if crit.get("item_count") is None:
                    crit["item_count"] = mult_info["item_count"]
                    logger.info(
                        f"Q{norm_id}: Linked multiplication annotation to per-item criterion "
                        f"'{crit['label']}': item_count={mult_info['item_count']}"
                    )

    # Validate
    errors = _validate_criteria(criteria, total_marks, increment)

    return {
        "detected": True,
        "valid": len(errors) == 0,
        "reason": "; ".join(errors) if errors else None,
        "criteria": criteria,
        "source": "structured_rubric",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Embedded Rubric Extraction (from solution text)
# ═══════════════════════════════════════════════════════════════════════════════

# Pattern: line starting with a label followed by a number
# Must be at start of line or after newline. Must not be mid-sentence.
# Handles: "Explanation (2)", "Example [1.5]", "Each term: 1", "Definition - 2 marks"
_EMBEDDED_RUBRIC_PATTERN = re.compile(
    r'^\s*([A-Za-z][A-Za-z\s]{1,30}?)\s*(?:\(|\[|:\s*|-\s*)?(\d+(?:\.\d+)?)(?:\)|\]|\s*marks?)?\s*$',
    re.MULTILINE | re.IGNORECASE
)

# Anti-false-positive patterns: things that look like rubric but aren't
_FALSE_POSITIVE_PATTERNS = [
    re.compile(r'\d+\s+types?\b', re.IGNORECASE),       # "2 types"
    re.compile(r'\d+\s+parts?\b', re.IGNORECASE),       # "3 parts"
    re.compile(r'\d+\s+examples?\b', re.IGNORECASE),    # "2 examples" inside text
    re.compile(r'\d+\s+cells?\b', re.IGNORECASE),       # "2 cells"
    re.compile(r'\d+\s+steps?\b', re.IGNORECASE),       # "3 steps"
]


def _is_false_positive(label: str, marks_str: str) -> bool:
    """Check if an apparent rubric match is actually a false positive.

    Example: "plant cells (2 types)" → the "(2" is not a rubric mark.
    """
    combined = f"{marks_str} "
    for pattern in _FALSE_POSITIVE_PATTERNS:
        if pattern.search(combined):
            return True

    # Labels that are too generic or suspiciously part of a sentence
    lower_label = label.lower().strip()
    if len(lower_label) < 2:
        return True

    # Check if it's a known rubric-style label or starts with "each"
    is_known_style = (
        lower_label in _KNOWN_LABELS
        or lower_label in _LABEL_CORRECTIONS
        or lower_label.startswith("each ")
        or lower_label in {
            "explanation", "example", "definition", "equation",
            "formula", "diagram", "comparison", "description",
            "calculation", "proof", "derivation",
        }
    )

    # If not a known label, be more conservative — require it to look like a heading
    if not is_known_style:
        # Allow if it's short and title-like (2-3 words max)
        words = lower_label.split()
        if len(words) > 4:
            return True

    return False


def extract_embedded_rubric(
    solution_text: str,
    total_marks: float,
    norm_id: str,
    increment: float,
) -> dict:
    """Extract embedded rubric criteria from teacher solution text.

    Looks for patterns like "Explanation (2)" at the start of lines.

    Returns a validation result (never None — always structured):
        {
            "detected": bool,
            "valid": bool,
            "reason": str | None,
            "criteria": list[dict],
            "source": "embedded_in_solution"
        }
    """
    if not solution_text or not solution_text.strip():
        return {
            "detected": False,
            "valid": False,
            "reason": "Empty solution text",
            "criteria": [],
            "source": "embedded_in_solution",
        }

    matches = list(_EMBEDDED_RUBRIC_PATTERN.finditer(solution_text))

    if not matches:
        return {
            "detected": False,
            "valid": False,
            "reason": "No embedded rubric patterns found",
            "criteria": [],
            "source": "embedded_in_solution",
        }

    # Filter false positives
    valid_matches = []
    for m in matches:
        raw_label = m.group(1).strip()
        marks_str = m.group(2)
        if not _is_false_positive(raw_label, marks_str):
            valid_matches.append(m)

    if not valid_matches:
        return {
            "detected": False,
            "valid": False,
            "reason": "All matches were false positives",
            "criteria": [],
            "source": "embedded_in_solution",
        }

    # Require at least 1 valid match for a single-criterion question,
    # or 2+ matches for multi-criteria questions
    # (A single match is still valid — e.g., "Each term (1)")

    # Build criteria with content boundaries
    criteria = []
    for i, m in enumerate(valid_matches):
        raw_label = m.group(1).strip()
        marks_str = m.group(2)

        try:
            marks_float = float(marks_str)
        except ValueError:
            continue

        canonical_label = _canonicalize_label(raw_label)
        criterion_id = _make_criterion_id(norm_id, canonical_label)

        # Content boundaries: from end of this heading to start of next heading (or end of text)
        content_start = m.end()
        if i + 1 < len(valid_matches):
            content_end = valid_matches[i + 1].start()
        else:
            content_end = len(solution_text)

        content = solution_text[content_start:content_end].strip()

        criteria.append({
            "criterion_id": criterion_id,
            "label": canonical_label,
            "marks": marks_float,
            "source": "embedded_in_solution",
            "allocation_mode": "fixed",
            "content": content,
            "content_start": content_start,
            "content_end": content_end,
        })

    if not criteria:
        return {
            "detected": False,
            "valid": False,
            "reason": "No valid criteria after filtering",
            "criteria": [],
            "source": "embedded_in_solution",
        }

    # Detect per-item criteria
    _detect_per_item_criteria(criteria, total_marks)

    # Validate
    errors = _validate_criteria(criteria, total_marks, increment)

    return {
        "detected": True,
        "valid": len(errors) == 0,
        "reason": "; ".join(errors) if errors else None,
        "criteria": criteria,
        "source": "embedded_in_solution",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Item Detection
# ═══════════════════════════════════════════════════════════════════════════════

_PER_ITEM_LABELS = {
    "each term", "each point", "each difference", "each example",
    "each part", "each component", "each definition", "each item",
}


def _detect_per_item_criteria(criteria: list[dict], total_marks: float):
    """Detect and annotate per-item criteria in-place.

    A per-item criterion is one like "Each term (1)" where the mark
    applies per individual item rather than being the total.
    """
    for crit in criteria:
        label_lower = crit["label"].lower()
        if label_lower in _PER_ITEM_LABELS or label_lower.startswith("each "):
            crit["allocation_mode"] = "per_item"
            per_item_marks = crit["marks"]

            # Try to determine item count from total_marks
            if per_item_marks > 0 and total_marks > 0:
                computed_count = total_marks / per_item_marks
                if computed_count == int(computed_count):
                    crit["item_count"] = int(computed_count)
                else:
                    # Can't determine clean count — flag for review
                    crit["item_count"] = None
                    crit["review_required"] = True


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_criteria(
    criteria: list[dict],
    total_marks: float,
    increment: float,
) -> list[str]:
    """Validate extracted criteria. Returns list of error strings (empty = valid).

    IMPORTANT: Never silently falls back. Returns errors for the caller to handle.
    """
    errors = []

    if not criteria:
        errors.append("No criteria to validate")
        return errors

    # Check for duplicate criterion IDs
    seen_ids = set()
    for crit in criteria:
        cid = crit.get("criterion_id", "")
        if cid in seen_ids:
            errors.append(f"Duplicate criterion ID: {cid}")
        seen_ids.add(cid)

    # Check individual marks
    for crit in criteria:
        marks = crit.get("marks", 0)
        if marks <= 0:
            errors.append(f"{crit.get('criterion_id')}: marks ({marks}) must be positive")
        if not _is_multiple_of(marks, increment):
            errors.append(
                f"{crit.get('criterion_id')}: marks ({marks}) not a multiple of {increment}"
            )

    # Check sum — but handle per-item criteria specially
    has_per_item = any(c.get("allocation_mode") == "per_item" for c in criteria)

    if has_per_item and len(criteria) == 1:
        # Single per-item criterion: validate per_item_marks × item_count == total_marks
        crit = criteria[0]
        per_item_marks = crit["marks"]
        item_count = crit.get("item_count")

        if item_count is not None:
            expected_total = per_item_marks * item_count
            if abs(expected_total - total_marks) > 0.01:
                errors.append(
                    f"{crit['criterion_id']}: per_item_marks ({per_item_marks}) × "
                    f"item_count ({item_count}) = {expected_total} != total_marks ({total_marks})"
                )
        # If item_count is None, we can't validate sum — already flagged review_required
    else:
        # Fixed criteria: sum must equal total_marks
        alloc_sum = sum(c.get("marks", 0) for c in criteria)
        if abs(alloc_sum - total_marks) > 0.01:
            errors.append(
                f"Sum of criteria marks ({alloc_sum}) != total_marks ({total_marks})"
            )

    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# Unified Rubric Resolution (main entry point)
# ═══════════════════════════════════════════════════════════════════════════════

# Patterns to recognize count-based question prompts (e.g. "Write any four differences")
_COUNT_BASED_QUESTION_PATTERN = re.compile(
    r'\b(?:write|give|state|list|name)\s+(?:any\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(differences|points|features|characteristics|factors|reasons|examples|terms|steps|parts)\b',
    re.IGNORECASE
)
_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}


def resolve_rubric(
    teacher_question: dict[str, Any],
    increment: float,
) -> dict:
    """Resolve the rubric for a teacher question using 4-tier precedence.

    Precedence:
      1. Explicit structured detailed rubric (dict or string format)
      2. Embedded detailed rubric in solution text
      3. Total-only rubric (total_marks present but no detailed breakdown)
      4. No rubric

    Returns:
        {
            "rubric_source": str,          # "structured_rubric" | "embedded_in_solution" | "total_only" | "none"
            "detected": bool,              # Was a detailed rubric detected?
            "valid": bool,                 # Is the detected rubric valid?
            "reason": str | None,          # Validation error if invalid
            "criteria": list[dict],        # Normalized criteria
            "requires_allocation_llm": bool,  # Does this need LLM for allocation?
        }
    """
    norm_id = teacher_question.get("normalized_question_id", "?")
    total_marks = float(teacher_question.get("total_marks", 0) or 0)
    rubric = teacher_question.get("rubric") or []
    solution = teacher_question.get("solution", "")
    question_text = teacher_question.get("question_text", "") or ""

    # --- Tier 1: Structured rubric (from teacher_grouping LLM) ---
    if rubric and isinstance(rubric, list):
        result = normalize_structured_rubric(rubric, norm_id, total_marks, increment)
        if result["detected"]:
            # Check if question explicitly requests N differences/items/points/examples
            # while the extracted rubric was merely a generic boilerplate like Explanation/Example
            count_match = _COUNT_BASED_QUESTION_PATTERN.search(question_text) if question_text else None
            if count_match and total_marks > 0:
                cnt_str = count_match.group(1).lower()
                cnt = int(cnt_str) if cnt_str.isdigit() else _WORD_TO_NUM.get(cnt_str)
                item_type = count_match.group(2).lower()
                labels_set = {c["label"].lower() for c in result.get("criteria", [])}
                is_generic_rubric = labels_set.issubset({"explanation", "example", "definition"})
                if cnt and cnt > 1 and is_generic_rubric:
                    per_item = total_marks / cnt
                    if _is_multiple_of(per_item, increment):
                        singular_type = item_type.rstrip('s')
                        label = f"Each {singular_type.title()}"
                        cid = _make_criterion_id(norm_id, label)
                        criteria = [{
                            "criterion_id": cid,
                            "label": label,
                            "marks": per_item,
                            "source": "question_structure",
                            "allocation_mode": "per_item",
                            "item_count": cnt,
                        }]
                        logger.info(
                            f"Q{norm_id}: Overriding generic rubric {rubric} with count-based criterion from question "
                            f"'{label}' ({cnt} items × {per_item} marks = {total_marks})"
                        )
                        return {
                            "rubric_source": "structured_rubric",
                            "detected": True,
                            "valid": True,
                            "reason": None,
                            "criteria": criteria,
                            "requires_allocation_llm": False,
                        }

            result["rubric_source"] = "structured_rubric"
            result["requires_allocation_llm"] = False
            if not result["valid"]:
                logger.warning(
                    f"Q{norm_id}: Structured rubric detected but invalid: {result['reason']}"
                )
            else:
                logger.info(f"Q{norm_id}: Structured rubric valid ({len(result['criteria'])} criteria)")
            return result

    # --- Tier 2: Embedded rubric in solution text ---
    if solution and solution.strip():
        result = extract_embedded_rubric(solution, total_marks, norm_id, increment)
        if result["detected"]:
            result["rubric_source"] = "embedded_in_solution"
            result["requires_allocation_llm"] = False
            if not result["valid"]:
                logger.warning(
                    f"Q{norm_id}: Embedded rubric detected but invalid: {result['reason']}"
                )
            else:
                logger.info(f"Q{norm_id}: Embedded rubric valid ({len(result['criteria'])} criteria)")
            return result

    # --- Tier 3: Total-only ---
    if total_marks > 0:
        return {
            "rubric_source": "total_only",
            "detected": False,
            "valid": True,
            "reason": None,
            "criteria": [],
            "requires_allocation_llm": True,
        }

    # --- Tier 4: No rubric ---
    return {
        "rubric_source": "none",
        "detected": False,
        "valid": True,
        "reason": None,
        "criteria": [],
        "requires_allocation_llm": True,
    }
