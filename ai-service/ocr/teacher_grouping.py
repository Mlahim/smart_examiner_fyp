import json
import re
import logging

import config
from ocr.provider_factory import get_llm_provider

logger = logging.getLogger(__name__)

# Same exact normalization as grouping.py to ensure 100% compatibility
def _normalize_question_id(q_id: str) -> str:
    digit_match = re.search(r'\d+', q_id)
    if digit_match:
        return str(int(digit_match.group(0)))
    return q_id

def _normalize_subquestion_id(sq_id: str) -> str:
    norm = sq_id.lower()
    if norm.startswith("part"):
        norm = norm.replace("part", "").strip()
    norm = re.sub(r'[\(\)\.\-:\s]', '', norm)
    return norm

def group_teacher_solution(parsed_pages: list) -> dict:
    """
    Takes the raw OCR pages and uses LLM to structure them into the Teacher JSON schema.
    Applies strict Python normalization to ensure perfect ID compatibility with the student structure.
    """
    if not parsed_pages:
        return {"pages": [], "questions": []}
        
    prompt_path = config.BASE_DIR / "prompts" / "teacher_grouping_prompt.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_instruction = f.read().strip()
        
    # Send the raw parsed pages as JSON text to LLM
    pages_json_str = json.dumps(parsed_pages, indent=2)
    
    logger.info("Calling LLM for Teacher Solution structuring pass...")
    try:
        llm = get_llm_provider()
        text, _ = llm.generate_json(
            model_name=config.GEMMA_MODEL_NAME,
            system_instruction=system_instruction,
            prompt_parts=[pages_json_str]
        )
        
        result = json.loads(text)
        
        # Ensure 'pages' array exists at root
        if "pages" not in result:
            all_pages = set()
            for page in parsed_pages:
                if "page_number" in page:
                    all_pages.add(page["page_number"])
            result["pages"] = sorted(list(all_pages))
            
        if "questions" not in result:
            result["questions"] = []
            
        # Enforce canonical normalized IDs
        for q in result.get("questions", []):
            raw_q_id = q.get("question_id", "")
            q["normalized_question_id"] = _normalize_question_id(raw_q_id)
            
            for sq in q.get("subquestions", []):
                raw_sq_id = sq.get("subquestion_id", "")
                sq["normalized_subquestion_id"] = _normalize_subquestion_id(raw_sq_id)
                
        return result
        
    except Exception as e:
        logger.exception("Failed to group teacher solution via LLM")
        return {"pages": [], "questions": [], "error": str(e)}
