"""
Smart Examiner — FastAPI Entry Point

Endpoints:
  GET  /health            → Service health check
  POST /api/ocr           → Extract text from handwritten image
  POST /api/ocr/evaluate  → Extract text and compare against ground truth
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

import config
from ocr.preprocessor import prepare_document, PreprocessingError
from ocr.evaluator import evaluate

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded globals (heavy models)
# ---------------------------------------------------------------------------
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, release on shutdown."""
    global pipeline

    if config.OCR_ENGINE == "gemma":
        from ocr.gemma_ocr import GemmaOCR
        logger.info("Initializing Gemma OCR...")
        pipeline = GemmaOCR()
    elif config.OCR_ENGINE == "qwen":
        from ocr.qwen_vl import QwenOCR
        logger.info("Initializing Qwen2.5-VL OCR...")
        pipeline = QwenOCR()
    elif config.OCR_ENGINE == "kraken_trocr":
        from ocr.kraken_detector import KrakenDetector
        from ocr.recognizer import TrOCRRecognizer
        from ocr.pipeline import OCRPipeline
        detector = KrakenDetector()
        recognizer = TrOCRRecognizer()
        pipeline = OCRPipeline(detector=detector, recognizer=recognizer)
    else:
        from ocr.detector import PaddleOCRDetector
        from ocr.recognizer import TrOCRRecognizer
        from ocr.pipeline import OCRPipeline

        logger.info("Initializing PaddleOCR + TrOCR models...")
        detector = PaddleOCRDetector()
        recognizer = TrOCRRecognizer()
        pipeline = OCRPipeline(detector=detector, recognizer=recognizer)

    logger.info(f"Engine '{config.OCR_ENGINE}' loaded. Server is ready.")
    yield

    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Smart Examiner — OCR Service",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================================================================
# Endpoints
# ===================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine": config.OCR_ENGINE,
        "device": config.TROCR_DEVICE,
    }


@app.post("/api/ocr")
async def ocr_extract(file: UploadFile = File(...)):
    """Extract text from a handwritten image."""
    try:
        contents = await file.read()
        document, mime_type = prepare_document(contents, file.filename or "upload")
        result = pipeline.extract_text(document, mime_type=mime_type)

        import json
        raw_gemini_obj = []
        try:
            raw_gemini_obj = json.loads(result.get("raw_json_string", "[]"))
        except:
            pass

        response = {
            "success": result.get("error") is None,
            "engine": result.get("engine"),
            "extracted_text": result.get("text", ""),
            "pages": result.get("pages", []),
            "grouped_data": result.get("grouped_data", {}),
            "structured_json": result.get("grouped_data", {}),
            "processing_time_ms": round(result.get("processing_time_ms", 0), 0),
            "api_response_time_ms": result.get("api_response_time_ms"),
            "token_usage": result.get("token_usage"),
            "error": result.get("error"),
        }
        
        # Add legacy fields if available
        if result.get("engine") == "paddle_trocr":
            response["confidence"] = round(result.get("avg_confidence", 0) * 100, 2)
            response["lines_detected"] = result.get("lines_detected")
            response["lines_recognized"] = result.get("lines_recognized")
            
        return response
    except PreprocessingError as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        logger.exception("OCR extraction failed")
        return {"success": False, "error": str(exc)}


@app.post("/api/ocr/evaluate")
async def ocr_evaluate(
    file: UploadFile = File(...),
    ground_truth: str = Form(...),
):
    """Extract text and evaluate accuracy against ground truth."""
    try:
        contents = await file.read()
        document, mime_type = prepare_document(contents, file.filename or "upload")
        result = pipeline.extract_text(document, mime_type=mime_type)
        
        extracted_text = result.get("text", "")
        eval_result = evaluate(extracted_text, ground_truth)

        response = {
            "success": result.get("error") is None,
            "engine": result.get("engine"),
            "extracted_text": extracted_text,
            "pages": result.get("pages", []),
            "grouped_data": result.get("grouped_data", {}),
            "structured_json": result.get("grouped_data", {}),
            "processing_time_ms": round(result.get("processing_time_ms", 0), 0),
            "api_response_time_ms": result.get("api_response_time_ms"),
            "token_usage": result.get("token_usage"),
            "error": result.get("error"),
            "cer": round(eval_result.cer * 100, 2),
            "wer": round(eval_result.wer * 100, 2),
            "accuracy": round(eval_result.accuracy, 2),
        }
        
        # Add legacy fields if available
        if result.get("engine") == "paddle_trocr":
            response["confidence"] = round(result.get("avg_confidence", 0) * 100, 2)
            response["lines_detected"] = result.get("lines_detected")
            response["lines_recognized"] = result.get("lines_recognized")

        return response
    except PreprocessingError as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        logger.exception("OCR evaluation failed")
        return {"success": False, "error": str(exc)}

@app.post("/api/teacher-solution")
async def ocr_teacher_solution(file: UploadFile = File(...)):
    """Extract and structure a teacher solution/marking scheme."""
    try:
        contents = await file.read()
        document, mime_type = prepare_document(contents, file.filename or "upload")
        result = pipeline.extract_text(document, mime_type=mime_type, document_type="teacher")

        response = {
            "success": result.get("error") is None,
            "engine": result.get("engine"),
            "extracted_text": result.get("text", ""),
            "pages": result.get("pages", []),
            "structured_json": result.get("grouped_data", {}),
            "processing_time_ms": round(result.get("processing_time_ms", 0), 0),
            "api_response_time_ms": result.get("api_response_time_ms"),
            "token_usage": result.get("token_usage"),
            "error": result.get("error"),
        }
        return response
    except PreprocessingError as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        logger.exception("Teacher OCR extraction failed")
        return {"success": False, "error": str(exc)}


@app.post("/api/match")
async def match_student_teacher(
    student_file: UploadFile = File(...),
    teacher_file: UploadFile = File(...),
):
    """Match student answers against teacher solution using frozen blueprint."""
    try:
        from ocr.blueprint import build_assessment_blueprint
        from ocr.matching import run_matching

        # Simple file-based cache for OCR outputs to prevent OCR non-determinism
        def get_cached_ocr_match(file_contents: bytes, doc_type: str):
            import hashlib, json
            h = hashlib.sha256(file_contents).hexdigest()
            cache_path = config.BASE_DIR / f".ocr_cache_{h}_{doc_type}.json"
            if cache_path.exists():
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass
            return None

        def set_cached_ocr_match(file_contents: bytes, doc_type: str, result: dict):
            import hashlib, json
            h = hashlib.sha256(file_contents).hexdigest()
            cache_path = config.BASE_DIR / f".ocr_cache_{h}_{doc_type}.json"
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(result, f)
            except Exception:
                pass

        # Step 1: Process student document
        student_contents = await student_file.read()
        student_result = get_cached_ocr_match(student_contents, "student")
        if not student_result:
            student_doc, student_mime = prepare_document(student_contents, student_file.filename or "student")
            student_result = pipeline.extract_text(student_doc, mime_type=student_mime, document_type="student")
            if not student_result.get("error"):
                set_cached_ocr_match(student_contents, "student", student_result)

        if student_result.get("error"):
            return {"success": False, "error": f"Student OCR failed: {student_result['error']}"}

        students_data = student_result.get("grouped_data", {}).get("students", [])

        # Step 2: Process teacher document
        teacher_contents = await teacher_file.read()
        teacher_result = get_cached_ocr_match(teacher_contents, "teacher")
        if not teacher_result:
            teacher_doc, teacher_mime = prepare_document(teacher_contents, teacher_file.filename or "teacher")
            teacher_result = pipeline.extract_text(teacher_doc, mime_type=teacher_mime, document_type="teacher")
            if not teacher_result.get("error"):
                set_cached_ocr_match(teacher_contents, "teacher", teacher_result)

        if teacher_result.get("error"):
            return {"success": False, "error": f"Teacher OCR failed: {teacher_result['error']}"}

        teacher_data = teacher_result.get("grouped_data", {}).get("teacher_solution", {})

        # Step 3: Build blueprint
        logger.info("Building frozen assessment blueprint...")
        blueprint = build_assessment_blueprint(teacher_data)

        # Step 4: Run matching pipeline with blueprint
        logger.info("Starting Student ↔ Teacher matching pipeline...")
        matching_results = run_matching(students_data, teacher_data, blueprint)

        return {
            "success": True,
            "api_call_count": matching_results.get("api_call_count", 0) + blueprint.get("api_calls", 0),
            "student_structured": {"students": students_data},
            "teacher_structured": {"teacher_solution": teacher_data},
            "matching_results": matching_results.get("matching_results", []),
            "blueprint": {
                "teacher_input_hash": blueprint.get("teacher_input_hash"),
                "blueprint_version": blueprint.get("blueprint_version"),
                "questions": blueprint.get("questions", {}),
            },
            "student_processing_time_ms": round(student_result.get("processing_time_ms", 0), 0),
            "teacher_processing_time_ms": round(teacher_result.get("processing_time_ms", 0), 0),
        }

    except PreprocessingError as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        logger.exception("Matching failed")
        return {"success": False, "error": str(exc)}


@app.post("/api/evaluate")
async def evaluate_student_answers(
    student_file: UploadFile = File(...),
    teacher_file: UploadFile = File(...),
):
    """Full pipeline: OCR → Blueprint → Matching → Deterministic Scoring.
    
    1. Build frozen assessment blueprint (once per teacher input)
    2. Run matching/evidence analysis (per student)
    3. Run deterministic Python scoring (per student)
    
    No evaluation LLM call — Python is the authoritative source of marks.
    """
    try:
        from ocr.blueprint import build_assessment_blueprint
        from ocr.matching import run_matching
        from ocr.evaluation import run_evaluation
        import hashlib
        import json
        import os

        # Simple file-based cache for OCR outputs to prevent OCR non-determinism
        def get_cached_ocr(file_contents: bytes, doc_type: str):
            h = hashlib.sha256(file_contents).hexdigest()
            cache_path = config.BASE_DIR / f".ocr_cache_{h}_{doc_type}.json"
            if cache_path.exists():
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass
            return None

        def set_cached_ocr(file_contents: bytes, doc_type: str, result: dict):
            h = hashlib.sha256(file_contents).hexdigest()
            cache_path = config.BASE_DIR / f".ocr_cache_{h}_{doc_type}.json"
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(result, f)
            except Exception:
                pass

        # Step 1: Process student document
        student_contents = await student_file.read()
        student_result = get_cached_ocr(student_contents, "student")
        if not student_result:
            student_doc, student_mime = prepare_document(student_contents, student_file.filename or "student")
            student_result = pipeline.extract_text(student_doc, mime_type=student_mime, document_type="student")
            if not student_result.get("error"):
                set_cached_ocr(student_contents, "student", student_result)

        if student_result.get("error"):
            return {"success": False, "error": f"Student OCR failed: {student_result['error']}"}

        students_data = student_result.get("grouped_data", {}).get("students", [])

        # Step 2: Process teacher document
        teacher_contents = await teacher_file.read()
        teacher_result = get_cached_ocr(teacher_contents, "teacher")
        if not teacher_result:
            teacher_doc, teacher_mime = prepare_document(teacher_contents, teacher_file.filename or "teacher")
            teacher_result = pipeline.extract_text(teacher_doc, mime_type=teacher_mime, document_type="teacher")
            if not teacher_result.get("error"):
                set_cached_ocr(teacher_contents, "teacher", teacher_result)

        if teacher_result.get("error"):
            return {"success": False, "error": f"Teacher OCR failed: {teacher_result['error']}"}

        teacher_data = teacher_result.get("grouped_data", {}).get("teacher_solution", {})

        # Step 3: Build frozen assessment blueprint (once per teacher input)
        logger.info("Building frozen assessment blueprint...")
        blueprint = build_assessment_blueprint(teacher_data)

        if blueprint.get("status") == "error":
            return {
                "success": False,
                "error": "Blueprint creation failed",
                "blueprint_warnings": blueprint.get("warnings", []),
            }

        # Step 4: Run matching with blueprint (evidence analysis)
        logger.info("Starting evidence analysis (matching) pipeline...")
        matching_output = run_matching(students_data, teacher_data, blueprint)

        # Step 5: Run deterministic evaluation (scoring engine)
        logger.info("Starting deterministic scoring pipeline...")
        evaluation_output = run_evaluation(
            matching_results=matching_output.get("matching_results", []),
            teacher_data=teacher_data,
            students_data=students_data,
            blueprint=blueprint,
        )

        return {
            "success": True,
            "matching_api_calls": matching_output.get("api_call_count", 0),
            "blueprint_api_calls": blueprint.get("api_calls", 0),
            "evaluation_api_calls": 0,  # No evaluation LLM in new architecture
            "student_structured": {"students": students_data},
            "teacher_structured": {"teacher_solution": teacher_data},
            "blueprint": {
                "teacher_input_hash": blueprint.get("teacher_input_hash"),
                "blueprint_version": blueprint.get("blueprint_version"),
                "grading_policy_version": blueprint.get("grading_policy_version"),
                "scoring_engine_version": blueprint.get("scoring_engine_version"),
                "questions": blueprint.get("questions", {}),
            },
            "matching_results": matching_output.get("matching_results", []),
            "mark_allocations": evaluation_output.get("mark_allocations", []),
            "evaluation_results": evaluation_output.get("evaluation_results", []),
            "evaluation_metadata": evaluation_output.get("metadata", {}),
            "blueprint_warnings": blueprint.get("warnings", []),
            "review_flags": evaluation_output.get("review_flags", []),
            "student_processing_time_ms": round(student_result.get("processing_time_ms", 0), 0),
            "teacher_processing_time_ms": round(teacher_result.get("processing_time_ms", 0), 0),
        }

    except PreprocessingError as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        logger.exception("Evaluation pipeline failed")
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
