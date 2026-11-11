"""
Smart Examiner - AI OCR Service
Uses Microsoft TrOCR (Large Handwritten) for handwritten text recognition.
"""

import io
import logging
import numpy as np
import cv2
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from jiwer import wer as compute_wer, cer as compute_cer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="Smart Examiner — OCR Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
MODEL_NAME = "./trocr-large-handwritten"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

logger.info(f"Loading TrOCR model ({MODEL_NAME}) — this may take a minute on first run …")
processor = TrOCRProcessor.from_pretrained(MODEL_NAME, use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)
model.eval()
logger.info("TrOCR model loaded successfully!")


# ---------------------------------------------------------------------------
# Line Segmentation
# ---------------------------------------------------------------------------
def segment_lines(image: Image.Image) -> list:
    """
    Split a page image into individual text-line crops using
    advanced contour detection and ruled-line removal, optimized for exam papers.
    """
    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]

    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # 1. Adaptive Thresholding to handle varying lighting/shadows
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )

    # 2. Remove horizontal ruled lines (notebook paper)
    horiz_len = max(img_w // 40, 20)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    detect_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    cnts, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        cv2.drawContours(binary, [c], -1, (0, 0, 0), 5)

    # 3. Remove vertical margin lines
    vert_len = max(img_h // 40, 20)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
    detect_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    cnts, _ = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        cv2.drawContours(binary, [c], -1, (0, 0, 0), 5)

    # 4. Dilate to connect characters into words
    k_x = max(img_w // 80, 15)
    k_y = max(img_h // 300, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_x, k_y))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # 5. Find contours of the text components
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out tiny noise and giant page borders
        if w > 10 and h > 10 and h < img_h * 0.2:
            bounding_boxes.append([x, y, w, h])

    # 6. Group bounding boxes into text lines
    bounding_boxes.sort(key=lambda b: b[1])  # Sort top-to-bottom

    lines = []
    for box in bounding_boxes:
        if not lines:
            lines.append([box])
            continue
            
        current_line = lines[-1]
        
        # Calculate median center-Y of current line
        centers_y = [b[1] + b[3]/2 for b in current_line]
        median_cy = sorted(centers_y)[len(centers_y)//2]
        
        # Calculate average height
        avg_h = sum(b[3] for b in current_line) / len(current_line)
        
        box_cy = box[1] + box[3]/2
        
        # If this box's center is within average height of the median center, it's on the same line
        if abs(box_cy - median_cy) < avg_h * 0.6:
            current_line.append(box)
        else:
            lines.append([box])

    # 7. Crop the grouped lines
    line_images = []
    for line in lines:
        x_min = min(b[0] for b in line)
        y_min = min(b[1] for b in line)
        x_max = max(b[0] + b[2] for b in line)
        y_max = max(b[1] + b[3] for b in line)

        pad_x = max(img_w // 100, 10)
        pad_y = max(img_h // 100, 10)
        
        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(img_w, x_max + pad_x)
        y2 = min(img_h, y_max + pad_y)

        line_images.append(image.crop((x1, y1, x2, y2)))

    return line_images if line_images else [image]


# ---------------------------------------------------------------------------
# OCR — Single Line
# ---------------------------------------------------------------------------
def extract_text_single_line(line_image: Image.Image) -> tuple:
    """Run TrOCR on a single text-line image and return (text, confidence)."""
    pixel_values = processor(
        images=line_image, return_tensors="pt"
    ).pixel_values.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=256,
        )

    # Decode
    text = processor.batch_decode(
        outputs.sequences, skip_special_tokens=True
    )[0].strip()

    # Confidence — average of per-token max softmax probabilities
    scores = outputs.scores
    if scores:
        confs = [torch.softmax(s, dim=-1).max().item() for s in scores]
        avg_conf = sum(confs) / len(confs)
    else:
        avg_conf = 0.0

    return text, avg_conf


# ---------------------------------------------------------------------------
# OCR — Full Image (multi-line)
# ---------------------------------------------------------------------------
def extract_text_from_image(image: Image.Image) -> tuple:
    """Segment → OCR each line → combine results."""
    lines = segment_lines(image)
    logger.info(f"Detected {len(lines)} text line(s)")

    texts: list[str] = []
    confs: list[float] = []

    import os
    DEBUG_DIR = "debug_crops"
    os.makedirs(DEBUG_DIR, exist_ok=True)
    
    # clear old debug crops
    for f in os.listdir(DEBUG_DIR):
        if f.endswith(".jpg") or f.endswith(".png"):
            os.remove(os.path.join(DEBUG_DIR, f))
            
    for idx, line_img in enumerate(lines):
        # save debug image
        line_img.save(os.path.join(DEBUG_DIR, f"line_{idx:02d}.jpg"))
        
        text, conf = extract_text_single_line(line_img)
        logger.info(f"  Line {idx + 1}: conf={conf:.2%}  text={text!r}")
        if text:
            texts.append(text)
            confs.append(conf)

    full_text = "\n".join(texts)
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return full_text, avg_conf, len(lines)


# ===================================================================
# Endpoints
# ===================================================================
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": str(device),
    }


@app.post("/api/ocr")
async def ocr_extract(file: UploadFile = File(...)):
    """Extract text from a handwritten image."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        text, confidence, num_lines = extract_text_from_image(image)
        return {
            "success": True,
            "extracted_text": text,
            "confidence": round(confidence * 100, 2),
            "lines_detected": num_lines,
        }
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
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        text, confidence, num_lines = extract_text_from_image(image)

        # Compute metrics
        if text.strip() and ground_truth.strip():
            cer_score = compute_cer(ground_truth, text)
            wer_score = compute_wer(ground_truth, text)
            accuracy = max(0.0, (1 - cer_score)) * 100
        else:
            cer_score = 1.0
            wer_score = 1.0
            accuracy = 0.0

        return {
            "success": True,
            "extracted_text": text,
            "confidence": round(confidence * 100, 2),
            "lines_detected": num_lines,
            "cer": round(cer_score * 100, 2),
            "wer": round(wer_score * 100, 2),
            "accuracy": round(accuracy, 2),
        }
    except Exception as exc:
        logger.exception("OCR evaluation failed")
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
