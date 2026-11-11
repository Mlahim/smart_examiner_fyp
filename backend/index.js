/**
 * Smart Examiner — Express Backend Server
 *
 * Acts as the API gateway between the React frontend and the
 * Python AI micro-service (FastAPI + TrOCR).
 */

const express = require("express");
const cors = require("cors");
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");
require("dotenv").config();

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const PORT = process.env.PORT || 5000;
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || "http://localhost:8000";

const app = express();

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Multer — store uploads in memory (small images only for now)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }, // 10 MB
  fileFilter: (_req, file, cb) => {
    const allowed = [
      "image/png",
      "image/jpeg",
      "image/jpg",
      "image/webp",
      "image/bmp",
      "image/tiff",
      "application/pdf",
    ];
    if (allowed.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error("Invalid file type. Only image files are allowed."));
    }
  },
});

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

/** Health-check — also pings the AI service. */
app.get("/api/health", async (_req, res) => {
  try {
    const aiRes = await axios.get(`${AI_SERVICE_URL}/health`, {
      timeout: 5000,
    });
    res.json({ server: "ok", aiService: aiRes.data });
  } catch {
    res.json({ server: "ok", aiService: "unavailable" });
  }
});

/** OCR Extract — upload an image, get back extracted text. */
app.post("/api/ocr/extract", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res
        .status(400)
        .json({ success: false, error: "No image file uploaded" });
    }

    const form = new FormData();
    form.append("file", req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });

    const aiRes = await axios.post(`${AI_SERVICE_URL}/api/ocr`, form, {
      headers: form.getHeaders(),
      timeout: 1800_000, // 10 min — model can be slow on CPU
    });

    res.json(aiRes.data);
  } catch (err) {
    console.error("OCR extract error:", err.message);
    res.status(500).json({
      success: false,
      error: err.response?.data?.error || "Failed to process image",
    });
  }
});

/** OCR Evaluate — upload image + ground truth, get accuracy metrics. */
app.post("/api/ocr/evaluate", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res
        .status(400)
        .json({ success: false, error: "No image file uploaded" });
    }
    if (!req.body.ground_truth) {
      return res
        .status(400)
        .json({ success: false, error: "Ground truth text is required" });
    }

    const form = new FormData();
    form.append("file", req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });
    form.append("ground_truth", req.body.ground_truth);

    const aiRes = await axios.post(
      `${AI_SERVICE_URL}/api/ocr/evaluate`,
      form,
      {
        headers: form.getHeaders(),
        timeout: 1800_000, // 10 min — model can be slow on CPU
      },
    );

    res.json(aiRes.data);
  } catch (err) {
    console.error("OCR evaluate error:", err.message);
    res.status(500).json({
      success: false,
      error: err.response?.data?.error || "Failed to evaluate",
    });
  }
});

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------
app.use((err, _req, res, _next) => {
  if (err instanceof multer.MulterError) {
    return res.status(400).json({ success: false, error: err.message });
  }
  res.status(500).json({ success: false, error: err.message });
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
app.listen(PORT, () => {
  console.log(`🚀  Smart Examiner Server running on port ${PORT}`);
  console.log(`📡  AI Service URL: ${AI_SERVICE_URL}`);
});
