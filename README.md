# Smart Examiner FYP

An AI-powered exam paper evaluation system that automates the grading of handwritten answer sheets by comparing student responses against teacher-provided rubrics.

## Project Structure

```
smart_examiner_fyp/
├── frontend/          # React + Vite UI (port 5173)
├── backend/           # Node.js + Express API gateway (port 5000)
├── ai-service/        # Python + FastAPI AI engine (port 8000)
└── README.md
```

### Frontend (`frontend/`)
- Built with **React** and **Vite**
- Upload teacher rubric (PDF/image) and student answer sheets
- Real-time evaluation progress bar
- Displays detailed scoring results per question

### Backend (`backend/`)
- **Express.js** API gateway
- Routes requests between frontend and AI service
- Handles file uploads via Multer

### AI Service (`ai-service/`)
- **FastAPI** server with the core AI pipeline:
  1. **OCR** — Extracts text from handwritten papers using Gemini Vision
  2. **Blueprint** — Builds structured rubric from teacher's answer key with mark allocation
  3. **Matching** — Maps student answers to rubric criteria
  4. **Scoring** — Deterministic mark calculation
  5. **Evaluation** — Generates detailed feedback per question

## Getting Started

### Prerequisites
- Node.js (v18+)
- Python (3.11+)
- Google Gemini API key

### Setup

1. **AI Service**
   ```bash
   cd ai-service
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   # Create .env with: GEMINI_API_KEY=your_key_here
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. **Backend**
   ```bash
   cd backend
   npm install
   # Create .env with: AI_SERVICE_URL=http://localhost:8000
   node index.js
   ```

3. **Frontend**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

4. Open `http://localhost:5173` in your browser.

## Tech Stack
- **Frontend:** React, Vite, Axios
- **Backend:** Node.js, Express, Multer
- **AI Service:** Python, FastAPI, Google Gemini API
- **OCR:** Gemini Vision API
