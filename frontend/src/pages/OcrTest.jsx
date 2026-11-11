import { useState, useRef, useCallback, useEffect } from 'react';
import './OcrTest.css';

const API_URL = 'http://localhost:8000';

const EVAL_STAGES = [
  {
    startPct: 0,
    endPct: 15,
    message: '📄 Preprocessing document pages & enhancing image quality...',
  },
  {
    startPct: 15,
    endPct: 35,
    message: '🔍 Transcribing handwritten text, formulas & diagrams...',
  },
  {
    startPct: 35,
    endPct: 52,
    message: '📐 Detecting question layout & mapping student answer blocks...',
  },
  {
    startPct: 52,
    endPct: 70,
    message: '🧩 Aligning student responses with teacher marking scheme...',
  },
  {
    startPct: 70,
    endPct: 88,
    message: '🧠 Evaluating concept understanding & checking scientific accuracy...',
  },
  {
    startPct: 88,
    endPct: 100,
    message: '⚖️ Calculating final marks and applying rubric rules...',
  },
];

function CopyButton({ data }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = (e) => {
    e.preventDefault();
    e.stopPropagation();
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <button onClick={handleCopy} style={{ marginLeft: '1rem', padding: '4px 8px', fontSize: '0.75rem', borderRadius: '4px', border: '1px solid #555', background: copied ? '#4ade80' : '#333', color: copied ? '#000' : '#fff', cursor: 'pointer' }}>
      {copied ? '✓ Copied' : '📋 Copy JSON'}
    </button>
  );
}

export default function OcrTest() {
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [extractedText, setExtractedText] = useState('');
  const [structuredJson, setStructuredJson] = useState(null);
  const [groundTruth, setGroundTruth] = useState('');
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  // Teacher states
  const [teacherImage, setTeacherImage] = useState(null);
  const [teacherImagePreview, setTeacherImagePreview] = useState(null);
  const [teacherStructuredJson, setTeacherStructuredJson] = useState(null);
  const [teacherLoading, setTeacherLoading] = useState(false);
  const [teacherError, setTeacherError] = useState('');
  const [teacherDragActive, setTeacherDragActive] = useState(false);
  const teacherFileInputRef = useRef(null);

  // Matching states
  const [matchingResults, setMatchingResults] = useState(null);
  const [matchingLoading, setMatchingLoading] = useState(false);
  const [matchingError, setMatchingError] = useState('');

  // Evaluation states
  const [evaluationResults, setEvaluationResults] = useState(null);
  const [evaluationLoading, setEvaluationLoading] = useState(false);
  const [evaluationError, setEvaluationError] = useState('');

  // Evaluation progress states (1-2 minutes simulation)
  const [evalProgress, setEvalProgress] = useState(0);
  const [evalStageIndex, setEvalStageIndex] = useState(0);
  const evalIntervalRef = useRef(null);
  const evalBackendDataRef = useRef(null);
  const evalFastForwardRef = useRef(false);

  useEffect(() => {
    return () => {
      if (evalIntervalRef.current) clearInterval(evalIntervalRef.current);
    };
  }, []);

  /* ------------------------------------------------------------------ */
  /* File handling                                                       */
  /* ------------------------------------------------------------------ */
  const handleFile = (file) => {
    if (!file) return;
    setImage(file);
    setImagePreview(URL.createObjectURL(file));
    setExtractedText('');
    setStructuredJson(null);
    setMetrics(null);
    setError('');
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragActive(false);
    handleFile(e.dataTransfer.files[0]);
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setDragActive(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setDragActive(false);
  }, []);

  // Teacher file handling
  const handleTeacherFile = (file) => {
    if (!file) return;
    setTeacherImage(file);
    setTeacherImagePreview(URL.createObjectURL(file));
    setTeacherStructuredJson(null);
    setTeacherError('');
  };

  const handleTeacherDrop = useCallback((e) => {
    e.preventDefault();
    setTeacherDragActive(false);
    handleTeacherFile(e.dataTransfer.files[0]);
  }, []);

  const handleTeacherDragOver = useCallback((e) => {
    e.preventDefault();
    setTeacherDragActive(true);
  }, []);

  const handleTeacherDragLeave = useCallback(() => {
    setTeacherDragActive(false);
  }, []);

  /* ------------------------------------------------------------------ */
  /* API calls                                                           */
  /* ------------------------------------------------------------------ */
  const handleExtract = async () => {
    if (!image) return;
    setLoading(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('file', image);
      const res = await fetch(`${API_URL}/api/ocr`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (data.success) {
        setExtractedText(data.extracted_text);
        if (data.structured_json) {
          setStructuredJson(data.structured_json);
        }
        setMetrics({
          confidence: data.confidence,
          lines_detected: data.lines_detected,
        });
      } else {
        setError(data.error || 'Extraction failed');
      }
    } catch {
      setError(
        'Failed to connect to server. Make sure both servers are running.',
      );
    }
    setLoading(false);
  };

  const handleTeacherExtract = async () => {
    if (!teacherImage) return;
    setTeacherLoading(true);
    setTeacherError('');
    try {
      const formData = new FormData();
      formData.append('file', teacherImage);
      const res = await fetch(`${API_URL}/api/teacher-solution`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (data.success) {
        if (data.structured_json) {
          setTeacherStructuredJson(data.structured_json);
        }
      } else {
        setTeacherError(data.error || 'Teacher extraction failed');
      }
    } catch {
      setTeacherError(
        'Failed to connect to server. Make sure both servers are running.',
      );
    }
    setTeacherLoading(false);
  };

  const handleEvaluate = async () => {
    if (!image || !groundTruth.trim()) return;
    setLoading(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('image', image);
      formData.append('ground_truth', groundTruth);
      const res = await fetch(`${API_URL}/api/ocr/evaluate`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (data.success) {
        setExtractedText(data.extracted_text);
        if (data.structured_json) {
          setStructuredJson(data.structured_json);
        }
        setMetrics({
          confidence: data.confidence,
          lines_detected: data.lines_detected,
          cer: data.cer,
          wer: data.wer,
          accuracy: data.accuracy,
        });
      } else {
        setError(data.error || 'Evaluation failed');
      }
    } catch {
      setError(
        'Failed to connect to server. Make sure both servers are running.',
      );
    }
    setLoading(false);
  };

  const handleReset = () => {
    if (evalIntervalRef.current) clearInterval(evalIntervalRef.current);
    setImage(null);
    setImagePreview(null);
    setExtractedText('');
    setStructuredJson(null);
    setGroundTruth('');
    setMetrics(null);
    setError('');
    setMatchingResults(null);
    setMatchingError('');
    setEvaluationResults(null);
    setEvaluationError('');
    setEvaluationLoading(false);
    setEvalProgress(0);
    setEvalStageIndex(0);
  };

  const handleMatch = async () => {
    if (!image || !teacherImage) return;
    setMatchingLoading(true);
    setMatchingError('');
    setMatchingResults(null);
    try {
      const formData = new FormData();
      formData.append('student_file', image);
      formData.append('teacher_file', teacherImage);
      const res = await fetch(`${API_URL}/api/match`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (data.success) {
        setMatchingResults(data);
      } else {
        setMatchingError(data.error || 'Matching failed');
      }
    } catch {
      setMatchingError(
        'Failed to connect to server. Make sure both servers are running.',
      );
    }
    setMatchingLoading(false);
  };

  const handleSkipProgress = () => {
    evalFastForwardRef.current = true;
    if (evalBackendDataRef.current) {
      if (evalIntervalRef.current) clearInterval(evalIntervalRef.current);
      setEvalProgress(100);
      setEvalStageIndex(EVAL_STAGES.length - 1);
      setTimeout(() => {
        setEvaluationResults(evalBackendDataRef.current);
        if (evalBackendDataRef.current.matching_results) {
          setMatchingResults(evalBackendDataRef.current);
        }
        setEvaluationLoading(false);
      }, 300);
    }
  };

  const handleEvaluateGrade = async () => {
    if (!image || !teacherImage) return;

    if (evalIntervalRef.current) clearInterval(evalIntervalRef.current);

    // Pick a natural duration in the 1-2 minutes range (e.g. 72 - 85 seconds)
    const durationSec = 72 + Math.floor(Math.random() * 14);
    const targetDurationMs = durationSec * 1000;

    setEvaluationLoading(true);
    setEvaluationError('');
    setEvaluationResults(null);
    setEvalProgress(0);
    setEvalStageIndex(0);

    evalBackendDataRef.current = null;
    evalFastForwardRef.current = false;

    const startTime = Date.now();

    // 1. Send the backend request in parallel
    const sendBackendRequest = async () => {
      try {
        const formData = new FormData();
        formData.append('student_file', image);
        formData.append('teacher_file', teacherImage);
        const res = await fetch(`${API_URL}/api/evaluate`, {
          method: 'POST',
          body: formData,
        });
        const data = await res.json();
        if (data.success) {
          evalBackendDataRef.current = data;
          // If user already clicked fast-forward before backend returned
          if (evalFastForwardRef.current) {
            if (evalIntervalRef.current) clearInterval(evalIntervalRef.current);
            setEvalProgress(100);
            setEvalStageIndex(EVAL_STAGES.length - 1);
            setTimeout(() => {
              setEvaluationResults(data);
              if (data.matching_results) {
                setMatchingResults(data);
              }
              setEvaluationLoading(false);
            }, 300);
          }
        } else {
          throw new Error(data.error || 'Evaluation failed on server');
        }
      } catch (err) {
        if (evalIntervalRef.current) clearInterval(evalIntervalRef.current);
        setEvaluationLoading(false);
        setEvaluationError(
          err.message || 'Failed to connect to server. Make sure both servers are running.',
        );
      }
    };

    sendBackendRequest();

    // 2. Animate the progress bar smoothly across 1-2 minutes
    evalIntervalRef.current = setInterval(() => {
      const now = Date.now();
      const elapsedMs = now - startTime;

      // Check if user requested fast-forward and data is ready
      if (evalFastForwardRef.current && evalBackendDataRef.current) {
        clearInterval(evalIntervalRef.current);
        setEvalProgress(100);
        setEvalStageIndex(EVAL_STAGES.length - 1);
        setTimeout(() => {
          setEvaluationResults(evalBackendDataRef.current);
          if (evalBackendDataRef.current.matching_results) {
            setMatchingResults(evalBackendDataRef.current);
          }
          setEvaluationLoading(false);
        }, 300);
        return;
      }

      // Compute progress % (capped at 96% until target duration + backend are both ready)
      const rawPct = (elapsedMs / targetDurationMs) * 100;

      if (elapsedMs >= targetDurationMs) {
        // Target duration has elapsed
        if (evalBackendDataRef.current) {
          // Both duration and backend response are ready!
          clearInterval(evalIntervalRef.current);
          setEvalProgress(100);
          setEvalStageIndex(EVAL_STAGES.length - 1);
          setTimeout(() => {
            setEvaluationResults(evalBackendDataRef.current);
            if (evalBackendDataRef.current.matching_results) {
              setMatchingResults(evalBackendDataRef.current);
            }
            setEvaluationLoading(false);
          }, 600);
          return;
        } else {
          // If backend is still calculating, hold at 96%
          setEvalProgress(96);
          setEvalStageIndex(EVAL_STAGES.length - 1);
        }
      } else {
        const cappedPct = Math.min(Math.round(rawPct), 96);
        setEvalProgress(cappedPct);

        const currentStageIdx = EVAL_STAGES.findIndex(
          (stg) => cappedPct >= stg.startPct && cappedPct < stg.endPct,
        );
        if (currentStageIdx !== -1) {
          setEvalStageIndex(currentStageIdx);
        }
      }
    }, 250);
  };

  /* ------------------------------------------------------------------ */
  /* Render                                                              */
  /* ------------------------------------------------------------------ */
  return (
    <div className="ocr-page">
      {/* ---- Header ---- */}
      <header className="ocr-header">
        <div className="logo">
          <span className="logo-icon">🧠</span>
          <h1>Smart Examiner</h1>
        </div>
        <p className="subtitle">
          AI-Powered Handwriting Recognition — OCR Testing Module
        </p>
      </header>

      <main className="ocr-main" style={{ display: 'flex', flexDirection: 'column', gap: '3rem' }}>
        {/* ---- Student Upload ---- */}
        <section className="upload-section">
          <h2>🧑‍🎓 Student Answer Extraction</h2>
          <div
            className={`drop-zone ${dragActive ? 'drag-active' : ''} ${imagePreview ? 'has-image' : ''}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => fileInputRef.current?.click()}
          >
            {imagePreview ? (
              <img
                src={imagePreview}
                alt="Uploaded preview"
                className="preview-image"
              />
            ) : (
              <div className="drop-content">
                <span className="drop-icon">📤</span>
                <p className="drop-title">Drop handwritten document here</p>
                <p className="drop-subtitle">
                  or click to browse — PDF, PNG, JPG, WEBP
                </p>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*,application/pdf"
              onChange={(e) => handleFile(e.target.files[0])}
              hidden
            />
          </div>

          <div className="action-buttons">
            <button
              id="btn-extract"
              className="btn btn-primary"
              onClick={handleExtract}
              disabled={!image || loading}
            >
              {loading ? <span className="spinner" /> : '🔍'} Extract Text
            </button>
            <button
              id="btn-reset"
              className="btn btn-secondary"
              onClick={handleReset}
              disabled={loading}
            >
              🔄 Reset
            </button>
          </div>
        </section>

        {/* ---- Error ---- */}
        {error && (
          <div className="error-banner">
            <span>⚠️</span> {error}
          </div>
        )}

        {/* ---- Extracted Text ---- */}
        {extractedText && (
          <section className="results-section">
            <div className="result-card">
              <h3>📝 Extracted Text</h3>
              <div className="text-output">{extractedText}</div>
            </div>
            {structuredJson && (
              <div className="result-card" style={{ marginTop: '1.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <h3 style={{ margin: 0 }}>🧩 Structured Student Data</h3>
                  <CopyButton data={structuredJson} />
                </div>
                <pre className="text-output" style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', backgroundColor: '#f8fafc', padding: '1rem', borderRadius: '8px', border: '1px solid #e2e8f0', color: '#1e293b', overflowX: 'auto' }}>
                  {JSON.stringify(structuredJson, null, 2)}
                </pre>
              </div>
            )}
          </section>
        )}

        {/* ---- Teacher Upload ---- */}
        <section className="upload-section" style={{ borderTop: '2px solid #e2e8f0', paddingTop: '2rem' }}>
          <h2>👩‍🏫 Teacher Solution / Marking Scheme</h2>
          <div
            className={`drop-zone ${teacherDragActive ? 'drag-active' : ''} ${teacherImagePreview ? 'has-image' : ''}`}
            onDrop={handleTeacherDrop}
            onDragOver={handleTeacherDragOver}
            onDragLeave={handleTeacherDragLeave}
            onClick={() => teacherFileInputRef.current?.click()}
          >
            {teacherImagePreview ? (
              <img
                src={teacherImagePreview}
                alt="Teacher preview"
                className="preview-image"
              />
            ) : (
              <div className="drop-content">
                <span className="drop-icon">📑</span>
                <p className="drop-title">Drop teacher solution/rubric here</p>
                <p className="drop-subtitle">
                  or click to browse — PDF, PNG, JPG, WEBP
                </p>
              </div>
            )}
            <input
              ref={teacherFileInputRef}
              type="file"
              accept="image/*,application/pdf"
              onChange={(e) => handleTeacherFile(e.target.files[0])}
              hidden
            />
          </div>

          <div className="action-buttons">
            <button
              id="btn-teacher-extract"
              className="btn btn-primary"
              onClick={handleTeacherExtract}
              disabled={!teacherImage || teacherLoading}
            >
              {teacherLoading ? <span className="spinner" /> : '🧠'} Extract Teacher Data
            </button>
            <button
              id="btn-teacher-reset"
              className="btn btn-secondary"
              onClick={() => {
                setTeacherImage(null);
                setTeacherImagePreview(null);
                setTeacherStructuredJson(null);
                setTeacherError('');
              }}
              disabled={teacherLoading}
            >
              🔄 Reset
            </button>
          </div>
        </section>

        {teacherError && (
          <div className="error-banner">
            <span>⚠️</span> {teacherError}
          </div>
        )}

        {teacherStructuredJson && (
          <section className="results-section">
            <div className="result-card">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                <h3 style={{ margin: 0 }}>🧩 Structured Teacher Solution</h3>
                <CopyButton data={teacherStructuredJson} />
              </div>
              <pre className="text-output" style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', backgroundColor: '#fdf4ff', padding: '1rem', borderRadius: '8px', border: '1px solid #f0abfc', color: '#701a75', overflowX: 'auto' }}>
                {JSON.stringify(teacherStructuredJson, null, 2)}
              </pre>
            </div>
          </section>
        )}

        {/* ---- Accuracy Evaluation ---- */}
        {extractedText && (
          <section className="evaluate-section">
            <div className="evaluate-card">
              <h3>🎯 Accuracy Evaluation</h3>
              <p className="evaluate-desc">
                Paste the actual text (ground truth) below to calculate accuracy
                metrics.
              </p>
              <textarea
                id="ground-truth-input"
                className="ground-truth-input"
                placeholder="Type or paste the actual handwritten text here…"
                value={groundTruth}
                onChange={(e) => setGroundTruth(e.target.value)}
                rows={5}
              />
              <button
                id="btn-evaluate"
                className="btn btn-accent"
                onClick={handleEvaluate}
                disabled={!groundTruth.trim() || loading}
              >
                {loading ? <span className="spinner" /> : '📊'} Evaluate
                Accuracy
              </button>
            </div>
          </section>
        )}

        {/* ---- Metrics ---- */}
        {metrics && (
          <section className="metrics-section">
            <h3>📊 Metrics</h3>
            <div className="metrics-grid">
              {/* Confidence */}
              <div className="metric-card">
                <span className="metric-label">Confidence</span>
                <span className="metric-value">{metrics.confidence}%</span>
                <div className="metric-bar">
                  <div
                    className="metric-fill confidence"
                    style={{ width: `${metrics.confidence}%` }}
                  />
                </div>
              </div>

              {/* Lines Detected */}
              <div className="metric-card">
                <span className="metric-label">Lines Detected</span>
                <span className="metric-value">{metrics.lines_detected}</span>
              </div>

              {/* Accuracy metrics (shown after evaluation) */}
              {metrics.accuracy !== undefined && (
                <>
                  <div className="metric-card">
                    <span className="metric-label">Accuracy</span>
                    <span className="metric-value accent">
                      {metrics.accuracy}%
                    </span>
                    <div className="metric-bar">
                      <div
                        className="metric-fill accuracy"
                        style={{
                          width: `${Math.max(0, metrics.accuracy)}%`,
                        }}
                      />
                    </div>
                  </div>

                  <div className="metric-card">
                    <span className="metric-label">CER</span>
                    <span className="metric-value warn">{metrics.cer}%</span>
                    <div className="metric-bar">
                      <div
                        className="metric-fill cer"
                        style={{
                          width: `${Math.min(100, metrics.cer)}%`,
                        }}
                      />
                    </div>
                  </div>

                  <div className="metric-card">
                    <span className="metric-label">WER</span>
                    <span className="metric-value warn">{metrics.wer}%</span>
                    <div className="metric-bar">
                      <div
                        className="metric-fill wer"
                        style={{
                          width: `${Math.min(100, metrics.wer)}%`,
                        }}
                      />
                    </div>
                  </div>
                </>
              )}
            </div>
          </section>
        )}
      {/* ---- Match & Assess Section ---- */}
        <section className="upload-section" style={{ borderTop: '2px solid #4CAF50', paddingTop: '2rem' }}>
          <h2>🔗 Match & Assess</h2>
          <p style={{ color: '#aaa', marginBottom: '1rem' }}>
            Upload both student and teacher documents above, then click Match & Assess to run the full comparison pipeline.
          </p>
          <button
            className="action-btn extract"
            onClick={handleMatch}
            disabled={!image || !teacherImage || matchingLoading}
            style={{ marginBottom: '1rem' }}
          >
            {matchingLoading ? '⏳ Matching...' : '🔗 Match & Assess'}
          </button>

          {matchingError && (
            <div className="error-message">{matchingError}</div>
          )}

          {matchingResults && (
            <div style={{ marginTop: '1rem' }}>
              <h3>Matching Results</h3>
              {matchingResults.matching_results?.map((studentResult, sIdx) => (
                <div key={sIdx} style={{ marginBottom: '2rem', padding: '1rem', border: '1px solid #333', borderRadius: '8px' }}>
                  <h4>
                    Student {studentResult.student_index}: {studentResult.student_identity?.name || 'Unknown'}
                    {studentResult.student_identity?.roll_no && ` (${studentResult.student_identity.roll_no})`}
                  </h4>
                  {studentResult.matches?.map((match, mIdx) => (
                    <div key={mIdx} style={{ marginLeft: '1rem', marginBottom: '1.5rem', padding: '0.8rem', background: '#1a1a2e', borderRadius: '6px' }}>
                      <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', marginBottom: '0.5rem' }}>
                        <strong>Q{match.question_mapping?.student_normalized_id}</strong>
                        <span style={{
                          padding: '2px 8px',
                          borderRadius: '4px',
                          fontSize: '0.8rem',
                          background: match.question_mapping?.mapping_status === 'matched' ? '#2d5a2d' : '#5a2d2d',
                          color: '#fff'
                        }}>
                          {match.question_mapping?.mapping_status}
                        </span>
                        {match.student_assessment?.overall_status && (
                          <span style={{
                            padding: '2px 8px',
                            borderRadius: '4px',
                            fontSize: '0.8rem',
                            background:
                              match.student_assessment.overall_status === 'correct' ? '#1b5e20' :
                              match.student_assessment.overall_status === 'partial' ? '#e65100' :
                              match.student_assessment.overall_status === 'incorrect' ? '#b71c1c' :
                              match.student_assessment.overall_status === 'unanswered' ? '#4a148c' :
                              match.student_assessment.overall_status === 'error' ? '#c62828' : '#424242',
                            color: '#fff'
                          }}>
                            {match.student_assessment.overall_status}
                          </span>
                        )}
                        {match.student_assessment?.confidence !== undefined && (
                          <span style={{ fontSize: '0.75rem', color: '#888' }}>
                            conf: {(match.student_assessment.confidence * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>

                      {/* Requirements */}
                      {match.student_assessment?.requirements?.length > 0 && (
                        <div style={{ marginLeft: '1rem', fontSize: '0.85rem' }}>
                          {match.student_assessment.requirements.map((req, rIdx) => (
                            <div key={rIdx} style={{ marginBottom: '4px' }}>
                              <span style={{
                                display: 'inline-block', width: '12px', height: '12px', borderRadius: '50%', marginRight: '6px',
                                background:
                                  req.status === 'satisfied' ? '#4CAF50' :
                                  req.status === 'partial' ? '#FF9800' :
                                  req.status === 'missing' ? '#f44336' :
                                  req.status === 'incorrect' ? '#b71c1c' : '#9e9e9e'
                              }} />
                              <strong>{req.requirement_id}</strong>: {req.status}
                              {req.evidence && <span style={{ color: '#888' }}> — {req.evidence}</span>}
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Conceptual Errors or System Errors */}
                      {match.student_assessment?.overall_status === 'error' && (
                        <div style={{ marginLeft: '1rem', marginTop: '0.5rem', color: '#ff5252', fontSize: '0.85rem' }}>
                          <strong>System Error:</strong> {match.student_assessment.error?.message || 'Unknown API failure'}
                        </div>
                      )}
                      
                      {match.student_assessment?.conceptual_errors?.length > 0 && (
                        <div style={{ marginLeft: '1rem', marginTop: '0.5rem', color: '#ff6b6b', fontSize: '0.85rem' }}>
                          <strong>Errors:</strong>
                          {match.student_assessment.conceptual_errors.map((err, eIdx) => (
                            <div key={eIdx}>⚠️ [{err.severity}] {err.description}</div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ))}

              <details style={{ marginTop: '1rem' }}>
                <summary style={{ cursor: 'pointer', color: '#888', display: 'flex', alignItems: 'center' }}>
                  <span>📋 Raw Matching JSON</span>
                  <CopyButton data={matchingResults} />
                </summary>
                <pre style={{ maxHeight: '400px', overflow: 'auto', fontSize: '0.75rem', background: '#0a0a0a', padding: '1rem', borderRadius: '6px' }}>
                  {JSON.stringify(matchingResults, null, 2)}
                </pre>
              </details>
            </div>
          )}
        </section>

        {/* ---- Evaluate & Grade Section ---- */}
        <section className="upload-section" style={{ borderTop: '2px solid #FF9800', paddingTop: '2rem' }}>
          <h2>📝 Evaluate & Grade</h2>
          <p style={{ color: '#aaa', marginBottom: '1rem' }}>
            Upload both student and teacher documents, then click Evaluate & Grade to run the full pipeline: OCR → Matching → Evaluation.
          </p>
          <button
            id="btn-evaluate-grade"
            className="action-btn extract"
            onClick={handleEvaluateGrade}
            disabled={!image || !teacherImage || evaluationLoading}
            style={{ marginBottom: '1rem', background: '#e65100' }}
          >
            {evaluationLoading ? '⏳ AI Evaluation in Progress...' : '📝 Evaluate & Grade'}
          </button>

          {evaluationLoading && (
            <div className="standard-progress-box">
              <div className="standard-progress-header">
                <span className="standard-progress-text">
                  <span className="standard-progress-spinner">⏳</span>
                  {evalProgress >= 100
                    ? '✅ Finalizing evaluation results...'
                    : EVAL_STAGES[evalStageIndex]?.message || 'Evaluating paper...'}
                </span>
                <span className="standard-progress-pct">{evalProgress}%</span>
              </div>
              <div className="standard-progress-track">
                <div
                  className="standard-progress-fill"
                  style={{ width: `${evalProgress}%` }}
                />
              </div>
            </div>
          )}

          {evaluationError && (
            <div className="error-message">{evaluationError}</div>
          )}

          {evaluationResults?.evaluation_results && (
            <div style={{ marginTop: '1rem' }}>
              <h3>📊 Evaluation Results</h3>
              <div style={{ fontSize: '0.8rem', color: '#888', marginBottom: '1rem' }}>
                Matching API calls: {evaluationResults.matching_api_calls || 0} | 
                Evaluation API calls: {evaluationResults.evaluation_api_calls || 0} |
                Model: {evaluationResults.evaluation_metadata?.model_name || 'unknown'} |
                Increment: {evaluationResults.evaluation_metadata?.grading_increment || 0.5}
              </div>

              {evaluationResults.evaluation_results.map((studentEval, sIdx) => (
                <div key={sIdx} style={{ marginBottom: '2rem', padding: '1.5rem', border: '1px solid #444', borderRadius: '8px', background: '#111' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                    <h4 style={{ margin: 0 }}>
                      Student {studentEval.student_index}: {studentEval.student_identity?.name || 'Unknown'}
                      {studentEval.student_identity?.roll_no && ` (${studentEval.student_identity.roll_no})`}
                    </h4>
                    <div style={{
                      padding: '6px 16px',
                      borderRadius: '20px',
                      fontSize: '1.1rem',
                      fontWeight: 'bold',
                      background: studentEval.total_marks_awarded >= studentEval.total_marks_available * 0.7 ? '#1b5e20' :
                                  studentEval.total_marks_awarded >= studentEval.total_marks_available * 0.4 ? '#e65100' : '#b71c1c',
                      color: '#fff'
                    }}>
                      {studentEval.total_marks_awarded} / {studentEval.total_marks_available}
                      <span style={{ fontSize: '0.8rem', marginLeft: '8px' }}>
                        ({studentEval.total_marks_available > 0 ? ((studentEval.total_marks_awarded / studentEval.total_marks_available) * 100).toFixed(0) : 0}%)
                      </span>
                    </div>
                  </div>

                  {studentEval.questions?.map((qEval, qIdx) => (
                    <div key={qIdx} style={{ marginLeft: '1rem', marginBottom: '1rem', padding: '0.8rem', background: '#1a1a2e', borderRadius: '6px' }}>
                      <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', marginBottom: '0.5rem' }}>
                        <strong>Q{qEval.student_normalized_question_id}</strong>
                        <span style={{
                          padding: '2px 8px',
                          borderRadius: '4px',
                          fontSize: '0.8rem',
                          background: qEval.evaluation_status === 'evaluated' ? '#2d5a2d' :
                                      qEval.evaluation_status === 'error' ? '#c62828' : '#5a2d2d',
                          color: '#fff'
                        }}>
                          {qEval.evaluation_status}
                        </span>
                        <span style={{ fontWeight: 'bold', fontSize: '0.95rem' }}>
                          {qEval.question_marks} / {qEval.total_marks}
                        </span>
                        {qEval.grading_confidence !== undefined && (
                          <span style={{ fontSize: '0.75rem', color: '#888' }}>
                            conf: {(qEval.grading_confidence * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>

                      {/* Requirement-level grading table */}
                      {qEval.requirements?.length > 0 && (
                        <table style={{ width: '100%', fontSize: '0.8rem', borderCollapse: 'collapse', marginTop: '0.5rem' }}>
                          <thead>
                            <tr style={{ borderBottom: '1px solid #333', textAlign: 'left' }}>
                              <th style={{ padding: '4px 8px' }}>Requirement</th>
                              <th style={{ padding: '4px 8px' }}>Status</th>
                              <th style={{ padding: '4px 8px' }}>Marks</th>
                              <th style={{ padding: '4px 8px' }}>Source</th>
                              <th style={{ padding: '4px 8px' }}>Reason</th>
                            </tr>
                          </thead>
                          <tbody>
                            {qEval.requirements.map((req, rIdx) => (
                              <tr key={rIdx} style={{ borderBottom: '1px solid #222' }}>
                                <td style={{ padding: '4px 8px', maxWidth: '200px', wordBreak: 'break-word' }}>
                                  {req.requirement_id}
                                  {req.matching_was_uncertain && <span title="Matching was uncertain" style={{ color: '#FF9800' }}> ⚠</span>}
                                </td>
                                <td style={{ padding: '4px 8px' }}>
                                  <span style={{
                                    display: 'inline-block', width: '10px', height: '10px', borderRadius: '50%', marginRight: '4px',
                                    background: req.status === 'satisfied' ? '#4CAF50' :
                                                req.status === 'partial' ? '#FF9800' :
                                                req.status === 'missing' ? '#f44336' :
                                                req.status === 'incorrect' ? '#b71c1c' : '#9e9e9e'
                                  }} />
                                  {req.status}
                                </td>
                                <td style={{ padding: '4px 8px', fontWeight: 'bold' }}>
                                  {req.awarded_marks} / {req.max_marks}
                                </td>
                                <td style={{ padding: '4px 8px', color: req.allocation_source === 'inferred' ? '#FF9800' : '#4CAF50', fontSize: '0.75rem' }}>
                                  {req.allocation_source}
                                </td>
                                <td style={{ padding: '4px 8px', color: '#aaa', fontSize: '0.75rem', maxWidth: '300px', wordBreak: 'break-word' }}>
                                  {req.reason}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      )}

                      {/* Error display */}
                      {qEval.evaluation_status === 'error' && (
                        <div style={{ marginTop: '0.5rem', color: '#ff5252', fontSize: '0.85rem' }}>
                          <strong>Error:</strong> {qEval.error?.message || 'Evaluation failed'}
                        </div>
                      )}
                    </div>
                  ))}

                  {/* Validation warnings */}
                  {studentEval.validation_warnings?.length > 0 && (
                    <details style={{ marginTop: '0.5rem' }}>
                      <summary style={{ cursor: 'pointer', color: '#FF9800', fontSize: '0.8rem' }}>⚠ Validation Warnings ({studentEval.validation_warnings.length})</summary>
                      <ul style={{ fontSize: '0.75rem', color: '#FF9800' }}>
                        {studentEval.validation_warnings.map((w, wIdx) => (
                          <li key={wIdx}>{w}</li>
                        ))}
                      </ul>
                    </details>
                  )}
                </div>
              ))}

              <details style={{ marginTop: '1rem' }}>
                <summary style={{ cursor: 'pointer', color: '#888', display: 'flex', alignItems: 'center' }}>
                  <span>📋 Raw Evaluation JSON</span>
                  <CopyButton data={evaluationResults} />
                </summary>
                <pre style={{ maxHeight: '400px', overflow: 'auto', fontSize: '0.75rem', background: '#0a0a0a', padding: '1rem', borderRadius: '6px' }}>
                  {JSON.stringify(evaluationResults, null, 2)}
                </pre>
              </details>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
