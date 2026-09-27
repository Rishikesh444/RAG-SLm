import React, { useState, useEffect, useRef } from 'react';
import { 
  Upload, FileText, Send, Trash2, Cpu, Sparkles, 
  Layers, Database, ShieldCheck, CheckCircle2, AlertCircle, 
  RefreshCw, BarChart2, ChevronDown, ChevronUp
} from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export default function App() {
  // System State
  const [health, setHealth] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  
  // RAG Configuration State
  const [retrievalMode, setRetrievalMode] = useState('hybrid'); // 'hybrid' | 'vector' | 'bm25'
  const [useReranker, setUseReranker] = useState(true);
  const [topK, setTopK] = useState(10);
  
  // Chat State
  const [messages, setMessages] = useState([]);
  const [inputQuestion, setInputQuestion] = useState('');
  const [loadingAnswer, setLoadingAnswer] = useState(false);

  // Accordion state for expanded citations per message index
  const [expandedCitations, setExpandedCitations] = useState({});

  // Evaluation Modal State
  const [showEvalModal, setShowEvalModal] = useState(false);
  const [evalData, setEvalData] = useState({ question: '', answer: '', groundTruth: '' });
  const [evalResult, setEvalResult] = useState(null);
  const [evaluating, setEvaluating] = useState(false);

  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  // Auto-scroll chat to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loadingAnswer]);

  // Fetch API Health on initial render
  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/health`);
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
      }
    } catch (err) {
      setHealth({ status: 'error', document_chunks_indexed: 0 });
    }
  };

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  // Handle Document Upload
  const uploadFile = async (file) => {
    if (!file) return;

    setUploading(true);
    setUploadError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE_URL}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || 'Failed to upload document');
      }

      setDocuments((prev) => [
        ...prev, 
        { name: data.filename, chunks: data.chunks_created, time: new Date().toLocaleTimeString() }
      ]);
      fetchHealth();
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const handleFileUpload = async (e) => {
    await uploadFile(e.target.files[0]);
    e.target.value = '';
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setIsDraggingFile(false);
    const file = e.dataTransfer.files[0];
    if (file) await uploadFile(file);
  };

  // Clear All Documents
  const handleClearDocuments = async () => {
    if (!window.confirm("Clear all uploaded documents and indices?")) return;
    try {
      await fetch(`${API_BASE_URL}/documents`, { method: 'DELETE' });
      setDocuments([]);
      setMessages([]);
      fetchHealth();
    } catch (err) {
      alert("Failed to clear documents.");
    }
  };

  // Toggle Citations Accordion
  const toggleCitations = (index) => {
    setExpandedCitations((prev) => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  // Submit Question to Chat API
  const handleSendQuestion = async (e) => {
    e.preventDefault();
    if (!inputQuestion.trim() || loadingAnswer) return;

    const userQ = inputQuestion.trim();
    setInputQuestion('');

    const userMessage = { role: 'user', content: userQ };
    setMessages((prev) => [...prev, userMessage]);
    setLoadingAnswer(true);

    try {
      const res = await fetch(`${API_BASE_URL}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: userQ,
          top_k: topK,
          search_mode: retrievalMode,
          use_reranker: useReranker
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || 'Failed to generate answer');
      }

      const assistantMessage = {
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        retrieval_mode: data.retrieval_mode,
        use_reranker: data.use_reranker
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      setMessages((prev) => [
        ...prev, 
        { role: 'assistant', content: `❌ Error: ${err.message}`, isError: true }
      ]);
    } finally {
      setLoadingAnswer(false);
    }
  };

  // Submit RAG Evaluation Request
  const handleRunEvaluation = async () => {
    if (!evalData.question || !evalData.answer) return;
    setEvaluating(true);
    try {
      const res = await fetch(`${API_BASE_URL}/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: evalData.question,
          answer: evalData.answer,
          retrieved_context: [evalData.answer],
          ground_truth: evalData.groundTruth || null
        }),
      });
      const data = await res.json();
      setEvalResult(data);
    } catch (err) {
      alert("Evaluation failed: " + err.message);
    } finally {
      setEvaluating(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw', background: 'var(--bg-primary)', overflow: 'hidden' }}>
      
      {/* Top Navbar */}
      <header className="glass-panel" style={{ height: '64px', padding: '0 24px', borderRadius: 0, borderBottom: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', shrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)', padding: '8px', borderRadius: '10px', display: 'flex' }}>
            <Cpu size={22} color="#ffffff" />
          </div>
          <div>
            <h1 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#f3f4f6', letterSpacing: '-0.3px' }}>
              Intelligent Document Research Assistant
            </h1>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Production Hybrid RAG Engine (Vector + BM25 + Cross-Encoder Reranker)
            </span>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'var(--bg-secondary)', padding: '6px 14px', borderRadius: '20px', border: '1px solid var(--border-color)' }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: health?.status === 'ok' ? '#10b981' : '#f43f5e', display: 'inline-block' }} />
            <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>
              {health?.status === 'ok' ? 'System Online' : 'Connecting...'}
            </span>
            <span style={{ color: 'var(--border-color)' }}>|</span>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              {health?.document_chunks_indexed || 0} Chunks Indexed
            </span>
          </div>

          <button className="btn-secondary" onClick={() => setShowEvalModal(true)} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem' }}>
            <BarChart2 size={16} color="#8b5cf6" />
            RAG Evaluation
          </button>
        </div>
      </header>

      {/* Main Body Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', flex: 1, height: 'calc(100vh - 64px)', overflow: 'hidden' }}>
        
        {/* Left Sidebar */}
        <aside style={{ background: 'var(--bg-secondary)', borderRight: '1px solid var(--border-color)', padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px', overflowY: 'auto' }}>
          
          {/* Document Upload Area */}
          <div className="glass-panel" style={{ padding: '16px' }}>
            <h3 style={{ fontSize: '0.88rem', fontWeight: 600, marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Upload size={16} color="#3b82f6" /> Upload Documents
            </h3>
            
            <label
              onDragOver={(e) => { e.preventDefault(); setIsDraggingFile(true); }}
              onDragLeave={() => setIsDraggingFile(false)}
              onDrop={handleDrop}
              style={{ 
              border: '2px dashed var(--border-color)', borderRadius: '10px', padding: '16px 12px', 
              display: 'flex', flexDirection: 'column', alignItems: 'center', cursor: 'pointer', 
              transition: 'border 0.2s, background 0.2s', textAlign: 'center',
              background: isDraggingFile ? 'rgba(59, 130, 246, 0.14)' : 'var(--bg-primary)',
              borderColor: isDraggingFile ? '#3b82f6' : 'var(--border-color)'
            }}>
              <FileText size={26} color="var(--text-muted)" style={{ marginBottom: '6px' }} />
              <span style={{ fontSize: '0.82rem', fontWeight: 600 }}>{isDraggingFile ? 'Drop to Index File' : 'Drop or Click to Upload'}</span>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '2px' }}>PDF, DOCX, TXT</span>
              <input type="file" accept=".pdf,.docx,.txt" onChange={handleFileUpload} style={{ display: 'none' }} />
            </label>

            {uploading && (
              <div style={{ marginTop: '10px', fontSize: '0.78rem', color: '#3b82f6', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <RefreshCw size={14} className="pulsing" /> Processing Document...
              </div>
            )}
            {uploadError && (
              <div style={{ marginTop: '10px', fontSize: '0.75rem', color: '#f43f5e', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <AlertCircle size={14} /> {uploadError}
              </div>
            )}
          </div>

          {/* Active Document List */}
          <div className="glass-panel" style={{ padding: '16px', flex: 1, minHeight: '140px', display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <h3 style={{ fontSize: '0.88rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Database size={16} color="#10b981" /> Active Documents ({documents.length})
              </h3>
              {documents.length > 0 && (
                <button onClick={handleClearDocuments} style={{ background: 'transparent', border: 'none', color: '#f43f5e', cursor: 'pointer' }} title="Clear all">
                  <Trash2 size={16} />
                </button>
              )}
            </div>

            <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {documents.length === 0 ? (
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textAlign: 'center', marginTop: '16px' }}>
                  No documents uploaded yet.
                </div>
              ) : (
                documents.map((doc, idx) => (
                  <div key={idx} style={{ background: 'var(--bg-primary)', padding: '8px 10px', borderRadius: '8px', border: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ overflow: 'hidden' }}>
                      <div style={{ fontSize: '0.8rem', fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{doc.name}</div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{doc.chunks} chunks • {doc.time}</div>
                    </div>
                    <CheckCircle2 size={16} color="#10b981" />
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Retrieval Engine Controls */}
          <div className="glass-panel" style={{ padding: '16px' }}>
            <h3 style={{ fontSize: '0.88rem', fontWeight: 600, marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Layers size={16} color="#8b5cf6" /> Retrieval Controls
            </h3>

            {/* Mode Select */}
            <div style={{ marginBottom: '12px' }}>
              <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>Search Mode</label>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '6px' }}>
                {['hybrid', 'vector', 'bm25'].map((mode) => (
                  <button
                    key={mode}
                    onClick={() => setRetrievalMode(mode)}
                    style={{
                      padding: '6px 4px',
                      fontSize: '0.75rem',
                      fontWeight: 600,
                      borderRadius: '6px',
                      border: '1px solid',
                      borderColor: retrievalMode === mode ? '#8b5cf6' : 'var(--border-color)',
                      background: retrievalMode === mode ? 'rgba(139, 92, 246, 0.2)' : 'var(--bg-primary)',
                      color: retrievalMode === mode ? '#c084fc' : 'var(--text-muted)',
                      cursor: 'pointer',
                      textTransform: 'capitalize'
                    }}
                  >
                    {mode}
                  </button>
                ))}
              </div>
            </div>

            {/* Reranker Toggle */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <div>
                <div style={{ fontSize: '0.8rem', fontWeight: 600 }}>Cross Reranker</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>ms-marco-TinyBERT</div>
              </div>
              <input
                type="checkbox"
                checked={useReranker}
                onChange={(e) => setUseReranker(e.target.checked)}
                style={{ width: 16, height: 16, accentColor: '#8b5cf6', cursor: 'pointer' }}
              />
            </div>

            {/* Top-K Slider */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', fontWeight: 600, marginBottom: '4px' }}>
                <span>Top-K Passages</span>
                <span style={{ color: '#3b82f6' }}>{topK}</span>
              </div>
              <input
                type="range"
                min="1"
                max="20"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
                style={{ width: '100%', accentColor: '#3b82f6', cursor: 'pointer' }}
              />
            </div>
          </div>

        </aside>

        {/* Right Main Chat Panel */}
        <main style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden', background: 'var(--bg-primary)' }}>
          
          {/* Scrollable Message Thread Container */}
          <div 
            ref={chatContainerRef} 
            style={{ 
              flex: 1, 
              padding: '24px', 
              overflowY: 'auto', 
              display: 'flex', 
              flexDirection: 'column', 
              gap: '20px',
              scrollBehavior: 'smooth'
            }}
          >
            {messages.length === 0 ? (
              <div style={{ margin: 'auto', textAlign: 'center', maxWidth: '460px' }}>
                <div style={{ background: 'rgba(59, 130, 246, 0.1)', padding: '20px', borderRadius: '50%', display: 'inline-flex', marginBottom: '16px' }}>
                  <Sparkles size={38} color="#3b82f6" />
                </div>
                <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '8px' }}>Ask Questions About Your Documents</h2>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                  Upload PDF, DOCX, or TXT documents on the left panel to begin querying your knowledge base using Hybrid RAG search!
                </p>
              </div>
            ) : (
              messages.map((msg, idx) => (
                <div 
                  key={idx} 
                  style={{
                    alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                    maxWidth: '85%',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '8px'
                  }}
                >
                  <div 
                    style={{
                      background: msg.role === 'user' ? 'linear-gradient(135deg, #3b82f6, #6366f1)' : 'var(--bg-card)',
                      color: '#ffffff',
                      padding: '14px 18px',
                      borderRadius: msg.role === 'user' ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
                      border: msg.role === 'user' ? 'none' : '1px solid var(--border-color)',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                      fontSize: '0.92rem',
                      lineHeight: '1.65',
                      whiteSpace: 'pre-wrap'
                    }}
                  >
                    {msg.content}
                  </div>

                  {/* Collapsible Source Citations Accordion */}
                  {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                    <div className="glass-panel" style={{ padding: '10px 14px', fontSize: '0.8rem' }}>
                      <button 
                        onClick={() => toggleCitations(idx)}
                        style={{
                          background: 'transparent',
                          border: 'none',
                          color: '#60a5fa',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          justify: 'space-between',
                          width: '100%',
                          fontWeight: 600,
                          fontSize: '0.8rem',
                          padding: '4px 0'
                        }}
                      >
                        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <ShieldCheck size={15} color="#10b981" /> 
                          Source Citations ({msg.sources.length} Passages)
                        </span>
                        {expandedCitations[idx] ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                      </button>

                      {/* Expanded Passages */}
                      {expandedCitations[idx] && (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '10px' }}>
                          {msg.sources.map((src, sIdx) => (
                            <div key={sIdx} style={{ background: 'var(--bg-primary)', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                                <span style={{ fontWeight: 600, color: '#60a5fa' }}>{src.source} (Chunk #{src.chunk_index})</span>
                                <div style={{ display: 'flex', gap: '6px' }}>
                                  {src.rerank_score !== null && src.rerank_score !== undefined && (
                                    <span className="badge badge-purple">Rerank: {src.rerank_score}</span>
                                  )}
                                  {src.rrf_score && <span className="badge badge-amber">RRF: {src.rrf_score}</span>}
                                  <span className="badge badge-blue">Score: {src.similarity_score}</span>
                                </div>
                              </div>
                              <div style={{ color: 'var(--text-muted)', fontStyle: 'italic', fontSize: '0.78rem', lineHeight: '1.4' }}>
                                "{src.text}"
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))
            )}

            {loadingAnswer && (
              <div style={{ alignSelf: 'flex-start', background: 'var(--bg-card)', padding: '14px 18px', borderRadius: '18px', display: 'flex', alignItems: 'center', gap: '10px', border: '1px solid var(--border-color)' }}>
                <RefreshCw size={16} className="pulsing" color="#8b5cf6" />
                <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Running Hybrid Retrieval & Answer Synthesis...</span>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* Sticky Bottom Input Form */}
          <form 
            onSubmit={handleSendQuestion} 
            style={{ 
              padding: '16px 24px', 
              background: 'var(--bg-secondary)', 
              borderTop: '1px solid var(--border-color)', 
              display: 'flex', 
              gap: '12px',
              shrink: 0
            }}
          >
            <input
              type="text"
              placeholder="Ask a question about your uploaded documents..."
              value={inputQuestion}
              onChange={(e) => setInputQuestion(e.target.value)}
              disabled={loadingAnswer}
              style={{
                flex: 1,
                background: 'var(--bg-primary)',
                border: '1px solid var(--border-color)',
                borderRadius: '10px',
                padding: '12px 16px',
                color: '#ffffff',
                fontSize: '0.9rem',
                outline: 'none'
              }}
            />
            <button type="submit" className="btn-primary" disabled={loadingAnswer || !inputQuestion.trim()}>
              <Send size={16} /> Send
            </button>
          </form>
        </main>
      </div>

      {/* RAG Evaluation Modal */}
      {showEvalModal && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(4px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100 }}>
          <div className="glass-panel" style={{ width: '540px', padding: '24px', background: 'var(--bg-secondary)' }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <BarChart2 size={20} color="#8b5cf6" /> RAG Output Evaluator
            </h3>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginBottom: '16px' }}>
              <div>
                <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Question</label>
                <input
                  type="text"
                  value={evalData.question}
                  onChange={(e) => setEvalData({ ...evalData, question: e.target.value })}
                  placeholder="e.g. What is RAG?"
                  style={{ width: '100%', padding: '8px', background: 'var(--bg-primary)', border: '1px solid var(--border-color)', color: '#fff', borderRadius: '6px' }}
                />
              </div>

              <div>
                <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Generated Answer</label>
                <textarea
                  rows="3"
                  value={evalData.answer}
                  onChange={(e) => setEvalData({ ...evalData, answer: e.target.value })}
                  placeholder="Paste LLM generated answer string..."
                  style={{ width: '100%', padding: '8px', background: 'var(--bg-primary)', border: '1px solid var(--border-color)', color: '#fff', borderRadius: '6px' }}
                />
              </div>

              <div>
                <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Ground Truth (Optional for Recall)</label>
                <input
                  type="text"
                  value={evalData.groundTruth}
                  onChange={(e) => setEvalData({ ...evalData, groundTruth: e.target.value })}
                  placeholder="Expected ideal answer..."
                  style={{ width: '100%', padding: '8px', background: 'var(--bg-primary)', border: '1px solid var(--border-color)', color: '#fff', borderRadius: '6px' }}
                />
              </div>
            </div>

            {evalResult && (
              <div style={{ background: 'var(--bg-primary)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-color)', marginBottom: '16px' }}>
                <div style={{ fontWeight: 600, marginBottom: '8px', color: '#10b981' }}>Overall Quality Score: {(evalResult.overall_score * 100).toFixed(1)}%</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '0.8rem' }}>
                  <div>Faithfulness: {(evalResult.faithfulness * 100).toFixed(1)}%</div>
                  <div>Answer Relevance: {(evalResult.answer_relevance * 100).toFixed(1)}%</div>
                  <div>Context Relevance: {(evalResult.context_relevance * 100).toFixed(1)}%</div>
                  <div>Context Recall: {evalResult.context_recall}</div>
                </div>
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
              <button className="btn-secondary" onClick={() => setShowEvalModal(false)}>Close</button>
              <button className="btn-primary" onClick={handleRunEvaluation} disabled={evaluating}>
                {evaluating ? 'Evaluating...' : 'Compute Metrics'}
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
