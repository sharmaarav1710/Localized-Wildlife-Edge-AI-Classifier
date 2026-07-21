import { useState } from 'react';

// Grab the backend URL from Vercel environment variables or fallback to local
const API_URL = import.meta.env.VITE_API_URL || 'https://localized-wildlife-edge-ai-classifier.onrender.com';

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to connect to backend.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '600px', margin: '40px auto', padding: '20px', fontFamily: 'sans-serif' }}>
      <h1>🌿 Wildlife Edge AI Classifier</h1>
      <p>Upload an image to identify the species using our backend model.</p>

      <div style={{ margin: '20px 0' }}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
      </div>

      {preview && (
        <div style={{ margin: '20px 0' }}>
          <img src={preview} alt="Selected sample" style={{ maxWidth: '100%', maxHeight: '300px', borderRadius: '8px' }} />
        </div>
      )}

      {selectedFile && (
        <button 
          onClick={handleUpload} 
          disabled={loading}
          style={{ padding: '10px 20px', fontSize: '16px', cursor: 'pointer' }}
        >
          {loading ? 'Analyzing image...' : 'Classify Image'}
        </button>
      )}

      {error && (
        <div style={{ color: 'red', marginTop: '20px' }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: '20px', padding: '15px', border: '1px solid #ccc', borderRadius: '8px' }}>
          <h3>Result:</h3>
          <p><strong>Prediction:</strong> {result.prediction}</p>
          <p><strong>Confidence:</strong> {result.confidence ? `${(result.confidence * 100).toFixed(2)}%` : 'N/A'}</p>
        </div>
      )}
    </div>
  );
}