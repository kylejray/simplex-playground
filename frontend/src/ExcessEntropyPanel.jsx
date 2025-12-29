import React, { useState, useRef } from 'react';
import Plot from 'react-plotly.js';

const API_URL = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

const PANEL_STYLES = {
  container: {
    background: 'white',
    padding: '20px',
    borderRadius: '12px',
    boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
    height: '100%',
    display: 'flex',
    flexDirection: 'column'
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '20px'
  },
  controls: {
    display: 'flex',
    gap: '20px',
    alignItems: 'center',
    flexWrap: 'wrap',
    marginBottom: '20px',
    padding: '15px',
    background: '#f9f9f9',
    borderRadius: '8px'
  },
  controlGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '5px'
  },
  label: {
    fontSize: '12px',
    fontWeight: 'bold',
    color: '#666',
    textTransform: 'uppercase'
  },
  slider: {
    width: '150px',
    cursor: 'pointer'
  },
  input: {
    padding: '8px',
    borderRadius: '4px',
    border: '1px solid #ccc',
    width: '70px'
  },
  button: {
    padding: '12px 24px',
    cursor: 'pointer',
    background: '#28a745',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    fontSize: '14px',
    fontWeight: 'bold',
    transition: 'all 0.2s',
    display: 'flex',
    alignItems: 'center',
    gap: '8px'
  },
  buttonDisabled: {
    background: '#ccc',
    cursor: 'not-allowed'
  },
  resultCard: {
    display: 'flex',
    gap: '30px',
    marginBottom: '20px',
    padding: '20px',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    borderRadius: '12px',
    color: 'white',
    flexWrap: 'wrap'
  },
  resultItem: {
    textAlign: 'center'
  },
  resultValue: {
    fontSize: '28px',
    fontWeight: 'bold',
    fontFamily: 'monospace'
  },
  resultLabel: {
    fontSize: '11px',
    opacity: 0.9,
    textTransform: 'uppercase',
    marginTop: '5px'
  },
  plotContainer: {
    flex: 1,
    minHeight: '400px'
  },
  placeholder: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    background: '#f9f9f9',
    borderRadius: '8px',
    color: '#999',
    gap: '15px'
  },
  spinner: {
    width: '24px',
    height: '24px',
    border: '3px solid #ffffff40',
    borderTop: '3px solid #ffffff',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite'
  },
  error: {
    color: '#dc3545',
    padding: '10px',
    background: '#f8d7da',
    borderRadius: '6px',
    marginBottom: '15px'
  },
  progressContainer: {
    background: '#e8f4fd',
    border: '1px solid #b8daff',
    borderRadius: '8px',
    padding: '15px',
    marginBottom: '15px'
  },
  progressHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '10px'
  },
  progressTitle: {
    fontWeight: 'bold',
    color: '#004085',
    display: 'flex',
    alignItems: 'center',
    gap: '10px'
  },
  progressLog: {
    fontFamily: 'monospace',
    fontSize: '13px',
    color: '#333',
    maxHeight: '120px',
    overflowY: 'auto',
    background: 'white',
    padding: '10px',
    borderRadius: '4px',
    border: '1px solid #ddd'
  },
  progressLine: {
    margin: '2px 0',
    display: 'flex',
    gap: '10px'
  }
};

const spinnerKeyframes = `
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

function ExcessEntropyPanel({ matrices, result, setResult }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [maxTimePerL, setMaxTimePerL] = useState(8);  // seconds
  const [mcSamples, setMcSamples] = useState(10000);
  const [maxBlockLength, setMaxBlockLength] = useState(20);
  const [progressLog, setProgressLog] = useState([]);
  const [currentL, setCurrentL] = useState(null);
  const abortControllerRef = useRef(null);

  const computeExcessEntropy = async () => {
    setLoading(true);
    setError(null);
    setProgressLog([]);
    setCurrentL(0);

    try {
      // Normalize matrices before sending
      const numSymbols = matrices.length;
      const numStates = matrices[0].length;

      const stateSums = new Array(numStates).fill(0);
      for (let i = 0; i < numStates; i++) {
        let sum = 0;
        for (let k = 0; k < numSymbols; k++) {
          for (let j = 0; j < numStates; j++) {
            sum += (parseFloat(matrices[k][i][j]) || 0);
          }
        }
        stateSums[i] = sum;
      }

      const normalizedMatrices = matrices.map((matrix) =>
        matrix.map((row, i) => {
          const sum = stateSums[i];
          return row.map(val => sum === 0 ? 0 : (parseFloat(val) || 0) / sum);
        })
      );

      // Create abort controller for cancellation
      abortControllerRef.current = new AbortController();

      // Use fetch with SSE
      const response = await fetch(`${API_URL}/excess_entropy`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          matrices: normalizedMatrices,
          max_block_length: maxBlockLength,
          max_time_per_L: maxTimePerL,
          mc_samples: mcSamples
        }),
        signal: abortControllerRef.current.signal
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        
        // Process complete SSE messages
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || ''; // Keep incomplete message in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.substring(6));
              
              if (data.type === 'progress') {
                setCurrentL(data.L);
                setProgressLog(prev => [...prev, {
                  L: data.L,
                  H: data.H,
                  method: data.method,
                  time: data.time
                }]);
              } else if (data.type === 'result') {
                setResult(data.data);
              } else if (data.type === 'error') {
                setError(`Computation failed: ${data.message}`);
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e);
            }
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error(err);
        setError(`Computation failed: ${err.message}`);
      }
    } finally {
      setLoading(false);
      setCurrentL(null);
    }
  };

  const cancelComputation = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  const getPlotData = () => {
    if (!result || !result.block_entropies) return null;

    // Block entropies now include timing: [L, H, method, time]
    const exactData = result.block_entropies.filter(be => be[2] === 'exact');
    const mcData = result.block_entropies.filter(be => be[2] === 'mc');
    
    const allLs = result.block_entropies.map(be => be[0]);
    
    // Compute fitted line: H(L) = h_mu * L + E
    const h_mu = result.entropy_rate;
    const E = result.excess_entropy;
    const fittedHs = allLs.map(L => h_mu * L + E);

    const data = [];
    
    // Exact enumeration points (solid markers)
    if (exactData.length > 0) {
      data.push({
        type: 'scatter',
        mode: 'markers+lines',
        x: exactData.map(be => be[0]),
        y: exactData.map(be => be[1]),
        name: 'Block Entropy (Exact)',
        marker: { size: 8, color: '#667eea', symbol: 'circle' },
        line: { color: '#667eea', width: 2 },
        hovertemplate: 'L=%{x}<br>H=%{y:.4f}<extra>Exact</extra>'
      });
    }
    
    // Monte Carlo points (hollow markers)
    if (mcData.length > 0) {
      data.push({
        type: 'scatter',
        mode: 'markers+lines',
        x: mcData.map(be => be[0]),
        y: mcData.map(be => be[1]),
        name: 'Block Entropy (Monte Carlo)',
        marker: { size: 8, color: '#667eea', symbol: 'circle-open', line: { width: 2, color: '#667eea' } },
        line: { color: '#667eea', width: 2, dash: 'dot' },
        hovertemplate: 'L=%{x}<br>H=%{y:.4f}<extra>Monte Carlo</extra>'
      });
    }
    
    // Fitted asymptote
    data.push({
      type: 'scatter',
      mode: 'lines',
      x: allLs,
      y: fittedHs,
      name: `Asymptote: ${h_mu.toFixed(4)}·L + ${E.toFixed(4)}`,
      line: { color: '#ff6b6b', width: 2, dash: 'dash' }
    });
    
    // Add vertical line at MC start if applicable
    const shapes = [];
    if (result.mc_start_L) {
      shapes.push({
        type: 'line',
        x0: result.mc_start_L - 0.5,
        x1: result.mc_start_L - 0.5,
        y0: 0,
        y1: Math.max(...result.block_entropies.map(be => be[1])),
        line: { color: '#aaa', width: 1, dash: 'dashdot' }
      });
    }

    return {
      data,
      layout: {
        title: 'Block Entropy Convergence',
        shapes,
        xaxis: {
          title: 'Block Length L',
          gridcolor: '#eee'
        },
        yaxis: {
          title: 'H(X₀ᴸ) [bits]',
          gridcolor: '#eee'
        },
        legend: {
          x: 0.02,
          y: 0.98,
          bgcolor: 'rgba(255,255,255,0.9)'
        },
        margin: { l: 60, r: 30, t: 50, b: 50 },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white'
      }
    };
  };

  const plotData = getPlotData();

  return (
    <div style={PANEL_STYLES.container}>
      <style>{spinnerKeyframes}</style>

      <div style={PANEL_STYLES.header}>
        <h3 style={{ margin: 0 }}>Excess Entropy Estimation</h3>
      </div>

      {/* Controls */}
      <div style={PANEL_STYLES.controls}>
        <div style={PANEL_STYLES.controlGroup}>
          <label style={PANEL_STYLES.label}>
            Max Wait Time: {maxTimePerL}s
          </label>
          <input
            type="range"
            min="1"
            max="30"
            step="1"
            value={maxTimePerL}
            onChange={(e) => setMaxTimePerL(parseInt(e.target.value))}
            style={PANEL_STYLES.slider}
            disabled={loading}
          />
        </div>

        <div style={PANEL_STYLES.controlGroup}>
          <label style={PANEL_STYLES.label}>
            MC Samples: {mcSamples.toLocaleString()}
          </label>
          <input
            type="range"
            min="1000"
            max="100000"
            step="1000"
            value={mcSamples}
            onChange={(e) => setMcSamples(parseInt(e.target.value))}
            style={PANEL_STYLES.slider}
            disabled={loading}
          />
        </div>

        <div style={PANEL_STYLES.controlGroup}>
          <label style={PANEL_STYLES.label}>Max Block Length</label>
          <input
            type="number"
            min="5"
            max="50"
            value={maxBlockLength}
            onChange={(e) => setMaxBlockLength(parseInt(e.target.value) || 20)}
            style={PANEL_STYLES.input}
            disabled={loading}
          />
        </div>

        <div style={{ flex: 1 }}></div>

        {loading ? (
          <button
            onClick={cancelComputation}
            style={{
              ...PANEL_STYLES.button,
              background: '#dc3545'
            }}
          >
            Cancel
          </button>
        ) : (
          <button
            onClick={computeExcessEntropy}
            style={PANEL_STYLES.button}
          >
            Compute Excess Entropy
          </button>
        )}
      </div>

      {error && <div style={PANEL_STYLES.error}>{error}</div>}

      {/* Progress Display */}
      {loading && (
        <div style={PANEL_STYLES.progressContainer}>
          <div style={PANEL_STYLES.progressHeader}>
            <div style={PANEL_STYLES.progressTitle}>
              <div style={{...PANEL_STYLES.spinner, borderColor: '#004085', borderTopColor: '#004085'}}></div>
              Computing L = {currentL || 1} of {maxBlockLength}...
            </div>
            <div style={{ color: '#004085', fontSize: '13px' }}>
              {progressLog.length > 0 && `${progressLog.length} completed`}
            </div>
          </div>
          {progressLog.length > 0 && (
            <div style={PANEL_STYLES.progressLog}>
              {progressLog.map((p, idx) => (
                <div key={idx} style={PANEL_STYLES.progressLine}>
                  <span style={{ color: '#666', minWidth: '50px' }}>L={p.L}</span>
                  <span style={{ color: '#333', minWidth: '100px' }}>H={p.H.toFixed(4)}</span>
                  <span style={{ 
                    color: p.method === 'exact' ? '#28a745' : '#fd7e14',
                    minWidth: '60px'
                  }}>
                    {p.method}
                  </span>
                  <span style={{ color: '#666' }}>{p.time.toFixed(2)}s</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Results Card */}
      {result && !loading && (
        <div style={PANEL_STYLES.resultCard}>
          <div style={PANEL_STYLES.resultItem}>
            <div style={PANEL_STYLES.resultValue}>
              {result.excess_entropy.toFixed(4)}
            </div>
            <div style={PANEL_STYLES.resultLabel}>Excess Entropy E (bits)</div>
          </div>
          <div style={PANEL_STYLES.resultItem}>
            <div style={PANEL_STYLES.resultValue}>
              {result.entropy_rate.toFixed(4)}
            </div>
            <div style={PANEL_STYLES.resultLabel}>Entropy Rate hμ (bits/symbol)</div>
          </div>
          <div style={PANEL_STYLES.resultItem}>
            <div style={PANEL_STYLES.resultValue}>
              {result.mc_start_L ? `L≥${result.mc_start_L}` : 'All Exact'}
            </div>
            <div style={PANEL_STYLES.resultLabel}>{result.mc_start_L ? 'MC Started At' : 'Method'}</div>
          </div>
          <div style={PANEL_STYLES.resultItem}>
            <div style={PANEL_STYLES.resultValue}>
              {result.computation_time?.toFixed(1) || '—'}s
            </div>
            <div style={PANEL_STYLES.resultLabel}>Total Time</div>
          </div>
        </div>
      )}

      {/* Plot or Placeholder */}
      {plotData && !loading ? (
        <div style={PANEL_STYLES.plotContainer}>
          <Plot
            data={plotData.data}
            layout={plotData.layout}
            style={{ width: '100%', height: '100%' }}
            config={{ displayModeBar: false }}
          />
        </div>
      ) : !loading && (
        <div style={PANEL_STYLES.placeholder}>
          <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M3 3v18h18" />
            <path d="M7 16l4-4 4 4 5-6" />
          </svg>
          <div style={{ fontSize: '16px', fontWeight: 'bold' }}>No Results Yet</div>
          <div style={{ fontSize: '14px', maxWidth: '400px', textAlign: 'center' }}>
            Configure the parameters above and click "Compute Excess Entropy".
            <br/><br/>
            <strong>Max Wait Time:</strong> If computing H(L) takes longer than this, 
            switches to Monte Carlo sampling for all subsequent L values.
          </div>
        </div>
      )}
    </div>
  );
}

export default ExcessEntropyPanel;
