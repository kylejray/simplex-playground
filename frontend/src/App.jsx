import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';
import MatrixEditor from './MatrixEditor';
import BeliefVisualizer from './BeliefVisualizer';

const DEFAULT_MATRICES = [
  [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
];

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const STYLES = {
  wordListContainer: {
    flex: 1,
    padding: '20px', 
    background: '#f9f9f9', 
    borderRadius: '12px', 
    border: '1px solid #eee',
    boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
    height: '430px',
    overflow: 'auto'
  },
  wordItem: {
    marginBottom: '12px', 
    fontFamily: 'monospace', 
    fontSize: '20px', 
    color: '#444', 
    letterSpacing: '3px', 
    whiteSpace: 'nowrap'
  },
  charSpan: {
    cursor: 'crosshair',
    padding: '2px 4px',
    borderRadius: '4px',
    transition: 'all 0.1s',
    borderBottom: '2px solid transparent'
  }
};

function App() {
  const [config, setConfig] = useState({
    batch_size: 512,
    sequence_len: 50,
    preset: 'mess3',
    x: 0.1,
    y: 0.7,
    a: 0.7
  });
  
  // Default 3 symbols, 3 states (Identity matrices)
  const [matrices, setMatrices] = useState(DEFAULT_MATRICES);

  const [selectedSymbol, setSelectedSymbol] = useState(0);
  const [plotData, setPlotData] = useState(null);
  const [words, setWords] = useState([]);
  const [beliefStates, setBeliefStates] = useState([]);
  const [initialState, setInitialState] = useState(null);
  const [hoveredBelief, setHoveredBelief] = useState(null);
  const [prevBelief, setPrevBelief] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch preset matrices when preset or params change
  useEffect(() => {
    if (config.preset === 'custom') return;

    const fetchPreset = async () => {
      try {
        const payload = {
          key: config.preset,
          kwargs: {}
        };
        if (config.preset === 'simple_constrained') {
          payload.kwargs = { x: parseFloat(config.x), y: parseFloat(config.y) };
        } else if (config.preset === 'mess3') {
          payload.kwargs = { x: parseFloat(config.x), a: parseFloat(config.a) };
        } else if (config.preset === 'left_right_mix') {
          payload.kwargs = {};
        }
        
        const response = await axios.post(`${API_URL}/get_preset`, payload);
        setMatrices(response.data);
      } catch (err) {
        console.error("Error fetching preset:", err);
        // Don't block UI, just log error
      }
    };

    fetchPreset();
  }, [config.preset, config.x, config.y, config.a]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setConfig(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      // Normalize matrices (strengths -> probabilities)
      // Normalization must happen across all symbols for a given state.
      // Sum of outgoing transitions from state i (across all symbols) must be 1.
      
      const numSymbols = matrices.length;
      const numStates = matrices[0].length;
      
      // Calculate total strength for each state
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

      const normalizedMatrices = matrices.map((matrix, k) => 
        matrix.map((row, i) => {
          const sum = stateSums[i];
          return row.map(val => sum === 0 ? 0 : (parseFloat(val) || 0) / sum);
        })
      );

      const payload = {
        batch_size: parseInt(config.batch_size),
        sequence_len: parseInt(config.sequence_len),
        matrices: normalizedMatrices
      };

      const response = await axios.post(`${API_URL}/generate`, payload);
      setPlotData(response.data.plot);
      setWords(response.data.words);
      setBeliefStates(response.data.belief_states);
      setInitialState(response.data.initial_state);
    } catch (err) {
      console.error(err);
      setError('Failed to fetch data: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>Simplex Generator</h1>
      <form onSubmit={handleSubmit} style={{ marginBottom: '20px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        
        <MatrixEditor 
            matrices={matrices} 
            onChange={setMatrices} 
            config={config}
            onConfigChange={handleChange}
            selectedSymbol={selectedSymbol}
            onSymbolChange={setSelectedSymbol}
            prevBelief={prevBelief}
        />

        <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '20px', 
            background: '#fff', 
            padding: '20px', 
            borderRadius: '12px',
            boxShadow: '0 2px 10px rgba(0,0,0,0.05)',
            flexWrap: 'wrap'
        }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 'bold', color: '#555' }}>Batch Size: </label>
              <input 
                type="number" 
                name="batch_size" 
                value={config.batch_size} 
                onChange={handleChange} 
                style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ccc', width: '100px' }}
              />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 'bold', color: '#555' }}>Sequence Length: </label>
              <input 
                type="number" 
                name="sequence_len" 
                value={config.sequence_len} 
                onChange={handleChange} 
                style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ccc', width: '100px' }}
              />
            </div>
            
            <div style={{ flex: 1 }}></div>

            <button 
                type="submit" 
                disabled={loading} 
                style={{ 
                    padding: '10px 30px', 
                    cursor: loading ? 'not-allowed' : 'pointer', 
                    background: loading ? '#ccc' : '#007bff',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    fontSize: '16px',
                    fontWeight: 'bold',
                    transition: 'background 0.2s'
                }}
            >
              {loading ? 'Generating...' : 'Generate Simulation'}
            </button>
        </div>
      </form>

      {error && <div style={{ color: 'red', marginBottom: '10px' }}>{error}</div>}

      {plotData && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {/* Top Row: Visualizer and Words */}
            <div style={{ display: 'flex', gap: '20px', alignItems: 'flex-start' }}>
                {/* Belief Visualizer (Left) */}
                <div style={{ flex: '0 0 auto' }}>
                    <BeliefVisualizer probabilities={hoveredBelief} />
                </div>

                {/* Words Display (Right) */}
                <div style={STYLES.wordListContainer}>
                    <h3 style={{ marginTop: 0, fontSize: '18px', color: '#333', marginBottom: '10px' }}>Generated Words</h3>
                    <div style={{ fontSize: '14px', color: '#666', marginBottom: '15px', fontStyle: 'italic' }}>
                        Hover over characters to see belief state
                    </div>
                    <ul style={{ paddingLeft: '0', margin: 0, listStyle: 'none' }}>
                        {words.map((fullWord, idx) => (
                            <li key={idx} style={STYLES.wordItem}>
                                {fullWord.split('').map((char, charIdx) => (
                                    <span 
                                        key={charIdx}
                                        onMouseEnter={() => {
                                            if (beliefStates[idx] && beliefStates[idx][charIdx]) {
                                                setHoveredBelief(beliefStates[idx][charIdx]);
                                                
                                                // Determine previous belief
                                                if (charIdx === 0) {
                                                    setPrevBelief(initialState);
                                                } else {
                                                    setPrevBelief(beliefStates[idx][charIdx - 1]);
                                                }
                                            }
                                            // Update selected symbol in Matrix Editor
                                            const symbolIdx = parseInt(char);
                                            if (!isNaN(symbolIdx) && symbolIdx >= 0 && symbolIdx < matrices.length) {
                                                setSelectedSymbol(symbolIdx);
                                            }
                                        }}
                                        onMouseLeave={() => {
                                            setHoveredBelief(null);
                                            setPrevBelief(null);
                                        }}
                                        style={STYLES.charSpan}
                                        className="word-char"
                                    >
                                        {char}
                                    </span>
                                ))}
                            </li>
                        ))}
                    </ul>
                </div>
            </div>

            {/* Bottom Row: Plot */}
            <div style={{ border: '1px solid #ccc', padding: '10px', width: '100%' }}>
              <Plot
                data={plotData.data}
                layout={plotData.layout}
                style={{ width: '100%', height: '600px' }}
                useResizeHandler={true}
              />
            </div>
        </div>
      )}
    </div>
  );
}

export default App;
