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

const API_URL = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

console.log('Current API_URL:', API_URL);

const STYLES = {
  wordListContainer: {
    flex: 1,
    padding: '20px', 
    background: '#f9f9f9', 
    borderRadius: '12px', 
    border: '1px solid #eee',
    boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
    maxHeight: '600px',
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

// Helper for Simplex Projection
const simplexToPolygon = (coords) => {
  // Vertices of equilateral triangle
  const v = [
    [0, 0],
    [1, 0],
    [0.5, Math.sqrt(3)/2]
  ];
  
  // coords is [p1, p2, p3]
  // Result is weighted sum of vertices
  const x = coords[0]*v[0][0] + coords[1]*v[1][0] + coords[2]*v[2][0];
  const y = coords[0]*v[0][1] + coords[1]*v[1][1] + coords[2]*v[2][1];
  return [x, y];
};

const getRGBString = (coords) => {
    // Simple mapping of 3 components to RGB
    // coords are 0-1
    const r = Math.floor(coords[0] * 255);
    const g = Math.floor(coords[1] * 255);
    const b = Math.floor(coords[2] * 255);
    return `rgba(${r}, ${g}, ${b}, 0.8)`;
};

function App() {
  const [config, setConfig] = useState({
    batch_size: 128,
    sequence_len: 25,
    preset: 'mess3',
    x: 0.15,
    y: 0.7,
    a: 0.6
  });
  
  // Default 3 symbols, 3 states (Identity matrices)
  const [matrices, setMatrices] = useState(DEFAULT_MATRICES);

  const [selectedSymbol, setSelectedSymbol] = useState(0);
  const [words, setWords] = useState([]);
  const [beliefStates, setBeliefStates] = useState([]);
  const [constrainedBeliefs, setConstrainedBeliefs] = useState([]);
  const [flatBeliefs, setFlatBeliefs] = useState([]);
  const [flatConstrainedBeliefs, setFlatConstrainedBeliefs] = useState([]);
  const [initialState, setInitialState] = useState(null);
  const [hoveredBelief, setHoveredBelief] = useState(null);
  const [prevBelief, setPrevBelief] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('2d'); // '2d' or '3d'
  const [beliefMode, setBeliefMode] = useState('standard'); // 'standard' or 'constrained'

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

  // Auto-generate when config changes (debounced)
  useEffect(() => {
    const timer = setTimeout(() => {
      generateData();
    }, 500); // 500ms debounce

    return () => clearTimeout(timer);
  }, [config, matrices]); // Re-run when config or matrices change

  const generateData = async () => {
    setLoading(true);
    setError(null);
    try {
      // Normalize matrices (strengths -> probabilities)
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
      
      setWords(response.data.words);
      setBeliefStates(response.data.belief_states);
      setConstrainedBeliefs(response.data.constrained_beliefs);
      setInitialState(response.data.initial_state);
      setFlatBeliefs(response.data.flat_beliefs);
      setFlatConstrainedBeliefs(response.data.flat_constrained_beliefs);
      
    } catch (err) {
      console.error(err);
      const errorMsg = err.response?.data?.detail || err.message;
      setError(`Failed to fetch data from ${API_URL}: ${errorMsg}`);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    generateData();
  };

  const getPlotData = () => {
    const activeFlatBeliefs = beliefMode === 'constrained' ? flatConstrainedBeliefs : flatBeliefs;
    if (!activeFlatBeliefs || activeFlatBeliefs.length === 0) return null;

    const colors = activeFlatBeliefs.map(b => getRGBString(b));
    const data = [];
    
    // Use a constant uirevision to preserve camera state across updates
    const layout = {
        uirevision: 'true',
        // Push plot to bottom-left to make room for inset
        margin: { l: 0, r: 200, b: 0, t: 100 },
        height: 600,
        autosize: true,
        showlegend: false
    };

    // Main scatter plot
    if (viewMode === '3d') {
      data.push({
        type: 'scatter3d',
        mode: 'markers',
        x: activeFlatBeliefs.map(b => b[0]),
        y: activeFlatBeliefs.map(b => b[1]),
        z: activeFlatBeliefs.map(b => b[2]),
        marker: {
          size: 3,
          color: colors,
          opacity: 0.8
        },
        name: 'Belief States'
      });

      layout.scene = {
        xaxis: { title: 'A', range: [0, 1], autorange: false },
        yaxis: { title: 'B', range: [0, 1], autorange: false },
        zaxis: { title: 'C', range: [0, 1], autorange: false },
        aspectmode: 'cube'
      };

      // Highlight vector
      if (prevBelief && hoveredBelief) {
        // Line
        data.push({
            type: 'scatter3d',
            mode: 'lines',
            x: [prevBelief[0], hoveredBelief[0]],
            y: [prevBelief[1], hoveredBelief[1]],
            z: [prevBelief[2], hoveredBelief[2]],
            line: { color: 'black', width: 5 },
            name: 'Transition'
        });
        
        // Cone for arrow head
        // Only add if there is movement
        const dx = hoveredBelief[0] - prevBelief[0];
        const dy = hoveredBelief[1] - prevBelief[1];
        const dz = hoveredBelief[2] - prevBelief[2];
        const len = Math.sqrt(dx*dx + dy*dy + dz*dz);
        
        if (len > 0.001) {
            data.push({
                type: 'cone',
                x: [hoveredBelief[0]],
                y: [hoveredBelief[1]],
                z: [hoveredBelief[2]],
                u: [dx],
                v: [dy],
                w: [dz],
                sizemode: 'absolute',
                sizeref: 0.1,
                anchor: 'tip',
                colorscale: [[0, 'black'], [1, 'black']],
                showscale: false
            });
        }
      }

      return { data, layout };
    } else {
      // 2D Simplex (Triangle)
      const points = activeFlatBeliefs.map(b => simplexToPolygon(b));
      
      // Triangle boundary
      const v = [[0, 0], [1, 0], [0.5, Math.sqrt(3)/2], [0,0]];
      
      data.push({
        type: 'scatter',
        mode: 'markers',
        x: points.map(p => p[0]),
        y: points.map(p => p[1]),
        marker: {
          size: 4,
          color: colors,
        },
        showlegend: false
      });

      // Boundary
      data.push({
        type: 'scatter',
        mode: 'lines',
        x: v.map(p => p[0]),
        y: v.map(p => p[1]),
        line: { color: 'black', width: 1 },
        hoverinfo: 'skip',
        showlegend: false
      });

      layout.xaxis = { showgrid: false, zeroline: false, showticklabels: false };
      layout.yaxis = { showgrid: false, zeroline: false, showticklabels: false, scaleanchor: 'x' };
      // Push 2D plot to bottom-left
      layout.margin = { l: 20, r: 300, b: 20, t: 100 };

      // Highlight vector
      if (prevBelief && hoveredBelief) {
        const p1 = simplexToPolygon(prevBelief);
        const p2 = simplexToPolygon(hoveredBelief);
        
        // Use annotation for arrow
        layout.annotations = [{
            x: p2[0],
            y: p2[1],
            ax: p1[0],
            ay: p1[1],
            xref: 'x',
            yref: 'y',
            axref: 'x',
            ayref: 'y',
            showarrow: true,
            arrowhead: 2,
            arrowsize: 1,
            arrowwidth: 2,
            arrowcolor: 'black'
        }];
      }

      return { data, layout };
    }
  };

  const plotData = getPlotData();

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif', maxWidth: '1600px', margin: '0 auto' }}>
      <h1>Simplex Generator</h1>
      
      <div style={{ display: 'flex', gap: '20px', alignItems: 'stretch', marginBottom: '30px' }}>
        
        {/* Left Column: Plot */}
        <div style={{ flex: 1, minWidth: '800px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div style={{ 
                background: 'white', 
                padding: '10px', 
                borderRadius: '12px', 
                boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
                position: 'relative'
            }}>
                <div style={{ marginBottom: '10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h3 style={{ margin: 0 }}>
                        {beliefMode === 'constrained' ? 'Constrained Belief Space' : 'Standard Belief Space'}
                    </h3>
                    <div style={{ display: 'flex', gap: '15px' }}>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '5px', cursor: 'pointer' }}>
                            <input 
                                type="checkbox" 
                                checked={beliefMode === 'constrained'} 
                                onChange={(e) => setBeliefMode(e.target.checked ? 'constrained' : 'standard')} 
                            />
                            Constrained Mode
                        </label>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '5px', cursor: 'pointer' }}>
                            <input 
                                type="checkbox" 
                                checked={viewMode === '2d'} 
                                onChange={(e) => setViewMode(e.target.checked ? '2d' : '3d')} 
                            />
                            Show 2D Simplex
                        </label>
                    </div>
                </div>
                
                {plotData ? (
                    <Plot
                        data={plotData.data}
                        layout={plotData.layout}
                        style={{ width: '100%', height: '600px' }}
                        config={{ displayModeBar: false }}
                    />
                ) : (
                    <div style={{ 
                        height: '600px', 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center', 
                        background: '#f9f9f9', 
                        color: '#999',
                        borderRadius: '8px'
                    }}>
                        Generate data to see visualization
                    </div>
                )}

                {/* Belief Visualizer (Inset top right) */}
                <div style={{
                    position: 'absolute',
                    top: '90px',
                    right: '20px',
                    width: '380px',
                    height: '330px',
                    zIndex: 10
                }}>
                    <BeliefVisualizer 
                        probabilities={hoveredBelief || initialState || [0.33, 0.33, 0.33]} 
                        minimal={true}
                        showButton={true}
                    />
                </div>
            </div>

            {/* Generated Words (Moved here) */}
            {words.length > 0 && (
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
                                            const activeBeliefs = beliefMode === 'constrained' ? constrainedBeliefs : beliefStates;
                                            
                                            if (activeBeliefs[idx] && activeBeliefs[idx][charIdx]) {
                                                setHoveredBelief(activeBeliefs[idx][charIdx]);
                                                
                                                // Determine previous belief
                                                if (charIdx === 0) {
                                                    setPrevBelief(initialState);
                                                } else {
                                                    setPrevBelief(activeBeliefs[idx][charIdx - 1]);
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
      )}
        </div>

        {/* Right Column: Matrix Editor */}
        <div style={{ flex: 1, minWidth: '500px' }}>
             <MatrixEditor 
                matrices={matrices} 
                onChange={setMatrices} 
                config={config}
                onConfigChange={handleChange}
                selectedSymbol={selectedSymbol}
                onSymbolChange={setSelectedSymbol}
                prevBelief={prevBelief}
                nextBelief={hoveredBelief}
                beliefMode={beliefMode}
            />
        </div>
      </div>

      {/* Controls Row */}
      <form onSubmit={handleSubmit} style={{ marginBottom: '30px' }}>
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

      {error && <div style={{ color: 'red', marginBottom: '20px' }}>{error}</div>}

    </div>
  );
}

export default App;
