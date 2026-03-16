import React, { useState, useEffect } from 'react';

const GRAPH_WIDTH = 450;
const GRAPH_HEIGHT = 400;
const NODE_RADIUS = 35;
const NODE_MARGIN = 60;

const COLORS = {
  nodeFill: 'white',
  nodeStroke: '#333',
  text: '#333',
  highlightText: '#2e7d32',
  highlightBg: 'rgba(232, 245, 233, 0.9)',
  highlightBorder: '#c8e6c9'
};

const NormalizedInput = ({ rawValue, totalSum, onChange, style, className }) => {
    const [localValue, setLocalValue] = useState(null);
    
    // Calculate normalized value (probability)
    const normValue = totalSum === 0 ? 0 : rawValue / totalSum;
    
    // Use local value if editing, otherwise formatted normalized value
    const displayValue = localValue !== null ? localValue : (Math.round(normValue * 100) / 100).toString();

    const commit = (val) => {
        const newP = parseFloat(val);
        if (isNaN(newP)) {
            setLocalValue(null);
            return;
        }
        
        // Back-calculate raw strength: x = p * (S_others) / (1 - p)
        const s_others = totalSum - rawValue;
        let newRaw;
        
        if (s_others <= 0.00001) {
             // If no other weights, we can't really set a probability < 1 unless we set raw=0
             newRaw = newP > 0 ? 1 : 0; 
        } else {
             // Cap probability to avoid division by zero or negative weights
             const p = Math.min(Math.max(newP, 0), 0.99);
             newRaw = (p * s_others) / (1 - p);
        }
        
        onChange(newRaw.toString());
        setLocalValue(null);
    };

    return (
        <input
            type="text"
            inputMode="decimal"
            style={style}
            className={className}
            value={displayValue}
            onChange={(e) => setLocalValue(e.target.value)}
            onBlur={(e) => commit(e.target.value)}
            onKeyDown={(e) => {
                if (e.key === 'Enter') {
                    commit(e.target.value);
                    e.target.blur();
                }
            }}
            autoComplete="off"
        />
    );
};

const MatrixEditor = ({ matrices, onChange, config, onConfigChange, selectedSymbol, onSymbolChange, prevBelief, nextBelief, beliefMode }) => {
  
  const handleCellChange = (row, col, value) => {
    const newMatrices = JSON.parse(JSON.stringify(matrices));
    
    // Validate input: allow only numbers and one decimal point
    if (value === '' || /^\d*\.?\d*$/.test(value)) {
        newMatrices[selectedSymbol][row][col] = value;
        onChange(newMatrices);
    }
  };

  const handleConfigChange = (e) => {
    const { name, value } = e.target;
    onConfigChange({
      target: {
        name,
        value
      }
    });
  };

  // Calculate normalized matrix for display
  // Normalization across all symbols
  const numStates = matrices[0].length;
  const stateSums = new Array(numStates).fill(0);
  
  matrices.forEach(matrix => {
    matrix.forEach((row, i) => {
        row.forEach(val => {
            stateSums[i] += (parseFloat(val) || 0);
        });
    });
  });

  const allNormalizedMatrices = matrices.map(matrix => 
    matrix.map((row, i) => {
      const sum = stateSums[i];
      return row.map(val => sum === 0 ? 0 : (parseFloat(val) || 0) / sum);
    })
  );

  // Dynamic state labels based on number of states
  const stateLabels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'];
  const states = stateLabels.slice(0, numStates);
  const is3State = numStates === 3;
  
  // Positions only used for 3-state HMM graph
  const positions = {
    0: { x: NODE_MARGIN, y: GRAPH_HEIGHT - NODE_MARGIN },      // A (Bottom Left)
    1: { x: GRAPH_WIDTH - NODE_MARGIN, y: GRAPH_HEIGHT - NODE_MARGIN }, // B (Bottom Right)
    2: { x: GRAPH_WIDTH / 2, y: NODE_MARGIN }           // C (Top)
  };

  // Calculate total flow for the current symbol (likelihood of observation)
  let totalFlow = 0;
  if (prevBelief) {
      matrices[selectedSymbol].forEach((row, r) => {
          row.forEach((val, c) => {
              const weight = parseFloat(val) || 0;
              totalFlow += (prevBelief[r] || 0) * weight;
          });
      });
  }

  const renderArrow = (from, to, value) => {
    // Value here is strength
    const numValue = parseFloat(value) || 0;
    
    // Calculate flow if prevBelief is available
    let displayFlow = 0;
    let isHighlighted = false;
    let opacity = 1;
    let strokeWidth = 2;
    let strokeColor = COLORS.nodeStroke;

    if (prevBelief && totalFlow > 0) {
        // Flow = P(State_t = from) * Weight(from->to)
        const rawFlow = prevBelief[from] * numValue;
        
        // Normalize by total flow (Z) to get the contribution to the next belief
        // Label = (Belief * Weight) / Z
        // This ensures that Sum(Incoming Arrows to State J) = NextBelief(J)
        displayFlow = rawFlow / totalFlow;
        
        if (displayFlow === 0) {
            opacity = 0;
        } else {
            // Always show flow, scaling visual weight
            isHighlighted = true;
            strokeWidth = 2 + displayFlow * 8; 
            // Minimum opacity 0.3 so even small flows are visible
            strokeColor = `rgba(0, 0, 0, ${0.3 + displayFlow * 0.7})`;
            opacity = 1;
        }
    }

    const start = positions[from];
    const end = positions[to];
    const isSelf = from === to;
    
    let path;
    let labelX, labelY;

    // Helper to normalize vector and get point at distance
    const getPointAtDistance = (p1, p2, dist) => {
      const dx = p2.x - p1.x;
      const dy = p2.y - p1.y;
      const len = Math.sqrt(dx*dx + dy*dy);
      if (len === 0) return p1;
      return {
        x: p1.x + (dx / len) * dist,
        y: p1.y + (dy / len) * dist
      };
    };

    if (isSelf) {
      // Self loop
      // Direction depends on node position to push loop outwards
      let dx = 0, dy = -1; 
      if (from === 0) { dx = -1; dy = 1; } // Down-Left for A
      if (from === 1) { dx = 1; dy = 1; }  // Down-Right for B
      if (from === 2) { dx = 0; dy = -1; } // Up for C

      // Perpendicular vector for spread to make the loop round
      // If v = (dx, dy), p = (-dy, dx)
      const pdx = -dy;
      const pdy = dx;
      
      const spread = 50;
      const distance = 100;

      const cp1x = start.x + dx * distance + pdx * spread;
      const cp1y = start.y + dy * distance + pdy * spread;
      const cp2x = start.x + dx * distance - pdx * spread;
      const cp2y = start.y + dy * distance - pdy * spread;
      
      const cp1 = { x: cp1x, y: cp1y };
      const cp2 = { x: cp2x, y: cp2y };

      // Calculate start and end points on the circle boundary
      const startPoint = getPointAtDistance(start, cp1, NODE_RADIUS);
      const endPoint = getPointAtDistance(start, cp2, NODE_RADIUS);
      
      path = `M ${startPoint.x} ${startPoint.y} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${endPoint.x} ${endPoint.y}`;

      labelX = start.x + dx * 85;
      labelY = start.y + dy * 85;

    } else {
      // Curved line
      const dx = end.x - start.x;
      const dy = end.y - start.y;
      const midX = (start.x + end.x) / 2;
      const midY = (start.y + end.y) / 2;
      
      // Offset perpendicular to the line
      const offsetX = -dy * 0.15; 
      const offsetY = dx * 0.15;
      
      const ctrlX = midX + offsetX;
      const ctrlY = midY + offsetY;
      const ctrl = { x: ctrlX, y: ctrlY };
      
      // Calculate start and end points on the circle boundary
      // Start point moves towards control point
      const startPoint = getPointAtDistance(start, ctrl, NODE_RADIUS);
      // End point moves from end towards control point (backwards)
      // Actually we want the point on the circle around 'end' that is towards 'ctrl'
      const endPoint = getPointAtDistance(end, ctrl, NODE_RADIUS);
      
      path = `M ${startPoint.x} ${startPoint.y} Q ${ctrlX} ${ctrlY} ${endPoint.x} ${endPoint.y}`;
      labelX = midX + offsetX * 0.8;
      labelY = midY + offsetY * 0.8;
    }

    // Input style on the graph
    const inputStyle = {
      width: '50px',
      height: '28px',
      background: prevBelief ? (isHighlighted ? '#fff' : '#f0f0f0') : 'rgba(255, 255, 255, 0.95)',
      border: prevBelief ? (isHighlighted ? '2px solid #000' : '1px solid #ddd') : '1px solid #e0e0e0',
      borderRadius: '6px',
      textAlign: 'center',
      fontSize: '14px',
      fontWeight: '600',
      color: prevBelief ? (isHighlighted ? '#000' : '#999') : COLORS.text,
      padding: '0',
      boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
      outline: 'none',
      cursor: prevBelief ? 'default' : 'pointer'
    };

    // Determine label value: if prevBelief exists, show transition prob (prob), else show input value
    // We use 'displayFlow' calculated at the start of the function which is the posterior probability of the transition
    const displayValue = prevBelief ? displayFlow : 0;

    return (
      <g key={`${from}-${to}`} style={{ opacity, transition: 'opacity 0.2s' }}>
        <path 
          d={path} 
          fill="none" 
          stroke={strokeColor} 
          strokeWidth={strokeWidth} 
          markerEnd="url(#arrowhead)" 
          style={{ transition: 'all 0.2s' }}
        />
                <foreignObject x={labelX - 25} y={labelY - 14} width="50" height="28">
          {prevBelief ? (
             null
          ) : (
          <NormalizedInput 
            rawValue={parseFloat(value) || 0}
            totalSum={stateSums[from]}
            onChange={(newVal) => handleCellChange(from, to, newVal)}
            style={inputStyle}
            className="matrix-input"
          />
          )}
        </foreignObject>
      </g>
    );
  };

  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      gap: '20px', 
      background: 'white', 
      padding: '20px',
      borderRadius: '12px',
      boxShadow: '0 4px 20px rgba(0,0,0,0.05)',
      height: '100%',
      boxSizing: 'border-box'
    }}>
      
      {/* Top Bar: Preset Selection & Parameters */}
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        gap: '20px', 
        padding: '15px', 
        background: '#f5f5f7', 
        borderRadius: '8px',
        flexWrap: 'wrap'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <label style={{ fontWeight: 600, color: '#555' }}>Preset:</label>
                    <select 
            name="preset" 
            value={config.preset} 
            onChange={handleConfigChange}
            style={{ padding: '5px 10px', borderRadius: '4px', border: '1px solid #ccc' }}
          >
            <optgroup label="Test Processes">
              <option value="even_process">Even Process (E≈0.92)</option>
              <option value="golden_mean">Golden Mean (E≈0.25)</option>
              <option value="rrxor">RRXOR (5-state, E=2)</option>
            </optgroup>
            <optgroup label="3-State Processes">
              <option value="fern">Fern</option>
              <option value="mess3">Mess 3</option>
              <option value="left_right_mix">Left/Right Mix</option>
              <option value="cyclic_rank1">Cyclic Rank-1</option>
              <option value="rank1">Rank-1 (Fuzzy)</option>
              <option value="abc_ratio">ABC Ratio</option>
              <option value="rank1_predefined">Rank-1 (Predefined)</option>
              <option value="rank1-xmas">Rank-1 (Xmas)</option>
              <option value="smiley">Smiley Face</option>
              <option value="smiley_nested">Nested Smiley</option>
              <option value="smiley_9state">Smiley (9-State)</option>
              <option value="parabolic_curve">Parabolic Curve</option>
            </optgroup>
            <option value="custom">Custom</option>
          </select>
        </div>

        {config.preset === 'fern' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <label style={{ fontWeight: 600, color: '#555', minWidth: '20px' }}>X:</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              name="x"
              value={config.x}
              onChange={handleConfigChange}
              style={{ width: '120px' }}
            />
            <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.x}</span>
          </div>
        )}

        {config.preset === 'mess3' && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '20px' }}>X:</label>
              <input 
                type="range" 
                min="0" 
                max="0.5" 
                step="0.01" 
                name="x" 
                value={config.x} 
                onChange={handleConfigChange} 
                style={{ width: '120px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.x}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '20px' }}>A:</label>
              <input 
                type="range" 
                min="0" 
                max="1" 
                step="0.01" 
                name="a" 
                value={config.a} 
                onChange={handleConfigChange} 
                style={{ width: '120px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.a}</span>
            </div>
          </>
        )}

        {config.preset === 'left_right_mix' && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '20px' }}>A:</label>
              <input 
                type="range" 
                min="-0.46" 
                max="0.4" 
                step="0.01" 
                name="a" 
                value={config.a} 
                onChange={handleConfigChange} 
                style={{ width: '120px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.a}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '20px' }}>B:</label>
              <input 
                type="range" 
                min="0" 
                max={Math.max(0, 0.44 - (parseFloat(config.a) || 0))} 
                step="0.01" 
                name="b" 
                value={config.b} 
                onChange={handleConfigChange} 
                style={{ width: '120px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.b}</span>
            </div>
          </>
        )}

        {config.preset === 'cyclic_rank1' && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '60px' }}>Symbols:</label>
              <input 
                type="number" 
                min="2" 
                max="10" 
                step="1" 
                name="n_symbols" 
                value={config.n_symbols} 
                onChange={handleConfigChange} 
                style={{ width: '60px', padding: '4px', borderRadius: '4px', border: '1px solid #ccc' }}
              />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '60px' }}>Decay:</label>
              <input 
                type="range" 
                min="0" 
                max="1" 
                step="0.01" 
                name="state_decay" 
                value={config.state_decay} 
                onChange={handleConfigChange} 
                style={{ width: '80px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.state_decay}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '60px' }}>Contrast:</label>
              <input 
                type="range" 
                min="0" 
                max="1" 
                step="0.01" 
                name="contrast" 
                value={config.contrast} 
                onChange={handleConfigChange} 
                style={{ width: '80px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.contrast}</span>
            </div>
          </>
        )}

        {config.preset === 'rank1' && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '60px' }}>Symbols:</label>
              <input 
                type="number" 
                min="2" 
                max="10" 
                step="1" 
                name="n_symbols" 
                value={config.n_symbols} 
                onChange={handleConfigChange} 
                style={{ width: '60px', padding: '4px', borderRadius: '4px', border: '1px solid #ccc' }}
              />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '60px' }}>Decay:</label>
              <input 
                type="range" 
                min="0" 
                max="1" 
                step="0.01" 
                name="state_decay" 
                value={config.state_decay} 
                onChange={handleConfigChange} 
                style={{ width: '80px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.state_decay}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '60px' }}>Fuzziness:</label>
              <input 
                type="range" 
                min="0" 
                max="10" 
                step="0.1" 
                name="fuzziness" 
                value={config.fuzziness} 
                onChange={handleConfigChange} 
                style={{ width: '80px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.fuzziness}</span>
            </div>
          </>
        )}

        {config.preset === 'abc_ratio' && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '20px' }}>A:</label>
              <input 
                type="number" 
                min="1" 
                max="50" 
                step="1" 
                name="ratio_a" 
                value={config.ratio_a} 
                onChange={handleConfigChange} 
                style={{ width: '60px', padding: '4px', borderRadius: '4px', border: '1px solid #ccc' }}
              />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '20px' }}>B:</label>
              <input 
                type="number" 
                min="1" 
                max="50" 
                step="1" 
                name="ratio_b" 
                value={config.ratio_b} 
                onChange={handleConfigChange} 
                style={{ width: '60px', padding: '4px', borderRadius: '4px', border: '1px solid #ccc' }}
              />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '20px' }}>C:</label>
              <input 
                type="number" 
                min="1" 
                max="50" 
                step="1" 
                name="ratio_c" 
                value={config.ratio_c} 
                onChange={handleConfigChange} 
                style={{ width: '60px', padding: '4px', borderRadius: '4px', border: '1px solid #ccc' }}
              />
            </div>
            <div style={{ borderLeft: '2px solid #ccc', height: '24px', margin: '0 10px' }}></div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '20px' }}>D:</label>
              <input 
                type="number" 
                min="1" 
                max="50" 
                step="1" 
                name="ratio_d" 
                value={config.ratio_d} 
                onChange={handleConfigChange} 
                style={{ width: '60px', padding: '4px', borderRadius: '4px', border: '1px solid #ccc' }}
              />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '20px' }}>E:</label>
              <input 
                type="number" 
                min="1" 
                max="50" 
                step="1" 
                name="ratio_e" 
                value={config.ratio_e} 
                onChange={handleConfigChange} 
                style={{ width: '60px', padding: '4px', borderRadius: '4px', border: '1px solid #ccc' }}
              />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '20px' }}>F:</label>
              <input 
                type="number" 
                min="1" 
                max="50" 
                step="1" 
                name="ratio_f" 
                value={config.ratio_f} 
                onChange={handleConfigChange} 
                style={{ width: '60px', padding: '4px', borderRadius: '4px', border: '1px solid #ccc' }}
              />
            </div>
          </>
        )}

        {config.preset === 'rank1_predefined' && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '60px' }}>P Scale:</label>
              <input 
                type="range" 
                min="0" 
                max="1" 
                step="0.01" 
                name="p_scale" 
                value={config.p_scale !== undefined ? config.p_scale : 0.5} 
                onChange={handleConfigChange} 
                style={{ width: '80px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.p_scale !== undefined ? config.p_scale : 0.5}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '60px' }}>Split (a):</label>
              <input 
                type="range" 
                min="0" 
                max="1" 
                step="0.01" 
                name="a" 
                value={config.a} 
                onChange={handleConfigChange} 
                style={{ width: '80px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.a}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '20px' }}>A:</label>
              <input type="number" min="1" max="50" name="ratio_a" value={config.ratio_a} onChange={handleConfigChange} style={{ width: '40px' }} />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '20px' }}>B:</label>
              <input type="number" min="1" max="50" name="ratio_b" value={config.ratio_b} onChange={handleConfigChange} style={{ width: '40px' }} />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '20px' }}>C:</label>
              <input type="number" min="1" max="50" name="ratio_c" value={config.ratio_c} onChange={handleConfigChange} style={{ width: '40px' }} />
            </div>
          </>
        )}

        {config.preset === 'rank1-xmas' && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '60px' }}>Scale A:</label>
              <input 
                type="range" 
                min="0" 
                max="1" 
                step="0.01" 
                name="scale_a" 
                value={config.scale_a !== undefined ? config.scale_a : 0.9} 
                onChange={handleConfigChange} 
                style={{ width: '80px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.scale_a !== undefined ? config.scale_a : 0.9}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '60px' }}>Scale B:</label>
              <input 
                type="range" 
                min="0" 
                max="1" 
                step="0.01" 
                name="scale_b" 
                value={config.scale_b !== undefined ? config.scale_b : 0.9} 
                onChange={handleConfigChange} 
                style={{ width: '80px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.scale_b !== undefined ? config.scale_b : 0.9}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '20px' }}>S1:</label>
              <input 
                type="range" 
                min="0" 
                max="1" 
                step="0.01" 
                name="s1" 
                value={config.s1 !== undefined ? config.s1 : 0.5} 
                onChange={handleConfigChange} 
                style={{ width: '60px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.s1 !== undefined ? config.s1 : 0.5}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '20px' }}>S2:</label>
              <input 
                type="range" 
                min="0" 
                max="1" 
                step="0.01" 
                name="s2" 
                value={config.s2 !== undefined ? config.s2 : 0.5} 
                onChange={handleConfigChange} 
                style={{ width: '60px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.s2 !== undefined ? config.s2 : 0.5}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '20px' }}>S3:</label>
              <input 
                type="range" 
                min="0" 
                max="1" 
                step="0.01" 
                name="s3" 
                value={config.s3 !== undefined ? config.s3 : 0.5} 
                onChange={handleConfigChange} 
                style={{ width: '60px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.s3 !== undefined ? config.s3 : 0.5}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '20px' }}>A:</label>
              <input type="number" min="1" max="50" name="ratio_a" value={config.ratio_a} onChange={handleConfigChange} style={{ width: '40px' }} />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '20px' }}>B:</label>
              <input type="number" min="1" max="50" name="ratio_b" value={config.ratio_b} onChange={handleConfigChange} style={{ width: '40px' }} />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '20px' }}>C:</label>
              <input type="number" min="1" max="50" name="ratio_c" value={config.ratio_c} onChange={handleConfigChange} style={{ width: '40px' }} />
            </div>
          </>
        )}
        
        {config.preset === 'smiley' && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '70px' }}>Curvature:</label>
              <input
                type="range" min="0.01" max="0.15" step="0.005"
                name="curvature"
                value={config.curvature !== undefined ? config.curvature : 0.06}
                onChange={handleConfigChange}
                style={{ width: '100px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.curvature !== undefined ? config.curvature : 0.06}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '70px' }}>Depth:</label>
              <input
                type="range" min="0.05" max="0.25" step="0.005"
                name="depth"
                value={config.depth !== undefined ? config.depth : 0.12}
                onChange={handleConfigChange}
                style={{ width: '100px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.depth !== undefined ? config.depth : 0.12}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '70px' }}>Eye Height:</label>
              <input
                type="range" min="0.50" max="0.90" step="0.01"
                name="eye_height"
                value={config.eye_height !== undefined ? config.eye_height : 0.70}
                onChange={handleConfigChange}
                style={{ width: '100px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.eye_height !== undefined ? config.eye_height : 0.70}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '70px' }}>Eye Spread:</label>
              <input
                type="range" min="0.02" max="0.25" step="0.01"
                name="eye_spread"
                value={config.eye_spread !== undefined ? config.eye_spread : 0.14}
                onChange={handleConfigChange}
                style={{ width: '100px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.eye_spread !== undefined ? config.eye_spread : 0.14}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '70px' }}>Eye Isolation:</label>
              <input
                type="range" min="0" max="0.95" step="0.05"
                name="eye_isolation"
                value={config.eye_isolation !== undefined ? config.eye_isolation : 0.85}
                onChange={handleConfigChange}
                style={{ width: '100px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.eye_isolation !== undefined ? config.eye_isolation : 0.85}</span>
            </div>
          </>
        )}

        {config.preset === 'smiley_nested' && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '80px' }}>Curvature:</label>
              <input
                type="range" min="0.01" max="0.15" step="0.005"
                name="curvature"
                value={config.curvature !== undefined ? config.curvature : 0.06}
                onChange={handleConfigChange}
                style={{ width: '100px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.curvature !== undefined ? config.curvature : 0.06}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '80px' }}>Depth:</label>
              <input
                type="range" min="0.05" max="0.25" step="0.005"
                name="depth"
                value={config.depth !== undefined ? config.depth : 0.12}
                onChange={handleConfigChange}
                style={{ width: '100px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.depth !== undefined ? config.depth : 0.12}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '80px' }}>Eye Height:</label>
              <input
                type="range" min="0.50" max="0.90" step="0.01"
                name="eye_height"
                value={config.eye_height !== undefined ? config.eye_height : 0.76}
                onChange={handleConfigChange}
                style={{ width: '100px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.eye_height !== undefined ? config.eye_height : 0.76}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '80px' }}>Eye Size:</label>
              <input
                type="range" min="0.02" max="0.15" step="0.01"
                name="eye_size"
                value={config.eye_size !== undefined ? config.eye_size : 0.06}
                onChange={handleConfigChange}
                style={{ width: '100px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.eye_size !== undefined ? config.eye_size : 0.06}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '80px' }}>Eye Sep:</label>
              <input
                type="range" min="0.05" max="0.30" step="0.01"
                name="eye_separation"
                value={config.eye_separation !== undefined ? config.eye_separation : 0.16}
                onChange={handleConfigChange}
                style={{ width: '100px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.eye_separation !== undefined ? config.eye_separation : 0.16}</span>
            </div>
          </>
        )}

        {config.preset === 'smiley_9state' && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '80px' }}>Curvature:</label>
              <input
                type="range" min="0.01" max="0.15" step="0.005"
                name="curvature"
                value={config.curvature !== undefined ? config.curvature : 0.06}
                onChange={handleConfigChange}
                style={{ width: '100px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.curvature !== undefined ? config.curvature : 0.06}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '80px' }}>Depth:</label>
              <input
                type="range" min="0.05" max="0.25" step="0.005"
                name="depth"
                value={config.depth !== undefined ? config.depth : 0.12}
                onChange={handleConfigChange}
                style={{ width: '100px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.depth !== undefined ? config.depth : 0.12}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '80px' }}>Eye Height:</label>
              <input
                type="range" min="0.50" max="0.90" step="0.01"
                name="eye_height"
                value={config.eye_height !== undefined ? config.eye_height : 0.76}
                onChange={handleConfigChange}
                style={{ width: '100px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.eye_height !== undefined ? config.eye_height : 0.76}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '80px' }}>Eye Size:</label>
              <input
                type="range" min="0.01" max="0.10" step="0.005"
                name="eye_size"
                value={config.eye_size !== undefined ? config.eye_size : 0.04}
                onChange={handleConfigChange}
                style={{ width: '100px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.eye_size !== undefined ? config.eye_size : 0.04}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '80px' }}>Eye Sep:</label>
              <input
                type="range" min="0.02" max="0.12" step="0.005"
                name="eye_separation"
                value={config.eye_separation !== undefined ? config.eye_separation : 0.08}
                onChange={handleConfigChange}
                style={{ width: '100px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.eye_separation !== undefined ? config.eye_separation : 0.08}</span>
            </div>
          </>
        )}

        {config.preset === 'parabolic_curve' && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '80px' }}>Attr Height:</label>
              <input
                type="range" min="0.05" max="0.95" step="0.01"
                name="attr_height"
                value={config.attr_height !== undefined ? config.attr_height : 0.15}
                onChange={handleConfigChange}
                style={{ width: '100px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.attr_height !== undefined ? config.attr_height : 0.15}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '80px' }}>Attr Lean:</label>
              <input
                type="range" min="-0.80" max="0.80" step="0.01"
                name="attr_lean"
                value={config.attr_lean !== undefined ? config.attr_lean : 0.0}
                onChange={handleConfigChange}
                style={{ width: '100px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.attr_lean !== undefined ? config.attr_lean : 0.0}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '80px' }}>Start Height:</label>
              <input
                type="range" min="0.05" max="0.95" step="0.01"
                name="start_height"
                value={config.start_height !== undefined ? config.start_height : 0.70}
                onChange={handleConfigChange}
                style={{ width: '100px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.start_height !== undefined ? config.start_height : 0.70}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '80px' }}>Spread:</label>
              <input
                type="range" min="0.00" max="0.45" step="0.01"
                name="spread"
                value={config.spread !== undefined ? config.spread : 0.25}
                onChange={handleConfigChange}
                style={{ width: '100px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.spread !== undefined ? config.spread : 0.25}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '80px' }}>Speed:</label>
              <input
                type="range" min="0.02" max="0.40" step="0.01"
                name="speed"
                value={config.speed !== undefined ? config.speed : 0.12}
                onChange={handleConfigChange}
                style={{ width: '100px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.speed !== undefined ? config.speed : 0.12}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555', minWidth: '80px' }}>Shape:</label>
              <input
                type="range" min="0.5" max="4.0" step="0.1"
                name="shape"
                value={config.shape !== undefined ? config.shape : 2.0}
                onChange={handleConfigChange}
                style={{ width: '100px' }}
              />
              <span style={{ fontFamily: 'monospace', width: '40px' }}>{config.shape !== undefined ? config.shape : 2.0}</span>
            </div>
          </>
        )}

        <div style={{ flex: 1 }}></div>

        <div style={{
            fontSize: '0.9em',
            color: '#666', 
            fontStyle: 'italic' 
        }}>
            input values are normalized probabilities
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '40px', alignItems: 'center' }}>
        
        {/* Visual Graph - only for 3-state machines */}
        {is3State ? (
        <div style={{ 
          width: '100%',
          display: 'flex',
          justifyContent: 'center',
          padding: '20px',
          position: 'relative'
        }}>
        {prevBelief && (
            <div style={{ 
                position: 'absolute',
                top: '10px',
                right: '10px',
                maxWidth: '200px',
                padding: '8px 12px',
                background: COLORS.highlightBg,
                border: `1px solid ${COLORS.highlightBorder}`,
                borderRadius: '6px',
                color: COLORS.highlightText,
                fontWeight: 'bold',
                fontSize: '0.9em',
                zIndex: 10,
                pointerEvents: 'none',
                textAlign: 'right'
            }}>
                <div style={{ fontWeight: 'normal', fontSize: '0.85em', lineHeight: '1.4' }}>
                    {beliefMode === 'constrained' 
                        ? "Constrained Mode: Arrows show additive update contributions"
                        : "Standard Mode: Arrows show multiplicative belief updates"}
                </div>
            </div>
        )}
          <svg width={GRAPH_WIDTH} height={GRAPH_HEIGHT} style={{ overflow: 'visible' }}>
            <defs>
              <marker id="arrowhead" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
                <path d="M2,2 L10,6 L2,10 L2,2" fill={COLORS.nodeStroke} />
              </marker>
            </defs>
            
            {/* Edges */}
            {matrices[selectedSymbol].map((row, r) => 
              row.map((val, c) => renderArrow(r, c, val))
            )}

            {/* Nodes */}
            {states.map((s, idx) => (
              <g key={s}>
                <circle 
                  cx={positions[idx].x} 
                  cy={positions[idx].y} 
                  r="30" 
                  fill={COLORS.nodeFill} 
                  stroke={COLORS.nodeStroke} 
                  strokeWidth="2" 
                  style={{ filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))' }}
                />
                <text 
                  x={positions[idx].x} 
                  y={positions[idx].y} 
                  dy={prevBelief ? "-0.5em" : ".35em"} 
                  textAnchor="middle" 
                  fontWeight="bold" 
                  fontSize="18"
                  fill={COLORS.text}
                >
                  {s}
                </text>
                {prevBelief && (
                  <text
                    x={positions[idx].x}
                    y={positions[idx].y}
                    dy="1.0em"
                    textAnchor="middle"
                    fontSize="12"
                    fill="#666"
                    fontWeight="bold"
                  >
                    {Math.round(prevBelief[idx] * 100)}%
                  </text>
                )}
              </g>
            ))}
          </svg>
        </div>
        ) : (
          <div style={{
            padding: '30px',
            background: '#f9f9f9',
            borderRadius: '8px',
            color: '#666',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '16px', fontWeight: 'bold', marginBottom: '10px' }}>
              {numStates}-State Machine
            </div>
            <div style={{ fontSize: '14px' }}>
              HMM graph visualization is only available for 3-state machines.
              <br/>Use the matrix view below to edit transitions.
            </div>
          </div>
        )}

        {/* Matrix Input (LaTeX style) */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 1, minWidth: '300px' }}>
          <h4 style={{ marginBottom: '20px', color: '#666', fontWeight: 500 }}>Normalized Transition Matrices (Click to Edit)</h4>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', justifyContent: 'center' }}>
            {allNormalizedMatrices.map((normMatrix, symbolIdx) => (
              <div key={symbolIdx} style={{ 
                display: 'flex', 
                flexDirection: 'column', 
                alignItems: 'center',
                opacity: selectedSymbol === symbolIdx ? 1 : 0.5,
                transform: selectedSymbol === symbolIdx ? 'scale(1.05)' : 'scale(1)',
                transition: 'all 0.3s ease',
                cursor: 'pointer',
                border: selectedSymbol === symbolIdx ? '1px solid #ccc' : '1px solid transparent',
                padding: '20px',
                borderRadius: '12px',
                background: selectedSymbol === symbolIdx ? '#f9f9f9' : 'transparent'
              }}
              onClick={() => onSymbolChange(symbolIdx)}
              >
                <div style={{ marginBottom: '15px', fontWeight: 'bold', color: '#555', fontSize: '1.3em' }}>Symbol {symbolIdx}</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                  <div style={{ 
                    display: 'grid', 
                    gridTemplateColumns: `repeat(${numStates}, 1fr)`, 
                    gap: '15px',
                    padding: '20px 25px',
                    position: 'relative',
                    margin: '0 5px'
                  }}>
                    {/* Brackets */}
                    <div style={{ position: 'absolute', top: 0, bottom: 0, left: 0, width: '15px', border: '3px solid #333', borderRight: 'none', borderRadius: '10px 0 0 10px' }}></div>
                    <div style={{ position: 'absolute', top: 0, bottom: 0, right: 0, width: '15px', border: '3px solid #333', borderLeft: 'none', borderRadius: '0 10px 10px 0' }}></div>

                    {normMatrix.map((row, r) => 
                      row.map((val, c) => (
                        <div key={`${r}-${c}`} style={{ width: '60px', textAlign: 'center', fontSize: '20px', fontFamily: 'monospace', fontWeight: 'bold', color: '#333' }}>
                          {typeof val === 'number' ? val.toFixed(2) : val}
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Sum Matrix (Full Transition Matrix) */}
          <div style={{ 
            marginTop: '30px', 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center',
            padding: '20px',
            borderRadius: '12px',
            background: '#e8f4e8',
            border: '2px solid #4caf50'
          }}>
            <div style={{ marginBottom: '15px', fontWeight: 'bold', color: '#2e7d32', fontSize: '1.3em' }}>
              Full Transition Matrix (Σ Symbols)
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: `repeat(${numStates}, 1fr)`, 
                gap: '15px',
                padding: '20px 25px',
                position: 'relative',
                margin: '0 5px'
              }}>
                {/* Brackets */}
                <div style={{ position: 'absolute', top: 0, bottom: 0, left: 0, width: '15px', border: '3px solid #2e7d32', borderRight: 'none', borderRadius: '10px 0 0 10px' }}></div>
                <div style={{ position: 'absolute', top: 0, bottom: 0, right: 0, width: '15px', border: '3px solid #2e7d32', borderLeft: 'none', borderRadius: '0 10px 10px 0' }}></div>

                {(() => {
                  // Calculate sum matrix
                  const sumMatrix = allNormalizedMatrices.reduce((acc, matrix) => {
                    return acc.map((row, r) => row.map((val, c) => val + matrix[r][c]));
                  }, allNormalizedMatrices[0].map(row => row.map(() => 0)));
                  
                  return sumMatrix.map((row, r) => 
                    row.map((val, c) => (
                      <div key={`sum-${r}-${c}`} style={{ width: '60px', textAlign: 'center', fontSize: '20px', fontFamily: 'monospace', fontWeight: 'bold', color: '#2e7d32' }}>
                        {typeof val === 'number' ? val.toFixed(2) : val}
                      </div>
                    ))
                  );
                })()}
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default MatrixEditor;
