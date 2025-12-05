import React from 'react';

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

const MatrixEditor = ({ matrices, onChange, config, onConfigChange, selectedSymbol, onSymbolChange, prevBelief }) => {
  
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

  const states = ['A', 'B', 'C'];
  
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
              totalFlow += prevBelief[r] * weight;
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
             <div style={{ ...inputStyle, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '13px' }}>
                {displayValue === 0 ? "0%" : (displayValue * 100 < 1 ? "<1%" : Math.round(displayValue * 100) + "%")}
             </div>
          ) : (
          <input 
            type="text" 
            inputMode="decimal"
            value={value}
            onChange={(e) => handleCellChange(from, to, e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
                e.preventDefault();
                let currentVal = parseFloat(value);
                if (isNaN(currentVal)) currentVal = 0;
                
                let newVal;
                // Increase/decrease by 20%
                const factor = 0.2; 

                if (e.key === 'ArrowUp') {
                    if (currentVal === 0) {
                        newVal = 0.1; // Jump start from 0
                    } else {
                        newVal = currentVal * (1 + factor);
                    }
                } else {
                    // ArrowDown
                    newVal = currentVal * (1 - factor);
                    // Snap to 0 if very small
                    if (newVal < 0.01) newVal = 0;
                }

                // Round to avoid floating point errors, keep reasonable precision
                const rounded = Math.round(newVal * 10000) / 10000;
                // Convert back to string for the handler
                handleCellChange(from, to, rounded.toString());
              }
            }}
            style={inputStyle}
            className="matrix-input"
            autoComplete="off"
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
      marginTop: '20px',
      background: 'white',
      padding: '20px',
      borderRadius: '12px',
      boxShadow: '0 4px 20px rgba(0,0,0,0.05)'
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
            <option value="santa">santa</option>
            <option value="mess3">mess3</option>
            <option value="left_right_mix">left_right_mix</option>
            <option value="custom">Custom (Manual)</option>
          </select>
        </div>

        {config.preset === 'mess3' && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555' }}>X:</label>
              <input 
                type="number" 
                step="any" 
                name="x" 
                value={config.x} 
                onChange={handleConfigChange} 
                style={{ width: '80px', padding: '5px', borderRadius: '4px', border: '1px solid #ccc' }}
              />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontWeight: 600, color: '#555' }}>A:</label>
              <input 
                type="number" 
                step="any" 
                name="a" 
                value={config.a} 
                onChange={handleConfigChange} 
                style={{ width: '80px', padding: '5px', borderRadius: '4px', border: '1px solid #ccc' }}
              />
            </div>
          </>
        )}
        
        <div style={{ flex: 1 }}></div>
        
        <div style={{ 
            fontSize: '0.9em', 
            color: '#666', 
            fontStyle: 'italic' 
        }}>
            Values are "strengths" & automatically normalized.
        </div>
      </div>

      <div style={{ display: 'flex', gap: '40px', flexWrap: 'wrap', alignItems: 'flex-start' }}>
        
        {/* Visual Graph */}
        <div style={{ 
          flex: 2, 
          minWidth: `${GRAPH_WIDTH}px`, 
          display: 'flex',
          justifyContent: 'center',
          padding: '20px',
          position: 'relative'
        }}>
        {prevBelief && (
            <div style={{ 
                position: 'absolute',
                top: '10px',
                left: '10px',
                padding: '8px 12px',
                background: COLORS.highlightBg,
                border: `1px solid ${COLORS.highlightBorder}`,
                borderRadius: '6px',
                color: COLORS.highlightText,
                fontWeight: 'bold',
                fontSize: '0.9em',
                zIndex: 10,
                pointerEvents: 'none'
            }}>
                Total Likelihood (Z): {(totalFlow * 100).toFixed(1)}%
                <div style={{ fontWeight: 'normal', fontSize: '0.85em', marginTop: '4px', lineHeight: '1.4' }}>
                    Sum of incoming arrows = Next Belief State
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
                  dy={prevBelief ? "-0.2em" : ".35em"} 
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
                    dy="1.2em"
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
                    gridTemplateColumns: 'repeat(3, 1fr)', 
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
        </div>

      </div>
    </div>
  );
};

export default MatrixEditor;
