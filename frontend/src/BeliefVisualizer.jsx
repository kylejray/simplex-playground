import React, { useState, useEffect } from 'react';

const VIS_WIDTH = 450;
const VIS_HEIGHT = 390;
const VIS_PADDING = 50;

const COLORS = {
  A: 'rgb(255, 0, 0)',
  B: 'rgb(0, 255, 0)',
  C: 'rgb(0, 0, 255)',
  background: '#f8f9fa',
  stroke: '#ccc',
  text: '#222',
  subText: '#555'
};

const BeliefVisualizer = ({ probabilities }) => {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    if (probabilities) {
      setHistory(prev => [...prev, probabilities]);
    }
  }, [probabilities]);

  // Default to equal probability if undefined
  const probs = probabilities || [0.33, 0.33, 0.33];
  
  // Equilateral Triangle Vertices
  // Rotated to match Plotly: A (Bottom Left), B (Bottom Right), C (Top)
  const A = { x: VIS_PADDING, y: VIS_HEIGHT - VIS_PADDING };
  const B = { x: VIS_WIDTH - VIS_PADDING, y: VIS_HEIGHT - VIS_PADDING };
  const C = { x: VIS_WIDTH / 2, y: VIS_PADDING };
  
  const getCoords = (p) => ({
    x: p[0] * A.x + p[1] * B.x + p[2] * C.x,
    y: p[0] * A.y + p[1] * B.y + p[2] * C.y
  });

  const getColor = (p) => {
    const r = Math.round(p[0] * 255);
    const g = Math.round(p[1] * 255);
    const b = Math.round(p[2] * 255);
    return `rgb(${r}, ${g}, ${b})`;
  };

  // Calculate point position based on probabilities (Barycentric coordinates)
  const { x, y } = getCoords(probs);
  const beliefColor = getColor(probs);

  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center',
      background: 'white',
      padding: '20px',
      borderRadius: '12px',
      boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
      width: '100%',
      boxSizing: 'border-box'
    }}>
      <h4 style={{ margin: '0 0 15px 0', color: '#444', fontSize: '18px', fontWeight: 600 }}>Belief State Visualization</h4>
      <svg width={VIS_WIDTH} height={VIS_HEIGHT} style={{ overflow: 'visible' }}>
        {/* Triangle Background */}
        <path 
          d={`M ${A.x} ${A.y} L ${B.x} ${B.y} L ${C.x} ${C.y} Z`} 
          fill={COLORS.background} 
          stroke={COLORS.stroke} 
          strokeWidth="2" 
        />
        
        {/* Vertices Labels & Indicators */}
        {[
          { label: 'A', pos: A, val: probs[0], color: COLORS.A },
          { label: 'B', pos: B, val: probs[1], color: COLORS.B },
          { label: 'C', pos: C, val: probs[2], color: COLORS.C }
        ].map((node, idx) => (
          <g key={idx}>
            {/* Probability Circle (Size varies) */}
            <circle 
              cx={node.pos.x} 
              cy={node.pos.y} 
              r={15 + node.val * 25} 
              fill={node.color} 
              opacity={0.2 + node.val * 0.6}
            />
            {/* Label */}
            <text 
              x={node.pos.x} 
              y={node.pos.y} 
              dy="6" 
              textAnchor="middle" 
              fill={COLORS.text} 
              fontWeight="bold" 
              fontSize="16"
            >
              {node.label}
            </text>
            {/* Value Text */}
            <text 
              x={node.pos.x} 
              y={node.pos.y + (node.pos.y > VIS_HEIGHT/2 ? 35 : -35)} 
              textAnchor="middle" 
              fill={COLORS.subText} 
              fontSize="14"
              fontWeight="500"
            >
              {(node.val * 100).toFixed(0)}%
            </text>
          </g>
        ))}

        {/* History Points */}
        {history.map((p, i) => {
          const coords = getCoords(p);
          return (
            <circle 
              key={i}
              cx={coords.x} 
              cy={coords.y} 
              r="4" 
              fill={getColor(p)} 
              opacity="0.6"
            />
          );
        })}

        {/* Current Belief Point */}
        <circle 
          cx={x} 
          cy={y} 
          r="10" 
          fill={beliefColor} 
          stroke="black" 
          strokeWidth="2"
          style={{ transition: 'all 0.2s ease-out' }}
        />
      </svg>
      
      <button 
        onClick={() => setHistory([])}
        style={{
          marginTop: '10px',
          padding: '6px 12px',
          fontSize: '14px',
          cursor: 'pointer',
          background: '#fff',
          border: '1px solid #ccc',
          borderRadius: '4px',
          color: '#555'
        }}
      >
        Clear History
      </button>
    </div>
  );
};

export default BeliefVisualizer;
