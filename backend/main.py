from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import jax
import jax.numpy as jnp
import numpy as np
from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel
from simplexity.generative_processes.transition_matrices import mess3
import matrices
from visualization import plotly_simplex_vs_raw
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    batch_size: int = 512
    sequence_len: int = 50
    matrices: list

class PresetRequest(BaseModel):
    key: str
    kwargs: dict = {}

def get_matrices_array(key, kwargs):
    if key == "simple_constrained":
        return matrices.simple_constrained(**kwargs)
    if key == 'santa':
        return matrices.santa(**kwargs)
    if key == 'mess3':
        return mess3(**kwargs)
    if key == 'left_right_mix':
        return matrices.left_right_mix(**kwargs)
    raise ValueError(f"Unknown matrix key: {key}")

@app.post("/get_preset")
async def get_preset(request: PresetRequest):
    try:
        m = get_matrices_array(request.key, request.kwargs)
        # Convert JAX array to list for JSON serialization
        return m.tolist()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/generate")
async def generate_data(request: GenerateRequest):
    try:
        # Convert list back to JAX array
        matrix_params = jnp.array(request.matrices)
        generator = HiddenMarkovModel(matrix_params)
        
        # Use a fixed seed for reproducibility or make it random
        key = jax.random.PRNGKey(int(1E9)) 
        batch_keys = jax.random.split(key, request.batch_size)
        
        initial_states = jnp.tile(generator.initial_state, (request.batch_size, 1))
        true_states, observations = generator.generate(initial_states, batch_keys, request.sequence_len, True)
        
        # Process observations to get words
        # Convert to numpy for easier handling
        obs_np = np.array(observations)
        
        words = []
        for i in range(min(50, request.batch_size)):
            seq = obs_np[i]
            # If one-hot (3D array), convert to indices
            if seq.ndim > 1:
                 seq = np.argmax(seq, axis=-1)
            
            # Convert to string
            # Using simple mapping 0->'0', 1->'1', etc.
            word = "".join([str(int(s)) for s in seq])
            
            words.append(word)

        # Calculate belief states for the first 5 sequences
        # matrix_params shape: (num_symbols, num_states, num_states)
        # T[k, i, j] = P(S_{t+1}=j, O_{t+1}=k | S_t=i)
        
        belief_states_list = []
        num_display = min(50, request.batch_size)
        
        # Convert to numpy for processing
        T = np.array(matrix_params) 
        
        for i in range(num_display):
            # Get full observation sequence (indices)
            seq_obs = obs_np[i]
            if seq_obs.ndim > 1:
                seq_obs = np.argmax(seq_obs, axis=-1)
            
            # Initial belief (uniform or from model)
            # Assuming uniform start for now or stationary distribution
            # But generator.initial_state is used in generation
            current_belief = np.array(generator.initial_state)
            
            seq_beliefs = []
            
            # For each observation in the sequence
            for obs_idx in seq_obs:
                # obs_idx is the symbol k
                # Update: b_{t+1} = b_t * T[k]
                # T[k] shape is (num_states, num_states) -> (from, to)
                
                # Vector-matrix multiplication: 
                # new_belief[j] = sum_i (old_belief[i] * T[k][i][j])
                next_belief = current_belief @ T[obs_idx]
                
                # Normalize
                norm = np.sum(next_belief)
                if norm > 0:
                    next_belief = next_belief / norm
                else:
                    # Fallback if probability vanishes (shouldn't happen in well-formed HMM)
                    next_belief = np.ones_like(next_belief) / len(next_belief)
                
                current_belief = next_belief
                seq_beliefs.append(current_belief.tolist())
            
            belief_states_list.append(seq_beliefs)

        fig = plotly_simplex_vs_raw(true_states)
        
        return {
            "plot": json.loads(fig.to_json()),
            "words": words,
            "belief_states": belief_states_list,
            "initial_state": np.array(generator.initial_state).tolist()
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "ok"}
