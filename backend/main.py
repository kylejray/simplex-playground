from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from hmm import HiddenMarkovModel
from matrices import mess3
import matrices
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
    batch_size: int = 128
    sequence_len: int = 25
    matrices: list

class PresetRequest(BaseModel):
    key: str
    kwargs: dict = {}

def get_matrices_array(key, kwargs):
    if key == 'mess3':
        return mess3(**kwargs)
    if key == 'left_right_mix':
        return matrices.left_right_mix(**kwargs)
    raise ValueError(f"Unknown matrix key: {key}")

@app.post("/get_preset")
async def get_preset(request: PresetRequest):
    try:
        m = get_matrices_array(request.key, request.kwargs)
        # Convert numpy array to list for JSON serialization
        return m.tolist()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/generate")
async def generate_data(request: GenerateRequest):
    try:
        # Convert list back to numpy array
        matrix_params = np.array(request.matrices)
        generator = HiddenMarkovModel(matrix_params)
        
        words, belief_states, initial_state = generator.generate(request.batch_size, request.sequence_len)
        
        # Generate plot data
        # We need to flatten the belief states for plotting
        # belief_states is list of lists of arrays
        
        # Flatten all belief states from all sequences
        all_beliefs = []
        for seq in belief_states:
            all_beliefs.extend(seq)
            
        # Return raw data for frontend visualization
        return {
            "words": words,
            "belief_states": belief_states,
            "initial_state": initial_state,
            "flat_beliefs": all_beliefs
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "ok"}
