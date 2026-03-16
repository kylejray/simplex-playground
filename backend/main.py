from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np
from hmm import HiddenMarkovModel
from matrices import mess3
import matrices
import json
import os
import sys
import time
import asyncio

print("Starting FastAPI application...", file=sys.stderr)

app = FastAPI()

# Configure CORS to allow all origins (public API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    batch_size: int = 128
    sequence_len: int = 25
    matrices: list
    generation_mode: str = 'random' # 'random' or 'systematic'
    max_words: int = 1000

class PresetRequest(BaseModel):
    key: str
    kwargs: dict = {}

class ExcessEntropyRequest(BaseModel):
    matrices: list
    max_block_length: int = 20
    max_time_per_L: float = 2.0  # seconds before switching to MC (sample_constrained mode)
    mc_samples: int = 50000  # increased from 10000
    sample_constrained: bool = False  # False = time constrained (default), True = MC fallback
    time_budget: float = 10.0  # total time budget for time constrained mode

def get_matrices_array(key, kwargs):
    if key == 'mess3':
        return mess3(**kwargs)
    if key == 'left_right_mix':
        return matrices.left_right_mix(**kwargs)
    if key == 'cyclic_rank1':
        return matrices.cyclic_rank1(**kwargs)
    if key == 'rank1':
        return matrices.rank1(**kwargs)
    if key == 'abc_ratio':
        return matrices.abc_ratio(**kwargs)
    if key == 'rank1_predefined':
        return matrices.rank1_predefined(**kwargs)
    if key == 'rank1-xmas':
        return matrices.rank1_xmas(**kwargs)
    if key == 'even_process':
        return matrices.even_process(**kwargs)
    if key == 'golden_mean':
        return matrices.golden_mean(**kwargs)
    if key == 'rrxor':
        return matrices.rrxor()
    if key == 'fern':
        return matrices.fern(**kwargs)
    if key == 'smiley':
        return matrices.smiley(**kwargs)
    if key == 'smiley_nested':
        return matrices.smiley_nested(**kwargs)
    if key == 'smiley_9state':
        return matrices.smiley_9state(**kwargs)
    if key == 'parabolic_curve':
        return matrices.parabolic_curve(**kwargs)
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
        
        if request.generation_mode == 'systematic':
            # Use sequence_len as max_len
            words, belief_states, constrained_beliefs, initial_state = generator.generate_systematic(
                max_len=request.sequence_len, 
                max_count=request.max_words
            )
        else:
            words, belief_states, constrained_beliefs, initial_state = generator.generate(request.batch_size, request.sequence_len)
        
        # Generate plot data
        # We need to flatten the belief states for plotting
        # belief_states is list of lists of arrays
        
        # Flatten all belief states from all sequences
        all_beliefs = []
        for seq in belief_states:
            all_beliefs.extend(seq)

        all_constrained_beliefs = []
        for seq in constrained_beliefs:
            all_constrained_beliefs.extend(seq)
            
        # Return raw data for frontend visualization
        return {
            "words": words,
            "belief_states": belief_states,
            "constrained_beliefs": constrained_beliefs,
            "initial_state": initial_state,
            "flat_beliefs": all_beliefs,
            "flat_constrained_beliefs": all_constrained_beliefs
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/excess_entropy")
async def compute_excess_entropy(request: ExcessEntropyRequest):
    """
    Compute excess entropy with streaming progress updates.
    Uses Server-Sent Events (SSE) to report progress for each L.

    Two modes:
    - Time Constrained (default): Stops when total time exceeds time_budget
    - Sample Constrained: Switches to MC when a single L takes too long, continues to max_block_length

    Sends heartbeat messages every 2 seconds during long computations to prevent proxy timeouts.
    """
    import concurrent.futures

    async def generate():
        try:
            # Convert list back to numpy array
            matrix_params = np.array(request.matrices)
            hmm = HiddenMarkovModel(matrix_params)

            block_entropies = [(0, 0.0, 'exact', 0.0)]  # (L, H, method, time)
            mc_start_L = None
            use_mc = False
            total_start = time.time()
            prev_H = 0.0  # H(L-1) for lower bound calculation

            # Use a thread pool to run blocking computations
            loop = asyncio.get_event_loop()
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

            for L in range(1, request.max_block_length + 1):
                # Time constrained mode: check total elapsed time before starting new L
                if not request.sample_constrained:
                    total_elapsed = time.time() - total_start
                    if total_elapsed >= request.time_budget:
                        break  # Stop - time budget exceeded

                L_start = time.time()

                if not use_mc:
                    # Run exact enumeration in a thread so we can send heartbeats
                    future = loop.run_in_executor(
                        executor,
                        hmm._compute_block_entropy_exact,
                        L
                    )

                    # Send heartbeats while waiting for computation
                    heartbeat_interval = 2.0  # seconds
                    while True:
                        try:
                            H_L = await asyncio.wait_for(
                                asyncio.shield(future),
                                timeout=heartbeat_interval
                            )
                            break  # Computation finished
                        except asyncio.TimeoutError:
                            # Send heartbeat to keep connection alive
                            heartbeat = {
                                'type': 'heartbeat',
                                'L': L,
                                'elapsed': time.time() - L_start,
                                'method': 'exact'
                            }
                            yield f"data: {json.dumps(heartbeat)}\n\n"

                    L_time = time.time() - L_start
                    block_entropies.append((L, H_L, 'exact', L_time))

                    # Sample constrained mode: switch to MC if this L took too long
                    if request.sample_constrained and L_time > request.max_time_per_L:
                        use_mc = True
                        mc_start_L = L + 1
                else:
                    # Use Monte Carlo (sample constrained mode only)
                    if mc_start_L is None:
                        mc_start_L = L

                    future = loop.run_in_executor(
                        executor,
                        hmm._compute_block_entropy_mc,
                        L,
                        request.mc_samples
                    )

                    # Send heartbeats while waiting
                    heartbeat_interval = 2.0
                    while True:
                        try:
                            H_L = await asyncio.wait_for(
                                asyncio.shield(future),
                                timeout=heartbeat_interval
                            )
                            break
                        except asyncio.TimeoutError:
                            heartbeat = {
                                'type': 'heartbeat',
                                'L': L,
                                'elapsed': time.time() - L_start,
                                'method': 'mc'
                            }
                            yield f"data: {json.dumps(heartbeat)}\n\n"

                    L_time = time.time() - L_start
                    block_entropies.append((L, H_L, 'mc', L_time))

                # Compute lower bound estimate using H(L) and H(L-1)
                E_lower, h_estimate = hmm._compute_lower_bound(H_L, prev_H, L)
                prev_H = H_L  # Update for next iteration

                # Send progress update with E_lower
                progress = {
                    'type': 'progress',
                    'L': L,
                    'H': H_L,
                    'method': 'mc' if (use_mc or (mc_start_L and L >= mc_start_L)) else 'exact',
                    'time': L_time,
                    'E_lower': E_lower,
                    'h_estimate': h_estimate
                }
                yield f"data: {json.dumps(progress)}\n\n"
                await asyncio.sleep(0)  # Allow other tasks to run

            executor.shutdown(wait=False)

            # Compute final result
            result = hmm._finalize_excess_entropy(
                [(L, H, m) for L, H, m, t in block_entropies],
                request.mc_samples,
                mc_start_L
            )

            # Add timing info and lower bound to result
            result['block_entropies'] = [(L, H, m, t) for L, H, m, t in block_entropies]
            result['computation_time'] = time.time() - total_start
            result['mode'] = 'sample_constrained' if request.sample_constrained else 'time_constrained'

            # Add final E_lower from last two block entropies
            if len(block_entropies) >= 2:
                L_final, H_final, _, _ = block_entropies[-1]
                _, H_prev, _, _ = block_entropies[-2]
                E_lower_final, h_final = hmm._compute_lower_bound(H_final, H_prev, L_final)
                result['E_lower'] = E_lower_final
                result['h_estimate'] = h_final

            # Send final result
            final = {'type': 'result', 'data': result}
            yield f"data: {json.dumps(final)}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            error = {'type': 'error', 'message': str(e)}
            yield f"data: {json.dumps(error)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/")
def read_root():
    return {"status": "ok"}
