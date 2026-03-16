import numpy as np


def even_process(p: float = 0.5) -> np.ndarray:
    """
    Even Process: Binary process where 1s always come in pairs.
    States: A (even parity - ready for 0 or start of 1-pair), B (odd parity - must emit another 1)
    Alphabet: {0, 1}
    
    From state A: emit 0 -> stay A (prob 1-p), emit 1 -> go to B (prob p)
    From state B: emit 1 -> go to A (prob 1, must complete the pair)
    
    Parameter p controls P(emit 1 | state A). Default 0.5.
    Known excess entropy: E ≈ 0.92 bits (at p=0.5)
    
    T[k, i, j] = P(next_state=j, symbol=k | current_state=i)
    """
    return np.array([
        # Symbol 0
        [[1-p, 0],    # From A: emit 0 with prob 1-p, stay A
         [0, 0]],     # From B: cannot emit 0
        # Symbol 1  
        [[0, p],      # From A: emit 1 with prob p, go to B
         [1, 0]]      # From B: emit 1 with prob 1, go to A
    ])


def golden_mean(p: float = 0.5) -> np.ndarray:
    """
    Golden Mean Process: Binary process where '11' is forbidden.
    States: A (can emit anything), B (just emitted 1, must emit 0 next)
    Alphabet: {0, 1}
    
    From state A: emit 0 -> stay A, emit 1 -> go to B  
    From state B: emit 0 -> go to A (cannot emit 1)
    
    Parameter p controls P(emit 1 | state A). Default 0.5.
    Known excess entropy: E ≈ 0.2516 bits (at p=0.5)
    
    T[k, i, j] = P(next_state=j, symbol=k | current_state=i)
    """
    return np.array([
        # Symbol 0
        [[1-p, 0],    # From A: emit 0 with prob 1-p, stay A
         [1, 0]],     # From B: emit 0 with prob 1, go to A
        # Symbol 1
        [[0, p],      # From A: emit 1 with prob p, go to B
         [0, 0]]      # From B: cannot emit 1
    ])


def rrxor() -> np.ndarray:
    """
    Random-Random XOR (RRXOR) Process - 5-state epsilon machine.
    
    The process generates X_t = R_t XOR R_{t-1} where R_t are iid Bernoulli(1/2).
    The epsilon machine has 5 causal states tracking mixed-state beliefs.
    
    States: 0, 1, 2, 3, 4
    Alphabet: {0, 1}
    
    Known: h_mu = 1 bit/symbol, E = 2 bits
    
    T[k, i, j] = P(next_state=j, symbol=k | current_state=i)
    """
    T0 = np.array([
        [0,   0.5, 0,   0,   0  ],  # From 0: -> 1
        [0,   0,   0,   0,   0.5],  # From 1: -> 4
        [0,   0,   0,   0.5, 0  ],  # From 2: -> 3
        [0,   0,   0,   0,   0  ],  # From 3: (no 0 emission)
        [1,   0,   0,   0,   0  ],  # From 4: -> 0
    ])
    
    T1 = np.array([
        [0,   0,   0.5, 0,   0  ],  # From 0: -> 2
        [0,   0,   0,   0.5, 0  ],  # From 1: -> 3
        [0,   0,   0,   0,   0.5],  # From 2: -> 4
        [1,   0,   0,   0,   0  ],  # From 3: -> 0
        [0,   0,   0,   0,   0  ],  # From 4: (no 1 emission)
    ])
    
    return np.array([T0, T1])


def mess3(x: float, a: float) -> np.ndarray:
    """
    Creates a transition matrix for the Mess3 Process.
    Returns shape (3, 3, 3).
    """
    b = (1 - a) / 2
    y = 1 - 2 * x

    ay = a * y
    bx = b * x
    by = b * y
    ax = a * x

    return np.array(
        [
            [
                [ay, bx, bx],
                [ax, by, bx],
                [ax, bx, by],
            ],
            [
                [by, ax, bx],
                [bx, ay, bx],
                [bx, ax, by],
            ],
            [
                [by, bx, ax],
                [bx, by, ax],
                [bx, bx, ay],
            ],
        ]
    )

def left_right_mix(a: float = 0.0, b: float = 0.0) -> np.ndarray:
    """
    Returns the 'left_right_mix' preset matrices.
    Symbol 0: Left Cycle (A->B->C->A) dominant.
    Symbol 1: Right Cycle (A->C->B->A) dominant.
    Symbol 2: Uniform noise.
    
    Parameters:
    a (float): Asymmetry parameter.
    b (float): Leak parameter for Symbol 1 (A->C).
    """
    # Symbol 0: Left Cycle dominant
    # Transitions A->B, B->C, C->A are 0.5 + a
    # Others are 0 (implied by normalization if not specified, but here we set explicitly)
    # Note: This sums to > 1 if we just set one entry. 
    # The user specification implies these are the weights.
    # We will set the cycle transitions to 0.5 + a, and others to 0.
    # Wait, if others are 0, then 0.5+a must be 1.0? No.
    # The user likely means the weights are these values, and they are normalized later.
    # OR, the user implies the *other* transitions fill the gap?
    # Given the precise math check (0.5+a + 0.44-a + ... = 1), 
    # it implies these are components of a larger system, but here we return 3 matrices.
    # We will return the weights as specified. The frontend normalizes them.
    
    val0 = 0.5 + a
    T0 = np.array([
        [0.0, val0, 0.0],  # A -> B
        [0.0, 0.0, val0],  # B -> C
        [val0, 0.0, 0.0]   # C -> A
    ])

    # Symbol 1: Right Cycle dominant
    # B->A and C->B are 0.44 - a
    # A->C is 0.44 - a - b
    val1 = 0.44 - a
    val1_leak = 0.44 - a - b
    
    T1 = np.array([
        [0.0, 0.0, val1_leak], # A -> C
        [val1, 0.0, 0.0],      # B -> A
        [0.0, val1, 0.0]       # C -> B
    ])

    # Symbol 2: Noise
    # B and C outgoing: 0.02 to all states
    # A outgoing: 0.02 + b/3 to all states
    val2_bc = 0.02
    val2_a = 0.02 + (b / 3.0)
    
    T2 = np.array([
        [val2_a, val2_a, val2_a],    # From A
        [val2_bc, val2_bc, val2_bc], # From B
        [val2_bc, val2_bc, val2_bc]  # From C
    ])

    return np.array([T0, T1, T2])


def cyclic_rank1(n_states: int = 3, n_symbols: int = 3, 
                 state_decay: float = 1.0,
                 contrast: float = 1.0) -> np.ndarray:
    """
    Generates M High-Rank symbol matrices that sum to a Rank-1 transition matrix.
    Uses a cyclic modulo strategy to ensure ALL M symbols are used.
    
    Args:
        n_states (int): Number of hidden states.
        n_symbols (int): Number of emission symbols.
        state_decay (float): Sharpness of next-state distribution (higher = more peaked).
        contrast (float): 0.0 to 1.0.
                          1.0 = Strict cyclic assignment (Symbol k is ONLY emitted if (j-i)%M == k).
                          0.0 = Uniform emissions (All symbols equally likely everywhere).
                          
    Returns:
        np.ndarray: Array of shape (n_symbols, n_states, n_states) containing the symbol matrices.
    """
    
    # --- 1. Create the Rank-1 State Transition Matrix ---
    # P(Next State) is independent of Current State
    x = np.arange(n_states, dtype=np.float32)
    weights = np.exp(-state_decay * x)
    weights /= np.sum(weights)
    vec_states = weights  # Peak is always state 0 (A)
    
    # The Rank-1 Base Matrix - Shape: (N, N)
    P_state_trans = np.tile(vec_states, (n_states, 1))

    # --- 2. Create Cyclic Emission Masks ---
    # Formula: Preferred_Symbol = (Col_Index - Row_Index) % M
    
    # Create grids of row and col indices
    rows, cols = np.indices((n_states, n_states))
    
    # Calculate the "distance" map modulo M
    cyclic_map = (cols - rows) % n_symbols
    
    # Create the Emission Tensor (N, N, M)
    # Start with uniform noise
    emission_probs = np.ones((n_states, n_states, n_symbols)) / n_symbols
    
    # Create the "Target" structured distribution
    target_probs = np.zeros((n_states, n_states, n_symbols))
    for k in range(n_symbols):
        target_probs[:, :, k] = (cyclic_map == k).astype(float)
        
    # Interpolate based on contrast
    P_emission_cond = (1 - contrast) * emission_probs + contrast * target_probs
    
    # --- 3. Combine to Form Symbol Matrices ---
    # T_k = P(S'|S) * P(O=k | S, S')
    
    # Expand P_state_trans to (N, N, 1) for broadcasting
    P_state_expanded = P_state_trans[:, :, None]
    
    # The Full Joint Tensor
    Full_Tensor = P_state_expanded * P_emission_cond
    
    # Return as array of shape (n_symbols, n_states, n_states)
    matrix_list = []
    for k in range(n_symbols):
        matrix_list.append(Full_Tensor[:, :, k])
        
    return np.array(matrix_list)


def rank1(n_states: int = 3, n_symbols: int = 3, 
          state_decay: float = 1.0,
          fuzziness: float = 2.0) -> np.ndarray:
    """
    Generates M High-Rank symbol matrices where transitions are SHARED 
    between symbols using a continuous circular distribution (Von Mises).
    
    Args:
        n_states (int): Number of hidden states.
        n_symbols (int): Number of emission symbols.
        state_decay (float): Sharpness of next-state distribution (0=flat).
        fuzziness (float): Controls the overlap between symbols.
                           High value (>5) = Sharp separation (mostly 0s and 1s).
                           Low value (<1) = High overlap (transitions belong to many symbols).
                           
    Returns:
        np.ndarray: Array of shape (n_symbols, n_states, n_states) containing the symbol matrices.
    """
    
    # --- 1. Rank-1 Base Transition Matrix ---
    x = np.arange(n_states, dtype=np.float32)
    weights = np.exp(-state_decay * x)
    weights /= np.sum(weights)
    vec_states = weights  # Peak is always state 0 (A)
    P_state_trans = np.tile(vec_states, (n_states, 1))

    # --- 2. Create "Soft" Mixing Mask (Von Mises / Circular) ---
    
    # A. Calculate the 'Angle' of every transition (i -> j)
    rows, cols = np.indices((n_states, n_states))
    distances = (cols - rows) % n_states
    angles = (distances / n_states) * 2 * np.pi
    
    # B. Define 'Centers' for each symbol
    symbol_centers = np.linspace(0, 2 * np.pi, n_symbols, endpoint=False)
    
    # C. Calculate Weights using Circular Gaussian (Von Mises) logic
    mixing_weights = np.zeros((n_states, n_states, n_symbols))
    
    for k in range(n_symbols):
        center = symbol_centers[k]
        cos_dist = np.cos(angles - center)
        mixing_weights[:, :, k] = np.exp(fuzziness * cos_dist)
        
    # D. Normalize so mixing weights sum to 1.0 for every transition
    sum_weights = np.sum(mixing_weights, axis=2, keepdims=True)
    P_emission_cond = mixing_weights / sum_weights

    # --- 3. Combine ---
    P_state_expanded = P_state_trans[:, :, None]
    Full_Tensor = P_state_expanded * P_emission_cond
    
    matrix_list = []
    for k in range(n_symbols):
        matrix_list.append(Full_Tensor[:, :, k])
        
    return np.array(matrix_list)


def abc_ratio(ratio_a: int = 4, ratio_b: int = 2, ratio_c: int = 1, 
              ratio_d: int = 4, ratio_e: int = 2, ratio_f: int = 1, **kwargs) -> np.ndarray:
    """
    Generates 3 symbol-labeled transition matrices based on ratios.
    
    1. The Full Transition Matrix (sum) is Rank-1 with column probabilities proportional to [ratio_a, ratio_b, ratio_c].
    2. Each cell (i, j) is split into 3 parts with ratios [ratio_d, ratio_e, ratio_f].
    3. These parts are assigned to symbols cyclically to ensure the symbol matrices are High-Rank.
    
    Args:
        ratio_a, ratio_b, ratio_c (int): The ratios for destination state probabilities (1-50 each).
        ratio_d, ratio_e, ratio_f (int): The ratios for symbol emission splitting (1-50 each).
                         
    Returns:
        np.ndarray: Array of shape (3, 3, 3) containing three symbol matrices.
    """
    # 1. Normalize ABC to create destination state probability distribution
    total_abc = ratio_a + ratio_b + ratio_c
    base_probs = np.array([ratio_a, ratio_b, ratio_c]) / total_abc
    
    # 2. Normalize DEF for the symbol splitting
    total_def = ratio_d + ratio_e + ratio_f
    split_factors = np.array([ratio_d, ratio_e, ratio_f]) / total_def
    
    # Initialize the 3 symbol matrices
    matrices = [np.zeros((3, 3)) for _ in range(3)]
    
    # 3. Fill the matrices using Cyclic Rotation
    for row in range(3):
        for col in range(3):
            # The total probability of going to 'col' (regardless of row)
            cell_total_prob = base_probs[col]
            
            # The 3 pieces we need to distribute for this cell (using DEF ratios)
            pieces = cell_total_prob * split_factors
            
            # Cyclic Logic: (col - row) % 3
            shift = (col - row) % 3
            
            for k in range(3):
                piece_index = (k - shift) % 3
                matrices[k][row, col] = pieces[piece_index]
                
    return np.array(matrices)
def rank1_predefined(p_scale: float = 0.5, a: float = 0.5, 
                     ratio_a: float = 1, ratio_b: float = 1, ratio_c: float = 1, **kwargs) -> np.ndarray:
    """
    Generates 3 symbol matrices that sum to a specific Rank-1 transition matrix.
    Uses a P/N set distribution with permutations to ensure variety.
    
    Args:
        p_scale: Scale factor for P (0 to 1)
        a: Splitting factor (0 to 1)
        ratio_a, ratio_b, ratio_c: Ratios for the target rank-1 vector
    """
    # 1. Define Target Rank-1 Matrix T
    ratios = np.array([ratio_a, ratio_b, ratio_c], dtype=float)
    v = ratios / ratios.sum()
    T = np.tile(v, (3, 1))
    
    # 2. Calculate Max P and Max |N| constraints
    # P_max = min(v) / (3 * max(a, 1-a))
    max_split = max(a, 1-a)
    if max_split == 0: max_split = 1e-9
    p_max = (1/3) * np.min(v) / max_split
    
    # |N|_max = min(v) / 3
    n_max = (1/3) * np.min(v)
    
    # 3. Set actual P and N
    P = p_scale * p_max
    N = -1 * 1.0 * n_max # n_scale is always 1.0
    
    # 4. Create P-set and N-set (sum to 0)
    p_set = np.array([P, -(1-a)*P, -a*P])
    n_set = np.array([N, -(1-a)*N, -a*N])
    
    # 5. Distribute into Raw matrices with permutation
    raw_matrices = np.zeros((3, 3, 3)) # (symbol, state_from, state_to)
    
    for i in range(3):
        for j in range(3):
            vals = p_set if i == j else n_set
            
            # Shift values cyclically based on position (i+j)
            shift = (i + j) % 3
            vals_permuted = np.roll(vals, shift)
            
            raw_matrices[0, i, j] = vals_permuted[0]
            raw_matrices[1, i, j] = vals_permuted[1]
            raw_matrices[2, i, j] = vals_permuted[2]

    # 6. Add 1/3 of T to each symbol matrix
    final_matrices = np.zeros_like(raw_matrices)
    for k in range(3):
        final_matrices[k] = raw_matrices[k] + (1/3) * T
        
    return final_matrices

def rank1_xmas(scale_a: float = 0.9, scale_b: float = 0.9, 
               s1: float = 0.5, s2: float = 0.5, s3: float = 0.5, 
               ratio_a: float = 11, ratio_b: float = 7, ratio_c: float = 7, **kwargs) -> np.ndarray:
    """
    Rank-1 net transition matrix generation method.
    Ensures non-negative elements by scaling P and N relative to the target matrix T.
    """
    ratios = [ratio_a, ratio_b, ratio_c]
        
    # 1. Define the Rank 1 Matrix T
    # Normalize ratios to get the stationary vector v
    v = np.array(ratios, dtype=float)
    v = v / v.sum()
    
    # T has rows equal to v (standard rank-1 transition matrix)
    # T_ij = v_j
    T = np.tile(v, (3, 1))

    # Calculate max A
    # Constraints from a_p_set (involves s1, affects cols 0 and 1)
    # We have -A*s1 and -A*(1-s1) in columns 0 and 1
    max_A_p = min(v[0], v[1]) / (3 * max(s1, 1-s1))
    
    # Constraints from a_m_set (involves -A, affects cols 0 and 2)
    # We have -A in columns 0 and 2
    max_A_m = min(v[0], v[2]) / 3
    
    max_A = min(max_A_p, max_A_m)
    A = max_A * scale_a

    # Calculate max B
    # Constraints from b_set (involves s3, affects cols 1 and 2)
    # We have -B*s3 and -B*(1-s3) in columns 1 and 2
    max_B = min(v[1], v[2]) / (3 * max(s3, 1-s3))
    B = max_B * scale_b

    a_p_set = np.array([A, -A*(s1), -A*(1-s1)])
    a_m_set = np.array([-A, A*(s2), A*(1-s2)])
    b_set = np.array([B, -B*(s3), -B*(1-s3)])

    # 5. Distribute these into Raw matrices
    raw_matrices = np.zeros((3, 3, 3)) # (symbol, state_from, state_to)
    
    placements = [ [ [0,1,0], [1,1,1]],
                [ [0,0,0], [1,0,2], [1,2,0],[2,2,2]],
                [ [0,0,1], [2,1,2],[2,2,1]]
        ]

    value_sets = [a_p_set, a_m_set, b_set ]
    for vals, places in zip(value_sets, placements):
        for place in places:
            k, i, j = place
            vals_permuted = np.roll(vals, k)
            raw_matrices[:, i, j] = vals_permuted


    # 6. Add 1/3 of T to each symbol matrix
    final_matrices = np.zeros_like(raw_matrices)
    for k in range(3):
        final_matrices[k] = raw_matrices[k] + (1/3) * T

    return final_matrices


def fern(x: float = 0.5) -> np.ndarray:
    """Creates a transition matrix for the Fern Process.

    2 symbols, 3 states. Parameter x in [0, 1].
    """
    assert 0.0 <= x <= 1.0
    return np.array([
        [[0.3942, 0.00512, 0.0381],
         [0.0, 0.53, 0.0],
         [0.0, 0.326 * x, 0.554]],
        [[0.3358, 0.01088, 0.2159],
         [0.0, 0.0, 0.47],
         [0.12, 0.326 * (1 - x), 0.0]],
    ])


def smiley(
    curvature: float = 0.06,
    depth: float = 0.12,
    eye_height: float = 0.70,
    eye_spread: float = 0.14,
    eye_isolation: float = 0.85,
    **kwargs,
) -> np.ndarray:
    """
    Smiley face on the simplex via reset symbols + two-timescale smile dynamics.

    The smile is a parabolic U-curve from two timescales:
      - Slow A↔B exchange (curvature) → horizontal convergence
      - Fast C drain (depth) → vertical drop

    Reset symbols snap belief to fixed target distributions (eyes, smile corners).
    State-dependent emission (eye_isolation) prevents smile symbol from being
    emitted when belief is near eyes (high C), keeping eyes visually disjoint.

    6 symbols: smile, left_start, right_start, left_eye, right_eye, noise

    Args:
        curvature: A↔B exchange rate. Higher = arms converge faster (0.01-0.15)
        depth: C drain rate. Higher = deeper smile (0.05-0.25)
        eye_height: C component of eye positions. Higher = eyes further up (0.5-0.9)
        eye_spread: A-B asymmetry of eyes. Higher = eyes further apart (0.05-0.30)
        eye_isolation: 0=uniform emission, 1=smile never emits from C (0-1)
    """
    c_leak = min(curvature * 0.3, 0.03)  # small leak proportional to exchange

    # Smile transition matrix
    T_smile = np.array([
        [1 - curvature - c_leak, curvature, c_leak],
        [curvature, 1 - curvature - c_leak, c_leak],
        [depth, depth, 1 - 2 * depth],
    ])

    # Smile start positions (corners of the U)
    smile_c = eye_height - 0.10  # slightly below eyes
    smile_spread = eye_spread + 0.15  # wider than eyes
    left_start = np.array([0.5 * (1 - smile_c) + smile_spread,
                           0.5 * (1 - smile_c) - smile_spread,
                           smile_c])
    right_start = np.array([0.5 * (1 - smile_c) - smile_spread,
                            0.5 * (1 - smile_c) + smile_spread,
                            smile_c])
    left_start = np.clip(left_start, 0.01, None)
    right_start = np.clip(right_start, 0.01, None)
    left_start /= left_start.sum()
    right_start /= right_start.sum()

    # Eye positions
    eye_ab = (1.0 - eye_height) / 2.0
    left_eye = np.array([eye_ab + eye_spread, eye_ab - eye_spread, eye_height])
    right_eye = np.array([eye_ab - eye_spread, eye_ab + eye_spread, eye_height])
    left_eye = np.clip(left_eye, 0.01, None)
    right_eye = np.clip(right_eye, 0.01, None)
    left_eye /= left_eye.sum()
    right_eye /= right_eye.sum()

    # State-dependent emission weights
    base_smile = 0.80
    w_smile = np.array([base_smile, base_smile,
                        base_smile * (1.0 - eye_isolation)])
    remaining = 1.0 - w_smile
    w_per_reset = remaining / 5.0  # 4 resets + 1 noise

    # Smile symbol (state-dependent weight)
    T0 = T_smile * w_smile[:, None]

    # Reset symbols: T[k, i, j] = w_per_reset[i] * target[j]
    targets = [left_start, right_start, left_eye, right_eye]
    symbol_matrices = [T0]
    for target in targets:
        v = target / target.sum()
        Tk = w_per_reset[:, None] * np.tile(v, (3, 1))
        symbol_matrices.append(Tk)

    # Noise symbol (uniform reset)
    v_noise = np.ones(3) / 3.0
    T_noise = w_per_reset[:, None] * np.tile(v_noise, (3, 1))
    symbol_matrices.append(T_noise)

    T = np.array(symbol_matrices)

    # Verify
    net = T.sum(axis=0)
    assert np.allclose(net.sum(axis=1), 1.0), f"Row sums: {net.sum(axis=1)}"
    assert np.all(T >= -1e-10), f"Negative entries!"

    return T


def smiley_nested(
    curvature: float = 0.06,
    depth: float = 0.12,
    eye_height: float = 0.76,
    eye_size: float = 0.06,
    eye_separation: float = 0.16,
    **kwargs,
) -> np.ndarray:
    """
    Nested smiley: mouth U-curve + eye V-curves using two dynamics symbols.

    The mouth dynamics has attractor at low C (bottom of simplex).
    The eye dynamics has attractor at high C (top of simplex).
    Eyes are miniature V-curves that stay compact near the top.

    9 symbols: mouth_dyn, eye_dyn, 4 eye arm resets, 2 mouth resets, noise

    Args:
        curvature: Mouth A↔B exchange rate (0.02-0.12)
        depth: Mouth C drain rate (0.05-0.20)
        eye_height: C position for eyes, higher = further up (0.60-0.85)
        eye_size: Spread of each eye V, higher = wider eyes (0.02-0.12)
        eye_separation: Offset between eyes, higher = further apart (0.06-0.20)
    """
    leak = 0.02

    # --- Mouth dynamics: attractor at low C ---
    T_mouth = np.array([
        [1 - curvature - leak, curvature, leak],
        [curvature, 1 - curvature - leak, leak],
        [depth, depth, 1 - 2 * depth],
    ])

    # --- Eye dynamics: attractor at high C (leak >> drain) ---
    eye_alpha = 0.06
    eye_drain = 0.03
    # Derive eye_leak so attractor C ≈ eye_height
    # π_C = eye_leak / (eye_leak + 2*eye_drain), solve for eye_leak:
    target_pi_c = min(eye_height, 0.90)
    eye_leak = target_pi_c * 2 * eye_drain / (1 - target_pi_c)
    eye_leak = np.clip(eye_leak, 0.02, 0.40)

    T_eye = np.array([
        [1 - eye_alpha - eye_leak, eye_alpha, eye_leak],
        [eye_alpha, 1 - eye_alpha - eye_leak, eye_leak],
        [eye_drain, eye_drain, 1 - 2 * eye_drain],
    ])

    # --- Mouth starting positions (symmetric) ---
    mc = depth + 0.30  # mouth C derived from depth
    mc = np.clip(mc, 0.30, 0.55)
    mc_center = (1 - mc) / 2
    mouth_spread = 0.24
    mouth_L = np.array([mc_center + mouth_spread, mc_center - mouth_spread, mc])
    mouth_R = np.array([mc_center - mouth_spread, mc_center + mouth_spread, mc])
    for v in [mouth_L, mouth_R]:
        v[:] = np.clip(v, 0.01, None)
        v /= v.sum()

    # --- Eye starting positions (4 arms: 2 per eye) ---
    ec = eye_height
    ec_center = (1 - ec) / 2
    eoff = eye_separation
    espread = eye_size

    eye_L_outer = np.array([ec_center + eoff + espread,
                            ec_center - eoff - espread, ec])
    eye_L_inner = np.array([ec_center + eoff - espread,
                            ec_center - eoff + espread, ec])
    eye_R_inner = np.array([ec_center - eoff + espread,
                            ec_center + eoff - espread, ec])
    eye_R_outer = np.array([ec_center - eoff - espread,
                            ec_center + eoff + espread, ec])

    eye_targets = [eye_L_outer, eye_L_inner, eye_R_inner, eye_R_outer]
    for v in eye_targets:
        v[:] = np.clip(v, 0.01, None)
        v /= v.sum()

    # --- State-dependent emission weights ---
    # Mouth dyn: suppressed at high C (eye territory)
    # Eye dyn: suppressed at low C (mouth territory)
    isolation = 0.85
    w_mouth = np.array([0.35, 0.35, 0.35 * (1 - isolation)])
    w_eye = np.array([0.15 * (1 - isolation), 0.15 * (1 - isolation), 0.15])
    w_reset = (1.0 - w_mouth - w_eye) / 7.0  # 6 resets + 1 noise

    # --- Build symbol matrices ---
    symbols = []

    # Symbol 0: mouth dynamics
    symbols.append(T_mouth * w_mouth[:, None])

    # Symbol 1: eye dynamics
    symbols.append(T_eye * w_eye[:, None])

    # Symbols 2-3: mouth resets
    for target in [mouth_L, mouth_R]:
        symbols.append(w_reset[:, None] * np.tile(target, (3, 1)))

    # Symbols 4-7: eye arm resets
    for target in eye_targets:
        symbols.append(w_reset[:, None] * np.tile(target, (3, 1)))

    # Symbol 8: noise
    symbols.append(w_reset[:, None] * np.tile(np.ones(3) / 3.0, (3, 1)))

    T = np.array(symbols)

    # Verify
    net = T.sum(axis=0)
    assert np.allclose(net.sum(axis=1), 1.0, atol=1e-6), f"Row sums: {net.sum(axis=1)}"
    assert np.all(T >= -1e-10), f"Negative entries!"

    return T


def smiley_9state(
    curvature: float = 0.06,
    depth: float = 0.12,
    eye_height: float = 0.76,
    eye_size: float = 0.04,
    eye_separation: float = 0.08,
    **kwargs,
) -> np.ndarray:
    """
    9-state smiley face on the simplex via 3 independent triplet subspaces.

    States 0-2: mouth (A, B, C)
    States 3-5: left eye (A', B', C')
    States 6-8: right eye (A'', B'', C'')

    Marginalization: p̃_A = b_0+b_3+b_6, p̃_B = b_1+b_4+b_7, p̃_C = b_2+b_5+b_8

    9 symbols: mouth_dyn, left_eye_dyn, right_eye_dyn,
               mouth_reset_L, mouth_reset_R,
               left_eye_outer, left_eye_inner,
               right_eye_outer, right_eye_inner

    Args:
        curvature: Mouth A↔B exchange rate α (0.01-0.15)
        depth: Mouth C drain rate β (0.05-0.25)
        eye_height: C component of eye attractor (0.50-0.90)
        eye_size: Spread of each eye V-arm (0.01-0.10)
        eye_separation: Left-right offset of eyes (0.02-0.12)
    """
    # --- Mouth dynamics (3x3) ---
    # For a smile (U-shape), C drain must be FASTER than A↔B exchange
    # so C drops first then arms converge → parabolic concave-up shape.
    # Scale rates up so the curve is traced in ~3-4 dynamics steps.
    leak = 0.02
    rate_scale = 2.0
    exchange = curvature * rate_scale  # A↔B rate
    drain = depth * rate_scale         # C→A,B rate (faster → U-shape)

    T_mouth = np.array([
        [1 - exchange - leak, exchange, leak],
        [exchange, 1 - exchange - leak, leak],
        [drain, drain, 1 - 2 * drain],
    ])

    # --- Eye dynamics (two 3x3 matrices, detailed balance) ---
    c_e = eye_height
    half = (1.0 - c_e) / 2.0

    # Derive eye rates for V-curve (shape ≈ 1: equal eigenvalue decay)
    beta_e = 0.06
    alpha_e = min(beta_e / max(1.0 - c_e, 0.05), 0.5)

    # Left eye attractor: lean left (more A than B)
    a_le = max(0.01, half + eye_separation)
    b_le = max(0.01, half - eye_separation)
    pi_le = np.array([a_le, b_le, c_e])
    pi_le /= pi_le.sum()
    a_le, b_le, c_le = pi_le

    T_left_eye = np.array([
        [1 - alpha_e * b_le - beta_e * c_le, alpha_e * b_le, beta_e * c_le],
        [alpha_e * a_le, 1 - alpha_e * a_le - beta_e * c_le, beta_e * c_le],
        [beta_e * a_le, beta_e * b_le, 1 - beta_e * (1.0 - c_le)],
    ])

    # Right eye attractor: lean right (more B than A)
    a_re = max(0.01, half - eye_separation)
    b_re = max(0.01, half + eye_separation)
    pi_re = np.array([a_re, b_re, c_e])
    pi_re /= pi_re.sum()
    a_re, b_re, c_re = pi_re

    T_right_eye = np.array([
        [1 - alpha_e * b_re - beta_e * c_re, alpha_e * b_re, beta_e * c_re],
        [alpha_e * a_re, 1 - alpha_e * a_re - beta_e * c_re, beta_e * c_re],
        [beta_e * a_re, beta_e * b_re, 1 - beta_e * (1.0 - c_re)],
    ])

    # --- Mouth reset targets (arms of the U) ---
    mouth_start_c = 0.35
    mc = (1.0 - mouth_start_c) / 2.0
    ms = 0.20
    mouth_L = np.array([mc + ms, mc - ms, mouth_start_c])
    mouth_R = np.array([mc - ms, mc + ms, mouth_start_c])
    for v in [mouth_L, mouth_R]:
        v[:] = np.clip(v, 0.01, None)
        v /= v.sum()

    # --- Eye reset targets (arms of each V) ---
    eye_L_outer = np.array([max(0.01, half + eye_separation + eye_size),
                            max(0.01, half - eye_separation - eye_size), c_e])
    eye_L_inner = np.array([max(0.01, half + eye_separation - eye_size),
                            max(0.01, half - eye_separation + eye_size), c_e])
    eye_R_outer = np.array([max(0.01, half - eye_separation - eye_size),
                            max(0.01, half + eye_separation + eye_size), c_e])
    eye_R_inner = np.array([max(0.01, half - eye_separation + eye_size),
                            max(0.01, half + eye_separation - eye_size), c_e])
    for v in [eye_L_outer, eye_L_inner, eye_R_outer, eye_R_inner]:
        v[:] = np.clip(v, 0.01, None)
        v /= v.sum()

    # --- Emission weights ---
    weights = [0.40, 0.08, 0.08, 0.08, 0.08, 0.07, 0.07, 0.07, 0.07]

    # --- Build 9x9 symbol matrices ---
    symbols = []

    # Symbol 0: Mouth dynamics (block 0)
    S = np.eye(9)
    S[0:3, 0:3] = T_mouth
    symbols.append(S * weights[0])

    # Symbol 1: Left eye dynamics (block 1)
    S = np.eye(9)
    S[3:6, 3:6] = T_left_eye
    symbols.append(S * weights[1])

    # Symbol 2: Right eye dynamics (block 2)
    S = np.eye(9)
    S[6:9, 6:9] = T_right_eye
    symbols.append(S * weights[2])

    # Symbols 3-8: Reset symbols (rank-1, target specific subspace)
    reset_specs = [
        (mouth_L, 0),       # 3: Mouth reset L
        (mouth_R, 0),       # 4: Mouth reset R
        (eye_L_outer, 1),   # 5: Left eye reset outer
        (eye_L_inner, 1),   # 6: Left eye reset inner
        (eye_R_outer, 2),   # 7: Right eye reset outer
        (eye_R_inner, 2),   # 8: Right eye reset inner
    ]
    for idx, (target, block) in enumerate(reset_specs):
        R = np.zeros((9, 9))
        s = block * 3
        for i in range(9):
            R[i, s:s + 3] = target
        symbols.append(R * weights[3 + idx])

    T = np.array(symbols)

    # Verify
    net = T.sum(axis=0)
    assert np.allclose(net.sum(axis=1), 1.0, atol=1e-6), f"Row sums: {net.sum(axis=1)}"
    assert np.all(T >= -1e-10), f"Negative entries!"

    return T


def parabolic_curve(
    attr_height: float = 0.15,
    attr_lean: float = 0.0,
    start_height: float = 0.70,
    spread: float = 0.25,
    speed: float = 0.12,
    shape: float = 2.0,
    **kwargs,
) -> np.ndarray:
    """
    Parabolic curve toward a controllable attractor on the 3-simplex.

    4 symbols: dynamics, reset_left, reset_right, noise.

    The dynamics matrix uses a detailed-balance construction with
    eigenvalue structure controlled by (speed, shape). shape=2 gives
    parabolic curves; shape=1 gives straight-line convergence.
    """
    # --- Attractor ---
    c = np.clip(attr_height, 0.05, 0.95)
    lean = np.clip(attr_lean, -0.80, 0.80)
    remaining = 1.0 - c
    a = remaining * (1.0 - lean) / 2.0
    b = remaining * (1.0 + lean) / 2.0
    a = np.clip(a, 0.01, None)
    b = np.clip(b, 0.01, None)
    pi = np.array([a, b, c])
    pi /= pi.sum()
    a, b, c = pi

    # --- Dynamics eigenvalues ---
    beta = np.clip(speed, 0.02, 0.40)
    r = np.clip(shape, 0.5, 4.0)

    # lambda_C = 1 - beta
    # lambda_AB = lambda_C^(1/r)
    # alpha = (1 - beta*c - lambda_AB) / (1 - c)
    lambda_AB = (1.0 - beta) ** (1.0 / r)
    alpha = (1.0 - beta * c - lambda_AB) / (1.0 - c) if (1.0 - c) > 1e-9 else beta

    # Clip alpha so all T_dyn entries stay non-negative
    alpha_max = (1.0 - beta * c) / max(a, b)
    alpha = np.clip(alpha, 0.0, alpha_max - 1e-9)

    # --- Build dynamics matrix (detailed balance) ---
    T_dyn = np.array([
        [1 - alpha * b - beta * c, alpha * b,               beta * c],
        [alpha * a,                1 - alpha * a - beta * c, beta * c],
        [beta * a,                 beta * b,                 1 - beta * (1 - c)],
    ])

    # --- Start points (mirrored across A=B line) ---
    sc = np.clip(start_height, 0.05, 0.95)
    sp = np.clip(spread, 0.0, 0.45)
    start_L = np.array([(1 - sc) * (0.5 + sp), (1 - sc) * (0.5 - sp), sc])
    start_R = np.array([(1 - sc) * (0.5 - sp), (1 - sc) * (0.5 + sp), sc])
    for v in [start_L, start_R]:
        v[:] = np.clip(v, 0.01, None)
        v /= v.sum()

    # --- Emission weights (state-independent) ---
    w_dyn = 0.80
    w_reset = 0.10

    symbols = [
        T_dyn * w_dyn,
        w_reset * np.tile(start_L, (3, 1)),
        w_reset * np.tile(start_R, (3, 1)),
    ]

    T = np.array(symbols)
    net = T.sum(axis=0)
    assert np.allclose(net.sum(axis=1), 1.0, atol=1e-6), f"Row sums: {net.sum(axis=1)}"
    assert np.all(T >= -1e-10), f"Negative entries!"

    return T
