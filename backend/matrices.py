import numpy as np

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


def abc_ratio(a: int = 4, b: int = 2, c: int = 1, d: int = 4, e: int = 2, f: int = 1) -> np.ndarray:
    """
    Generates 3 symbol-labeled transition matrices based on ratios.
    
    1. The Full Transition Matrix (sum) is Rank-1 with column probabilities proportional to [a, b, c].
    2. Each cell (i, j) is split into 3 parts with ratios [d, e, f].
    3. These parts are assigned to symbols cyclically to ensure the symbol matrices are High-Rank.
    
    Args:
        a, b, c (int): The ratios for destination state probabilities (1-50 each).
        d, e, f (int): The ratios for symbol emission splitting (1-50 each).
                         
    Returns:
        np.ndarray: Array of shape (3, 3, 3) containing three symbol matrices.
    """
    # 1. Normalize ABC to create destination state probability distribution
    total_abc = a + b + c
    base_probs = np.array([a, b, c]) / total_abc
    
    # 2. Normalize DEF for the symbol splitting
    total_def = d + e + f
    split_factors = np.array([d, e, f]) / total_def
    
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
               ratios: list = None) -> np.ndarray:
    """
    Rank-1 net transition matrix generation method.
    Ensures non-negative elements by scaling P and N relative to the target matrix T.
    """
    if ratios is None:
        ratios = [11, 7, 7]
        
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
