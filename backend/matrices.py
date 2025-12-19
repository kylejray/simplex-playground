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