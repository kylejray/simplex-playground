import numpy as np

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
    # Normalize ABC, DEF
    total_abc = a + b + c
    base_probs = np.array([a, b, c]) / total_abc
    
    total_def = d + e + f
    split_factors = np.array([d, e, f]) / total_def
    
    
    matrices = [np.zeros((3, 3)) for _ in range(3)]
    
    #Fill the matrices using Cyclic Rotation
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


def rank1(n_states: int = 3, n_symbols: int = 3, 
          state_decay: float = 1.0,
          fuzziness: float = 2.0) -> np.ndarray:
    """
    Generates M High-Rank symbol matrices from a Rank-1 full transition matrix.
    
    Args:
        n_states (int): Number of hidden states.
        n_symbols (int): Number of emission symbols.
        state_decay (float): Sharpness of next-state distribution (0=flat).
        fuzziness (float): Controls the overlap between symbols.
                           High value (>5) = Sharp separation (symbols dont share transitions much)
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

