import numpy as np

def test_rank1_construction(p_scale, n_scale, a, ratios):
    """
    Workshopping a new rank-1 net transition matrix generation method.
    Ensures non-negative elements by scaling P and N relative to the target matrix T.
    
    Args:
        p_scale: Scale factor for P (0 to 1), relative to max possible P
        n_scale: Scale factor for N (0 to 1), relative to max possible |N|
        a: Splitting factor (0 to 1)
        ratios: List of 3 numbers defining the rank-1 vector ratios
    """
    print(f"--- Testing Rank 1 Construction (Non-negative) ---")
    print(f"Params: p_scale={p_scale}, n_scale={n_scale}, a={a}, ratios={ratios}")
    
    # 1. Define the Rank 1 Matrix T
    # Normalize ratios to get the stationary vector v
    v = np.array(ratios, dtype=float)
    v = v / v.sum()
    print(f"Target Vector v: {v}")
    
    # T has rows equal to v (standard rank-1 transition matrix)
    # T_ij = v_j
    T = np.tile(v, (3, 1))
    print("Target Net Matrix T:")
    print(T)
    
    # 2. Calculate Max P and Max |N| to ensure non-negativity
    # We add (1/3)T to the raw matrices.
    # So Raw_ijk >= -(1/3)T_ij
    
    # For Diagonal elements (using P-set):
    # P-set = {P, -(1-a)P, -aP}
    # The negative values are -(1-a)P and -aP.
    # We need: -(1-a)P >= -(1/3)T_ii  =>  P <= (1/3)T_ii / (1-a)
    # And:     -aP     >= -(1/3)T_ii  =>  P <= (1/3)T_ii / a
    # So P_max_i = (1/3)T_ii / max(a, 1-a)
    # Global P_max is the minimum of these across all i.
    
    # Note: T_ii = v_i
    max_split = max(a, 1-a)
    if max_split == 0: max_split = 1e-9 # avoid div by zero
    
    p_max_possible = (1/3) * np.min(v) / max_split
    
    # For Off-Diagonal elements (using N-set):
    # N-set = {N, -(1-a)N, -aN} where N is negative.
    # The negative value is N.
    # We need: N >= -(1/3)T_ij
    # Since N is negative, |N| <= (1/3)T_ij
    # Global |N|_max is the minimum of (1/3)T_ij across all off-diagonals.
    # Since T_ij = v_j, and we cover all columns j in off-diagonals (e.g. 0->1, 1->0),
    # this is just min(v) again.
    
    n_abs_max_possible = (1/3) * np.min(v)
    
    print(f"Calculated Constraints:")
    print(f"  P_max_possible: {p_max_possible:.6f}")
    print(f"  |N|_max_possible: {n_abs_max_possible:.6f}")
    
    # 3. Set actual P and N based on scales
    P = p_scale * p_max_possible
    N = -1 * n_scale * n_abs_max_possible
    
    print(f"Derived Values:")
    print(f"  P: {P:.6f}")
    print(f"  N: {N:.6f}")

    # 4. Create the 2 sets of numbers (P-set and N-set)
    # P-set sums to 0: P, -(1-a)P, -aP
    p_set = np.array([P, -(1-a)*P, -a*P])
    
    # N-set sums to 0: N, -(1-a)N, -aN
    n_set = np.array([N, -(1-a)*N, -a*N])
    
    print(f"P-set: {p_set}")
    print(f"N-set: {n_set}")
    
    # 5. Distribute these into Raw matrices
    raw_matrices = np.zeros((3, 3, 3)) # (symbol, state_from, state_to)
    
    for i in range(3):
        for j in range(3):
            if i == j:
                vals = p_set
            else:
                vals = n_set
            
            # Assign to symbols 0, 1, 2
            raw_matrices[0, i, j] = vals[0]
            raw_matrices[1, i, j] = vals[1]
            raw_matrices[2, i, j] = vals[2]

    # 6. Add 1/3 of T to each symbol matrix
    final_matrices = np.zeros_like(raw_matrices)
    for k in range(3):
        final_matrices[k] = raw_matrices[k] + (1/3) * T
        
    # 7. Check results
    print("\nFinal Matrices:")
    min_val = np.min(final_matrices)
    print(f"Minimum value in matrices: {min_val} (Should be >= -epsilon)")
    
    for k in range(3):
        print(f"Symbol {k}:")
        print(final_matrices[k])
        
    # Check net transition matrix
    net_final = np.sum(final_matrices, axis=0)
    print("\nNet Final Matrix (should be T):")
    print(net_final)
    print(f"Matches T? {np.allclose(net_final, T)}")
    
    if min_val < -1e-9:
        print("WARNING: Negative values detected!")

if __name__ == "__main__":
    # Test with full scales
    test_rank1_construction(p_scale=1.0, n_scale=1.0, a=0.5, ratios=[2, 1, 1])
    print("\n" + "="*30 + "\n")
    # Test with uneven ratios
    test_rank1_construction(p_scale=0.5, n_scale=0.2, a=0.5, ratios=[1, 1, 1])
