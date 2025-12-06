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

def left_right_mix() -> np.ndarray:
    """
    Returns the 'left_right_mix' preset matrices.
    Symbol 0: Deterministic cycle A->B->C->A
    Symbol 1: Probabilistic reverse cycle A->C->B->A
    Symbol 2: Uniform noise
    Returns shape (3, 3, 3).
    """
    # Symbol 0: A->B->C->A (Deterministic cycle)
    # Weight = 1.0
    T0 = np.array([
        [0.0, 1.0, 0.0],  # A -> B
        [0.0, 0.0, 1.0],  # B -> C
        [1.0, 0.0, 0.0]   # C -> A
    ])

    # Symbol 1: A->C->B->A (Opposite deterministic cycle)
    # Weight = 0.6 (mostly), one is 0.3 to break symmetry
    T1 = np.array([
        [0.0, 0.0, 0.3],   # A -> C
        [0.6, 0.0, 0.0],   # B -> A
        [0.0, 0.6, 0.0]    # C -> B
    ])

    # Symbol 2: Uniform over all transitions
    # Adjusted to compensate for T1 asymmetry so that Total(A)=Total(B)=Total(C)
    # T0 weights: 1.0, 1.0, 1.0
    # T1 weights: 0.3, 0.6, 0.6
    # Target Total: 1.7
    
    val_A = 0.4 / 3.0
    val_others = 0.1 / 3.0
    
    T2 = np.array([
        [val_A, val_A, val_A],          # From A (needs more weight to balance T1's 0.3 vs 0.6)
        [val_others, val_others, val_others], # From B
        [val_others, val_others, val_others]  # From C
    ])

    return np.array([T0, T1, T2])