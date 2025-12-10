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