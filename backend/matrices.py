import jax
import jax.numpy as jnp

def simple_constrained(x: float, y: float) -> jnp.ndarray:
    """
    Creates a set of 2 transition matrices based on parameters x and y.
    Returns shape (2, 3, 3).
    """
    T0 = jnp.array(
        [
            [x, x, 1-y],
            [y, y, 1-y],
            [y, 1-y, x],
        ])
    
    T1 = 1 - T0

    return jnp.array([T0, T1]) / 3

def fractal_constrained(x: float, y: float) -> jnp.ndarray:
    """
    Creates a set of 3 transition matrices based on parameters x and y.
    Returns shape (3, 3, 3).
    """
    x9 = x * 9
    y9 = y * 9
    normalization_factor = 9 * 2.4

    T0 = jnp.array(
        [
            [1, 1+y9, 1+x9],
            [1, 1, 1],
            [1+y9, 1, 1+x9],
        ])
    
    T1 = jnp.array(
        [
            [1+x9, 1+y9, 1],
            [1+x9, 1, 1+y9],
            [1, 1, 1],
        ])

    T2 = jnp.array(
        [
            [1, 1, 1],
            [1, 1+x9, 1+y9],
            [1+y9, 1+x9, 1],
        ])

    return jnp.array([T0, T1, T2]) / normalization_factor

def santa() -> jnp.ndarray:
    """
    Returns the 'santa' preset matrices.
    Returns shape (3, 3, 3).
    """
    v = .0611111
    w = .211111
    x = .0111111
    y = .161111
    z = .311111

    T0 = jnp.array(
        [
            [x, w, y],
            [z, v, x],
            [y, x, y],
        ])
    
    T1 = jnp.array(
        [
            [y, v, x],
            [x, w, x],
            [x, x, y],
        ])

    T2 = jnp.array(
        [
            [y, v, y],
            [x, v, z],
            [y, z, x],
        ])

    return jnp.array([T0, T1, T2])

def left_right_mix() -> jnp.ndarray:
    """
    Returns the 'left_right_mix' preset matrices.
    Symbol 0: Deterministic cycle A->B->C->A
    Symbol 1: Probabilistic reverse cycle A->C->B->A
    Symbol 2: Uniform noise
    Returns shape (3, 3, 3).
    """
    # Symbol 0: A->B->C->A (Deterministic cycle)
    # Weight = 1.0
    T0 = jnp.array([
        [0.0, 1.0, 0.0],  # A -> B
        [0.0, 0.0, 1.0],  # B -> C
        [1.0, 0.0, 0.0]   # C -> A
    ])

    # Symbol 1: A->C->B->A (Opposite deterministic cycle)
    # Weight = 0.6 (mostly), one is 0.3 to break symmetry
    T1 = jnp.array([
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
    
    T2 = jnp.array([
        [val_A, val_A, val_A],          # From A (needs more weight to balance T1's 0.3 vs 0.6)
        [val_others, val_others, val_others], # From B
        [val_others, val_others, val_others]  # From C
    ])

    return jnp.array([T0, T1, T2])