import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def simplex_to_polygon_coords(simplex_coords, vertices=None):
    if vertices is None:
        vertices = regular_polygon(simplex_coords.shape[-1])

    if not all(np.isclose(np.sum(simplex_coords, axis=-1), 1)):
        print("Warning: Simplex coordinates do not sum to 1. Normalizing.")
        simplex_coords = simplex_coords / np.sum(simplex_coords, axis=-1, keepdims=True)

    return simplex_coords @ np.array(vertices), vertices

def regular_polygon(n, start_point=[0,0], start_vector=[1,0]):
    angle = 2*np.pi / n
    start_point = np.array(start_point)
    base_vector = np.array(start_vector)

    S, C = np.sin(angle), np.cos(angle)
    R = np.array([[C,-S],[S,C]])
    vertices = [start_point]
    for i in range(n-1):
        vertices.append( vertices[-1] +  base_vector)
        base_vector = R @ base_vector
    x_min, y_min = np.min(vertices, axis=0)

    return [ [item[0]-x_min, item[1]-y_min] for item in vertices ]


def get_rgb_string(states, state_min=None, state_max=None):

    if states.shape[-1] > 3:
        print("Performing PCA to reduce to 3 dimensions for RGB mapping.")
        pca = PCA(n_components=3)
        states = pca.fit_transform(states)
        print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

    if state_min is None:
        state_min = states.min(axis=0)
    
    if state_max is None:
        state_max = states.max(axis=0)

    states -= state_min
    states /= (state_max-state_min)
    if len(states.shape) == 1:
        states = [states]
    color_strings = [ f'rgba({int(item[0])}, {int(item[1])}, {int(item[2])}, 1.)' for item in 255*states]
    return color_strings
