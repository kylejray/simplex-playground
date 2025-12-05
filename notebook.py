import os
import simplexity
import jax
import jax.numpy as jnp
from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel
from simplexity.generative_processes.torch_generator import generate_data_batch
from simplexity.generative_processes.transition_matrices import mess3
import matrices
import yaml


from visualization import plotly_simplex_vs_raw, plotly_pca_explained_variance

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

print(config)
def get_matrices(config):
    key = config["matrix_key"]
    kwargs = config.get("matrix_kwargs", {})

    if key == "simple_constrained":
        return matrices.simple_constrained(**kwargs)
    if key == 'santa':
        return matrices.santa(**kwargs)
    if key == 'mess3':
        return mess3(**kwargs)

generator = HiddenMarkovModel(get_matrices(config))

key = jax.random.PRNGKey(int(1E9))

batch_size = config["batch_size"]
sequence_len = config["sequence_len"]
batch_keys = jax.random.split(key, batch_size)

fig_directory =f"./figs/"
os.makedirs(fig_directory, exist_ok=True)

def main():
    true_states, observations = generator.generate(jnp.tile(generator.initial_state, (batch_size, 1)), batch_keys, sequence_len, True)

    _ = plotly_simplex_vs_raw(true_states, dir=fig_directory)

if __name__ == "__main__":
    main()