import numpy as np

class HiddenMarkovModel:
    def __init__(self, transition_matrices):
        """
        Initialize the HMM with transition matrices.
        
        Args:
            transition_matrices: numpy array of shape (num_symbols, num_states, num_states)
                                 T[k, i, j] = P(S_{t+1}=j, O_{t+1}=k | S_t=i)
        """
        self.transition_matrices = np.array(transition_matrices)
        self.num_symbols, self.num_states, _ = self.transition_matrices.shape
        
        # Calculate marginal transition matrix P(S_{t+1}|S_t)
        # Sum over symbols (axis 0)
        self.state_transition_matrix = np.sum(self.transition_matrices, axis=0)
        
        # Calculate initial state (stationary distribution)
        self.initial_state = self._get_stationary_state()

    def _get_stationary_state(self):
        """Compute the stationary distribution of the state transition matrix."""
        # We want v such that v @ T = v
        # This corresponds to eigenvector of T.T with eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(self.state_transition_matrix.T)
        
        # Find index of eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1))
        
        stationary = np.real(eigenvectors[:, idx])
        return stationary / np.sum(stationary)

    def generate(self, batch_size, sequence_len):
        """
        Generate sequences of observations and hidden states.
        
        Returns:
            words: list of strings (observations)
            belief_states: list of lists of arrays (belief states for each step)
        """
        # Initialize states based on stationary distribution
        current_states = np.random.choice(
            self.num_states, 
            size=batch_size, 
            p=self.initial_state
        )
        
        all_observations = np.zeros((batch_size, sequence_len), dtype=int)
        
        # Pre-compute flattened transition probabilities for efficient sampling
        # We want to sample (symbol, next_state) given current_state
        # T[k, i, j] = P(symbol=k, next=j | current=i)
        # We want a matrix where row i contains all probabilities for current state i
        
        # Transpose to (current_state, symbol, next_state) -> (num_states, num_symbols, num_states)
        T_ordered = self.transition_matrices.transpose(1, 0, 2)
        
        # Reshape to (num_states, num_symbols * num_states)
        flat_probs = T_ordered.reshape(self.num_states, -1)
        
        for t in range(sequence_len):
            # Vectorized sampling
            # 1. Get probabilities for the current state of each batch item
            # probs shape: (batch_size, num_symbols * num_states)
            probs = flat_probs[current_states]
            
            # 2. Compute cumulative probabilities
            cum_probs = np.cumsum(probs, axis=1)
            
            # 3. Normalize (avoid floating point issues)
            # Divide by the last column (sum) to ensure it ends at 1.0
            sums = cum_probs[:, -1:]
            # Avoid division by zero
            sums[sums == 0] = 1.0
            cum_probs /= sums
            
            # 4. Sample
            rand_vals = np.random.random((batch_size, 1))
            
            # This is the vectorized equivalent of searchsorted
            # (cum_probs < rand_vals) creates a boolean matrix where True means "not reached yet"
            # Summing gives the index of the first False (where cum_prob >= rand_val)
            sample_idx = (cum_probs < rand_vals).sum(axis=1)
            
            # 5. Decode index
            symbol = sample_idx // self.num_states
            next_state = sample_idx % self.num_states
            
            all_observations[:, t] = symbol
            current_states = next_state

        # Convert observations to words
        words = []
        for i in range(batch_size):
            word = "".join(str(o) for o in all_observations[i])
            words.append(word)
            
        # Calculate belief states for all sequences
        # We do this separately because we only need it for visualization
        num_display = batch_size
        
        # Vectorized belief update
        # Initialize beliefs: (num_display, num_states)
        current_beliefs = np.tile(self.initial_state, (num_display, 1))
        
        # Store history: list of (num_display, num_states) arrays
        belief_history_steps = []
        
        for t in range(sequence_len):
            # Get symbols for this step: (num_display,)
            syms = all_observations[:num_display, t]
            
            # Get transition matrices for these symbols: (num_display, num_states, num_states)
            Ts = self.transition_matrices[syms]
            
            # Update: b_new = b_old @ T
            # b_old: (B, S), T: (B, S, S) -> (B, S)
            # einsum 'bi, bij -> bj'
            next_beliefs = np.einsum('bi,bij->bj', current_beliefs, Ts)
            
            # Normalize
            norms = np.sum(next_beliefs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0 # Avoid div by zero
            next_beliefs /= norms
            
            current_beliefs = next_beliefs
            belief_history_steps.append(current_beliefs)
            
        # Transpose belief history to be list of sequences
        # Currently: list of T arrays of shape (B, S)
        # Want: list of B lists of T arrays of shape (S,)
        
        belief_states_list = []
        for i in range(num_display):
            seq_beliefs = []
            for t in range(sequence_len):
                seq_beliefs.append(belief_history_steps[t][i].tolist())
            belief_states_list.append(seq_beliefs)
            
        # Calculate constrained belief states (Equation 5 from paper)
        constrained_beliefs_list = self._generate_constrained_beliefs(all_observations[:num_display])
            
        return words, belief_states_list, constrained_beliefs_list, self.initial_state.tolist()

    def _generate_constrained_beliefs(self, observations):
        """
        Generate constrained belief states using the additive update rule.
        r_d = pi + sum_{s=1}^d ( pi @ T_{|z_s} @ T^{d-s} - pi )
        """
        batch_size, seq_len = observations.shape
        constrained_beliefs = []
        
        # Precompute local posteriors: pi @ T_{|z} (normalized)
        # T_{|z} here refers to the transition matrix conditioned on observing z
        # In our notation, self.transition_matrices[z] is P(S', O=z | S)
        # So pi @ T[z] gives the joint distribution P(S', O=z)
        # We normalize this to get P(S' | O=z) which represents the belief after 1 step observing z
        
        local_posteriors = np.zeros((self.num_symbols, self.num_states))
        for k in range(self.num_symbols):
            joint = self.initial_state @ self.transition_matrices[k]
            norm = np.sum(joint)
            if norm > 0:
                local_posteriors[k] = joint / norm
            else:
                local_posteriors[k] = self.initial_state

        # Precompute powers of the state transition matrix T^k
        # T_powers[k] = T^k
        T_powers = [np.eye(self.num_states)] # T^0
        curr_T = np.eye(self.num_states)
        for _ in range(seq_len):
            curr_T = curr_T @ self.state_transition_matrix
            T_powers.append(curr_T)
            
        # Compute beliefs for each sequence
        for b in range(batch_size):
            seq_beliefs = []
            seq_obs = observations[b]
            
            for d in range(seq_len):
                # d is 0-indexed, so it corresponds to position d+1 in 1-based indexing
                # The sum is from s=1 to d+1
                
                # Start with pi
                r_d = self.initial_state.copy()
                
                # Add contributions from history
                for s in range(d + 1):
                    # s is 0-indexed index of observation
                    z_s = seq_obs[s]
                    
                    # Distance from s to d is (d - s) steps
                    # We apply T^(d-s)
                    
                    # Contribution: (Posterior_s @ T^(d-s)) - pi
                    posterior = local_posteriors[z_s]
                    propagated = posterior @ T_powers[d - s]
                    
                    contribution = propagated - self.initial_state
                    r_d += contribution
                
                # Normalize to simplex
                # The additive approximation doesn't guarantee valid probability distribution
                # So we clip and normalize
                r_d = np.maximum(r_d, 0)
                norm = np.sum(r_d)
                if norm > 0:
                    r_d /= norm
                else:
                    r_d = self.initial_state.copy()
                    
                seq_beliefs.append(r_d.tolist())
            constrained_beliefs.append(seq_beliefs)
            
        return constrained_beliefs
