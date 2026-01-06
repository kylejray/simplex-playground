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

    def generate_systematic(self, max_len=10, max_count=1000):
        """
        Systematically generate all possible observation sequences of length max_len,
        limited by max_count.
        """
        import itertools
        words = []
        count = 0
        
        # Generate words of exactly max_len
        for seq in itertools.product(range(self.num_symbols), repeat=max_len):
            if count >= max_count:
                break
            
            word = "".join(str(s) for s in seq)
            words.append(word)
            count += 1
        
        batch_size = len(words)
        if batch_size == 0:
            return [], [], [], self.initial_state.tolist()
            
        belief_states_list = []
        
        # Standard belief update
        for word in words:
            seq_beliefs = []
            current_belief = self.initial_state
            
            for char in word:
                symbol = int(char)
                T = self.transition_matrices[symbol]
                
                # Update
                next_belief = current_belief @ T
                
                # Normalize
                norm = np.sum(next_belief)
                if norm > 0:
                    next_belief /= norm
                else:
                    next_belief = self.initial_state
                
                current_belief = next_belief
                seq_beliefs.append(current_belief.tolist())
            
            belief_states_list.append(seq_beliefs)
            
        # Constrained beliefs
        constrained_beliefs_list = []
        
        # Precompute local posteriors
        local_posteriors = np.zeros((self.num_symbols, self.num_states))
        for k in range(self.num_symbols):
            joint = self.initial_state @ self.transition_matrices[k]
            norm = np.sum(joint)
            if norm > 0:
                local_posteriors[k] = joint / norm
            else:
                local_posteriors[k] = self.initial_state

        # Precompute powers of T
        max_seq_len = len(words[-1]) if words else 0
        T_powers = [np.eye(self.num_states)]
        curr_T = np.eye(self.num_states)
        for _ in range(max_seq_len):
            curr_T = curr_T @ self.state_transition_matrix
            T_powers.append(curr_T)
            
        for word in words:
            seq_obs = [int(c) for c in word]
            seq_len = len(seq_obs)
            seq_constrained = []
            
            for d in range(seq_len):
                r_d = self.initial_state.copy()
                for s in range(d + 1):
                    z_s = seq_obs[s]
                    posterior = local_posteriors[z_s]
                    propagated = posterior @ T_powers[d - s]
                    contribution = propagated - self.initial_state
                    r_d += contribution
                
                r_d = np.maximum(r_d, 0)
                norm = np.sum(r_d)
                if norm > 0:
                    r_d /= norm
                else:
                    r_d = self.initial_state.copy()
                seq_constrained.append(r_d.tolist())
            
            constrained_beliefs_list.append(seq_constrained)
            
        return words, belief_states_list, constrained_beliefs_list, self.initial_state.tolist()

    def compute_excess_entropy(self, max_block_length=20, max_samples=10000):
        """
        Compute excess entropy using a two-phase approach:
        1. Exact enumeration for small L where num_sequences <= max_samples
        2. Monte Carlo sampling for large L where enumeration is too expensive
        
        Args:
            max_block_length: Maximum sequence length L to compute
            max_samples: Maximum samples per length (also threshold for switching to MC)
            
        Returns:
            dict with keys:
                - excess_entropy: Estimated E value
                - entropy_rate: Estimated h_mu
                - block_entropies: List of (L, H_L, method) tuples
                - max_samples: Number of max samples used
                - mc_start_L: Block length where Monte Carlo begins (None if all enumerated)
        """
        block_entropies = [(0, 0.0, 'exact')]  # H(X_0^0) = 0 by definition
        mc_start_L = None
        
        for L in range(1, max_block_length + 1):
            num_sequences = self.num_symbols ** L
            
            if num_sequences <= max_samples:
                # Phase 1: Exact enumeration
                H_L = self._compute_block_entropy_exact(L)
                block_entropies.append((L, H_L, 'exact'))
            else:
                # Phase 2: Monte Carlo sampling
                if mc_start_L is None:
                    mc_start_L = L
                H_L = self._compute_block_entropy_mc(L, max_samples)
                block_entropies.append((L, H_L, 'mc'))
                
        return self._finalize_excess_entropy(block_entropies, max_samples, mc_start_L)
    
    def _compute_block_entropy_exact(self, L):
        """
        Compute exact block entropy H(X_0^L) by enumerating all sequences.
        """
        import itertools
        
        total_entropy = 0.0
        
        for seq in itertools.product(range(self.num_symbols), repeat=L):
            # Compute P(sequence) using forward algorithm
            prob = self._sequence_probability(seq)
            
            if prob > 1e-300:
                total_entropy -= prob * np.log2(prob)
        
        return total_entropy
    
    def _sequence_probability(self, seq):
        """
        Compute exact probability of a symbol sequence P(x_1, ..., x_L).
        """
        # Start from stationary distribution
        state_probs = self.initial_state.copy()
        
        for symbol in seq:
            # P(x_t, s_t | x_1..x_{t-1}) = sum_s P(s_{t-1}=s) * T[symbol, s, :]
            # Then marginalize over s_t to get P(x_t | x_1..x_{t-1})
            joint = state_probs @ self.transition_matrices[symbol]
            prob_symbol = np.sum(joint)
            
            if prob_symbol < 1e-300:
                return 0.0
            
            # Update state distribution
            state_probs = joint / prob_symbol
        
        # The sequence probability is product of conditional probabilities
        # which we can compute by tracking the unnormalized forward variable
        alpha = self.initial_state.copy()
        for symbol in seq:
            alpha = alpha @ self.transition_matrices[symbol]
        
        return np.sum(alpha)
    
    def _compute_block_entropy_mc(self, L, n_samples):
        """
        Estimate block entropy H(X_0^L) using Monte Carlo sampling.
        """
        log_probs = []
        
        for _ in range(n_samples):
            current_state_probs = self.initial_state.copy()
            log_prob = 0.0
            
            for t in range(L):
                # Compute joint distribution over (symbol, next_state)
                joint = np.einsum('s,kso->ko', current_state_probs, self.transition_matrices)
                
                # Sample from joint distribution
                flat_joint = joint.flatten()
                flat_joint = np.maximum(flat_joint, 0)
                flat_sum = np.sum(flat_joint)
                if flat_sum < 1e-300:
                    log_prob = -np.inf
                    break
                flat_joint = flat_joint / flat_sum
                idx = np.random.choice(len(flat_joint), p=flat_joint)
                symbol = idx // self.num_states
                
                # Compute probability of this symbol
                symbol_prob = np.sum(joint[symbol])
                if symbol_prob < 1e-300:
                    log_prob = -np.inf
                    break
                    
                log_prob += np.log2(symbol_prob)
                
                # Update state distribution
                current_state_probs = joint[symbol] / symbol_prob
            
            if np.isfinite(log_prob):
                log_probs.append(log_prob)
        
        if len(log_probs) > 0:
            return -np.mean(log_probs)
        return 0.0
    
    def _compute_lower_bound(self, H_L, H_L_minus_1, L):
        """
        Compute lower bound estimate on excess entropy using consecutive block entropies.

        Uses the asymptotic formula H(L) ≈ h*L + E:
        - h ≈ H(L) - H(L-1)
        - E_lower ≈ H(L) - h*L

        Returns (E_lower, h_estimate)
        """
        h_estimate = H_L - H_L_minus_1
        E_lower = H_L - h_estimate * L
        return E_lower, h_estimate

    def _finalize_excess_entropy(self, block_entropies, max_samples, mc_start_L):

        """
        Compute final excess entropy and entropy rate from block entropies.
        Uses linear fit on last 2 points for h_mu: H(L) = h_mu * L + E
        """
        n_points = len(block_entropies)
        
        if n_points >= 2:
            # Use only the LAST TWO block entropy points for linear fit
            # H(L) = h_mu * L + E
            # Slope h_mu = (H_2 - H_1) / (L_2 - L_1)
            L1, H1, _ = block_entropies[-2]
            L2, H2, _ = block_entropies[-1]
            
            h_mu = (H2 - H1) / (L2 - L1) if L2 != L1 else 0
            
            # Compute E as y-intercept: E = H - h_mu * L (using last point)
            E = H2 - h_mu * L2
        elif n_points == 1:
            L, H, _ = block_entropies[0]
            h_mu = H / L if L > 0 else 0
            E = 0
        else:
            h_mu = 0
            E = 0
        
        # Convert to output format (L, H, method)
        output_entropies = [(L, H, method) for L, H, method in block_entropies]
        
        return {
            'excess_entropy': float(E),
            'entropy_rate': float(h_mu),
            'block_entropies': output_entropies,
            'max_samples': max_samples,
            'mc_start_L': mc_start_L
        }
