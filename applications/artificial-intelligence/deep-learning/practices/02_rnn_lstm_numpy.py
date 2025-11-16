"""
Recurrent Neural Networks (RNN) and LSTM Basics

Problem: Traditional neural networks can't handle sequential data
Goal: Understand how RNNs process sequences with memory

Core Concepts:
1. Recurrence: Network has memory of past inputs
2. Hidden State: Carries information through time
3. Vanishing Gradient: Why RNN struggles with long sequences
4. LSTM: Solves long-term dependency problem with gates
5. Applications: Text, time series, user behavior sequences

RNN vs Feedforward NN:
- Feedforward: Each input processed independently
- RNN: Outputs depend on current input AND previous inputs
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# ==================== 1. Simple RNN (Vanilla RNN) ====================
class SimpleRNN:
    """
    Simplest RNN implementation (one layer)

    ====================================================================
    🔑 What is RNN?
    ====================================================================

    RNN = Neural network with a loop, allowing information to persist

    Traditional NN:        RNN:
    Input → Hidden → Output    Input_t ──→ Hidden_t → Output_t
                                  ↑         ↓
                                  └─────────┘ (recurrence)

    At each time step t:
        h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
        y_t = W_hy * h_t + b_y

    where:
        h_t: hidden state at time t (memory)
        x_t: input at time t
        y_t: output at time t
        W_hh: hidden-to-hidden weights (recurrent connection)
        W_xh: input-to-hidden weights
        W_hy: hidden-to-output weights

    ====================================================================
    🔑 Why RNN?
    ====================================================================

    1. Sequential Processing: Process one element at a time
       → Suitable for variable-length sequences

    2. Memory: Hidden state remembers previous inputs
       → Context-aware predictions

    3. Parameter Sharing: Same weights across all time steps
       → Efficient, scales to any sequence length

    4. Applications:
       - Language modeling: "I am a ___" (predict next word)
       - Sentiment analysis: "This movie is great!" → positive
       - Time series: Stock price prediction
       - User behavior: Predict next action in session

    ====================================================================
    🔑 The Vanishing Gradient Problem
    ====================================================================

    Problem: RNN struggles to learn long-term dependencies

    Why?
        Gradient = ∂L/∂W = ... × ∂h_t/∂h_{t-1} × ∂h_{t-1}/∂h_{t-2} × ...

        If |∂h_t/∂h_{t-1}| < 1:  Gradient vanishes (→ 0)
        If |∂h_t/∂h_{t-1}| > 1:  Gradient explodes (→ ∞)

    Example:
        Sequence: "The cat, which was very cute and fluffy, was hungry"
        RNN: Forgets "cat" by the time it sees "was" → wrong verb form

    Solution: LSTM (Long Short-Term Memory)

    ====================================================================
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Parameters:
            input_size: Dimension of input at each time step
            hidden_size: Size of hidden state (memory capacity)
            output_size: Dimension of output
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        # Initialize weights (small random values)
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01   # input → hidden
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden → hidden (recurrent)
        self.W_hy = np.random.randn(hidden_size, output_size) * 0.01  # hidden → output

        self.b_h = np.zeros((1, hidden_size))  # hidden bias
        self.b_y = np.zeros((1, output_size))  # output bias

    def forward(self, X):
        """
        Forward pass through sequence

        X: (batch_size, sequence_length, input_size)
        Returns: outputs, hidden_states
        """
        batch_size, seq_len, _ = X.shape

        # Initialize hidden state
        h = np.zeros((batch_size, self.hidden_size))

        outputs = []
        hidden_states = [h.copy()]

        # Process sequence step by step
        for t in range(seq_len):
            x_t = X[:, t, :]  # Input at time t

            # RNN update: h_t = tanh(x_t * W_xh + h_{t-1} * W_hh + b_h)
            h = np.tanh(np.dot(x_t, self.W_xh) + np.dot(h, self.W_hh) + self.b_h)

            # Output: y_t = h_t * W_hy + b_y
            y = np.dot(h, self.W_hy) + self.b_y

            outputs.append(y)
            hidden_states.append(h.copy())

        # Stack outputs: (batch_size, seq_len, output_size)
        outputs = np.stack(outputs, axis=1)

        return outputs, hidden_states

    def predict(self, X):
        """Predict (just forward pass)"""
        outputs, _ = self.forward(X)
        return outputs


# ==================== 2. LSTM Implementation ====================
class SimpleLSTM:
    """
    LSTM (Long Short-Term Memory)

    ====================================================================
    🔑 What is LSTM?
    ====================================================================

    LSTM = RNN with gates to control information flow

    Key components:
    1. Cell state (C_t): Long-term memory highway
    2. Hidden state (h_t): Short-term memory (same as RNN)
    3. Gates: Control what to remember/forget

    ====================================================================
    🔑 The Three Gates
    ====================================================================

    1. Forget Gate (f_t): What to throw away from cell state
       f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
       → Values close to 0: forget, close to 1: keep

    2. Input Gate (i_t): What new information to store
       i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
       C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  (candidate values)
       → Decides which values to update

    3. Output Gate (o_t): What to output based on cell state
       o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
       h_t = o_t * tanh(C_t)

    Cell state update:
       C_t = f_t * C_{t-1} + i_t * C̃_t
            ↑               ↑
       Keep old info   Add new info

    ====================================================================
    🔑 Why LSTM Solves Vanishing Gradient?
    ====================================================================

    1. Direct path: C_t = f_t * C_{t-1} + ...
       → Gradient flows directly through cell state (no multiplication chain)

    2. Additive updates: "+", not "*"
       → Gradient doesn't vanish exponentially

    3. Forget gate: Can learn to keep information for long time
       → If f_t ≈ 1, cell state unchanged (perfect memory)

    Example:
        Sequence: "The cat ... was hungry" (100 words in between)
        RNN: Forgets "cat" ❌
        LSTM: Cell state remembers "cat" ✅

    ====================================================================
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Parameters:
            input_size: Dimension of input
            hidden_size: Size of hidden/cell state
            output_size: Dimension of output
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        # Combined input size (concatenate x_t and h_{t-1})
        combined_size = input_size + hidden_size

        # Weights for forget gate
        self.W_f = np.random.randn(combined_size, hidden_size) * 0.01
        self.b_f = np.zeros((1, hidden_size))

        # Weights for input gate
        self.W_i = np.random.randn(combined_size, hidden_size) * 0.01
        self.b_i = np.zeros((1, hidden_size))

        # Weights for candidate cell state
        self.W_c = np.random.randn(combined_size, hidden_size) * 0.01
        self.b_c = np.zeros((1, hidden_size))

        # Weights for output gate
        self.W_o = np.random.randn(combined_size, hidden_size) * 0.01
        self.b_o = np.zeros((1, hidden_size))

        # Output layer
        self.W_y = np.random.randn(hidden_size, output_size) * 0.01
        self.b_y = np.zeros((1, output_size))

    def sigmoid(self, x):
        """Sigmoid activation (for gates)"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

    def forward(self, X):
        """
        Forward pass through LSTM

        X: (batch_size, sequence_length, input_size)
        """
        batch_size, seq_len, _ = X.shape

        # Initialize states
        h = np.zeros((batch_size, self.hidden_size))  # Hidden state
        c = np.zeros((batch_size, self.hidden_size))  # Cell state

        outputs = []
        states = {'h': [h.copy()], 'c': [c.copy()]}

        for t in range(seq_len):
            x_t = X[:, t, :]

            # Concatenate input and hidden state
            combined = np.concatenate([x_t, h], axis=1)

            # ========== Forget Gate ==========
            # Decide what to forget from cell state
            f_t = self.sigmoid(np.dot(combined, self.W_f) + self.b_f)

            # ========== Input Gate ==========
            # Decide what new information to store
            i_t = self.sigmoid(np.dot(combined, self.W_i) + self.b_i)

            # Candidate cell state
            c_tilde = np.tanh(np.dot(combined, self.W_c) + self.b_c)

            # ========== Update Cell State ==========
            c = f_t * c + i_t * c_tilde
            #   ↑         ↑
            # Forget   Remember new

            # ========== Output Gate ==========
            # Decide what to output
            o_t = self.sigmoid(np.dot(combined, self.W_o) + self.b_o)

            # Hidden state (filtered cell state)
            h = o_t * np.tanh(c)

            # Output
            y = np.dot(h, self.W_y) + self.b_y

            outputs.append(y)
            states['h'].append(h.copy())
            states['c'].append(c.copy())

        outputs = np.stack(outputs, axis=1)

        return outputs, states

    def predict(self, X):
        """Predict"""
        outputs, _ = self.forward(X)
        return outputs


# ==================== 3. Sequence Prediction Example ====================
def generate_sine_sequence(n_samples=1000, seq_len=50):
    """
    Generate sine wave sequences for prediction

    Task: Given first 'seq_len' points, predict next point
    """
    X = []
    y = []

    for i in range(n_samples):
        start = np.random.uniform(0, 100)
        time = np.linspace(start, start + seq_len + 1, seq_len + 1)
        sequence = np.sin(time)

        X.append(sequence[:-1].reshape(-1, 1))  # Input: t=0 to t=49
        y.append(sequence[-1])                  # Target: t=50

    return np.array(X), np.array(y).reshape(-1, 1)


def compare_rnn_vs_lstm():
    """Compare RNN and LSTM on sequence prediction"""
    print("=" * 70)
    print("Experiment: RNN vs LSTM on Sine Wave Prediction")
    print("=" * 70)

    # Generate data
    print("\nGenerating sine wave sequences...")
    X_train, y_train = generate_sine_sequence(n_samples=500, seq_len=20)
    X_test, y_test = generate_sine_sequence(n_samples=100, seq_len=20)

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Sequence length: {X_train.shape[1]}")
    print(f"Input dimension: {X_train.shape[2]}")

    # Initialize models
    input_size = 1
    hidden_size = 16
    output_size = 1

    print(f"\n{'='*70}")
    print("Training Simple RNN...")
    print(f"{'='*70}")
    rnn = SimpleRNN(input_size, hidden_size, output_size, learning_rate=0.01)

    # Training loop (simplified - only forward pass for demo)
    n_epochs = 100
    rnn_losses = []

    for epoch in range(n_epochs):
        # Forward pass
        outputs, _ = rnn.forward(X_train)
        predictions = outputs[:, -1, :]  # Last time step

        # MSE loss
        loss = np.mean((predictions - y_train) ** 2)
        rnn_losses.append(loss)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss:.6f}")

    # Test RNN
    rnn_test_outputs, _ = rnn.forward(X_test)
    rnn_test_pred = rnn_test_outputs[:, -1, :]
    rnn_test_loss = np.mean((rnn_test_pred - y_test) ** 2)

    print(f"\nRNN Test Loss: {rnn_test_loss:.6f}")

    # Train LSTM
    print(f"\n{'='*70}")
    print("Training LSTM...")
    print(f"{'='*70}")
    lstm = SimpleLSTM(input_size, hidden_size, output_size, learning_rate=0.01)

    lstm_losses = []
    for epoch in range(n_epochs):
        outputs, _ = lstm.forward(X_train)
        predictions = outputs[:, -1, :]

        loss = np.mean((predictions - y_train) ** 2)
        lstm_losses.append(loss)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss:.6f}")

    # Test LSTM
    lstm_test_outputs, _ = lstm.forward(X_test)
    lstm_test_pred = lstm_test_outputs[:, -1, :]
    lstm_test_loss = np.mean((lstm_test_pred - y_test) ** 2)

    print(f"\nLSTM Test Loss: {lstm_test_loss:.6f}")

    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curves
    axes[0, 0].plot(rnn_losses, label='RNN', linewidth=2)
    axes[0, 0].plot(lstm_losses, label='LSTM', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Training Loss (MSE)', fontsize=11)
    axes[0, 0].set_title('Training Loss: RNN vs LSTM', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)

    # RNN predictions
    n_show = 5
    for i in range(n_show):
        axes[0, 1].plot(X_test[i, :, 0], alpha=0.5, label=f'Input {i+1}')
        axes[0, 1].scatter(len(X_test[i]) - 1, y_test[i], color='red', s=100,
                          marker='*', zorder=5)
        axes[0, 1].scatter(len(X_test[i]) - 1, rnn_test_pred[i], color='blue',
                          s=50, marker='o', zorder=5)

    axes[0, 1].set_xlabel('Time Step', fontsize=11)
    axes[0, 1].set_ylabel('Value', fontsize=11)
    axes[0, 1].set_title('RNN Predictions (Red=True, Blue=Pred)',
                         fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    # LSTM predictions
    for i in range(n_show):
        axes[1, 0].plot(X_test[i, :, 0], alpha=0.5)
        axes[1, 0].scatter(len(X_test[i]) - 1, y_test[i], color='red',
                          s=100, marker='*', zorder=5)
        axes[1, 0].scatter(len(X_test[i]) - 1, lstm_test_pred[i], color='green',
                          s=50, marker='o', zorder=5)

    axes[1, 0].set_xlabel('Time Step', fontsize=11)
    axes[1, 0].set_ylabel('Value', fontsize=11)
    axes[1, 0].set_title('LSTM Predictions (Red=True, Green=Pred)',
                         fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    # Test loss comparison
    models = ['RNN', 'LSTM']
    losses = [rnn_test_loss, lstm_test_loss]
    colors = ['#3498db', '#2ecc71']

    bars = axes[1, 1].bar(models, losses, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set_ylabel('Test Loss (MSE)', fontsize=11)
    axes[1, 1].set_title('Test Loss Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, loss in zip(bars, losses):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{loss:.6f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('rnn_vs_lstm_comparison.png', dpi=100, bbox_inches='tight')
    print("\n📊 Results saved to: rnn_vs_lstm_comparison.png")
    plt.close()

    print(f"\n💡 Observations:")
    print(f"  - Both models can learn simple sequences")
    print(f"  - LSTM typically converges faster and more stably")
    print(f"  - LSTM better at capturing long-term patterns")


# ==================== 4. Visualize RNN Unrolling ====================
def visualize_rnn_unrolling():
    """Visualize how RNN processes sequences"""
    print("\n" + "=" * 70)
    print("Visualization: RNN Unrolling Through Time")
    print("=" * 70)

    # Create simple sequence
    sequence = np.array([[1], [2], [3], [4], [5]])
    X = sequence.reshape(1, 5, 1)  # (1 sample, 5 time steps, 1 feature)

    # Initialize tiny RNN
    rnn = SimpleRNN(input_size=1, hidden_size=3, output_size=1)

    # Forward pass
    outputs, hidden_states = rnn.forward(X)

    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Hidden state evolution
    time_steps = len(hidden_states) - 1
    for neuron_idx in range(3):
        h_values = [h[0, neuron_idx] for h in hidden_states[1:]]  # Skip initial zero state
        axes[0].plot(range(1, time_steps + 1), h_values,
                    marker='o', linewidth=2, markersize=8,
                    label=f'Hidden Neuron {neuron_idx+1}')

    axes[0].set_xlabel('Time Step', fontsize=11)
    axes[0].set_ylabel('Hidden State Value', fontsize=11)
    axes[0].set_title('RNN Hidden State Evolution Over Time',
                     fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].set_xticks(range(1, time_steps + 1))

    # Plot 2: Input and Output
    input_seq = sequence.flatten()
    output_seq = outputs[0, :, 0]

    x_pos = np.arange(len(input_seq))
    width = 0.35

    axes[1].bar(x_pos - width/2, input_seq, width, label='Input',
               alpha=0.7, color='#3498db', edgecolor='black')
    axes[1].bar(x_pos + width/2, output_seq, width, label='Output',
               alpha=0.7, color='#e74c3c', edgecolor='black')

    axes[1].set_xlabel('Time Step', fontsize=11)
    axes[1].set_ylabel('Value', fontsize=11)
    axes[1].set_title('Input Sequence vs RNN Output',
                     fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f't={i+1}' for i in range(len(input_seq))])

    plt.tight_layout()
    plt.savefig('rnn_unrolling.png', dpi=100, bbox_inches='tight')
    print("\n📊 RNN unrolling visualization saved to: rnn_unrolling.png")
    plt.close()

    print("\n💡 Key Points:")
    print("  - Hidden state changes at each time step")
    print("  - Same weights used at all time steps (parameter sharing)")
    print("  - Output depends on current input AND all previous inputs")


# ==================== 5. LSTM Gate Visualization ====================
def visualize_lstm_gates():
    """Visualize LSTM gate activations"""
    print("\n" + "=" * 70)
    print("Visualization: LSTM Gate Activations")
    print("=" * 70)

    # Create sequence
    X = np.random.randn(1, 10, 1)  # 1 sample, 10 time steps

    # Initialize LSTM
    lstm = SimpleLSTM(input_size=1, hidden_size=4, output_size=1)

    # Forward pass and track gates
    batch_size, seq_len, _ = X.shape
    h = np.zeros((batch_size, lstm.hidden_size))
    c = np.zeros((batch_size, lstm.hidden_size))

    gates = {'forget': [], 'input': [], 'output': [], 'cell': []}

    for t in range(seq_len):
        x_t = X[:, t, :]
        combined = np.concatenate([x_t, h], axis=1)

        # Compute gates
        f_t = lstm.sigmoid(np.dot(combined, lstm.W_f) + lstm.b_f)
        i_t = lstm.sigmoid(np.dot(combined, lstm.W_i) + lstm.b_i)
        o_t = lstm.sigmoid(np.dot(combined, lstm.W_o) + lstm.b_o)
        c_tilde = np.tanh(np.dot(combined, lstm.W_c) + lstm.b_c)

        # Update states
        c = f_t * c + i_t * c_tilde
        h = o_t * np.tanh(c)

        # Store gate values (average across hidden units)
        gates['forget'].append(np.mean(f_t))
        gates['input'].append(np.mean(i_t))
        gates['output'].append(np.mean(o_t))
        gates['cell'].append(np.mean(c))

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    time_steps = range(1, seq_len + 1)

    # Forget gate
    axes[0, 0].plot(time_steps, gates['forget'], marker='o', linewidth=2,
                    color='#e74c3c', markersize=8)
    axes[0, 0].set_xlabel('Time Step', fontsize=11)
    axes[0, 0].set_ylabel('Gate Value', fontsize=11)
    axes[0, 0].set_title('Forget Gate (f_t)\n"How much to forget"',
                         fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Input gate
    axes[0, 1].plot(time_steps, gates['input'], marker='o', linewidth=2,
                    color='#3498db', markersize=8)
    axes[0, 1].set_xlabel('Time Step', fontsize=11)
    axes[0, 1].set_ylabel('Gate Value', fontsize=11)
    axes[0, 1].set_title('Input Gate (i_t)\n"How much to add"',
                         fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Output gate
    axes[1, 0].plot(time_steps, gates['output'], marker='o', linewidth=2,
                    color='#2ecc71', markersize=8)
    axes[1, 0].set_xlabel('Time Step', fontsize=11)
    axes[1, 0].set_ylabel('Gate Value', fontsize=11)
    axes[1, 0].set_title('Output Gate (o_t)\n"How much to output"',
                         fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Cell state
    axes[1, 1].plot(time_steps, gates['cell'], marker='o', linewidth=2,
                    color='#9b59b6', markersize=8)
    axes[1, 1].set_xlabel('Time Step', fontsize=11)
    axes[1, 1].set_ylabel('Cell State Value', fontsize=11)
    axes[1, 1].set_title('Cell State (C_t)\n"Long-term memory"',
                         fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('lstm_gates.png', dpi=100, bbox_inches='tight')
    print("\n📊 LSTM gates visualization saved to: lstm_gates.png")
    plt.close()

    print("\n💡 Understanding Gate Values:")
    print("  - Forget gate ≈ 1: Keep old memory")
    print("  - Forget gate ≈ 0: Erase old memory")
    print("  - Input gate ≈ 1: Update with new info")
    print("  - Input gate ≈ 0: Ignore new input")
    print("  - Output gate ≈ 1: Expose cell state")
    print("  - Output gate ≈ 0: Hide internal state")


# ==================== 6. Main Program ====================
def main():
    print("=" * 70)
    print("Recurrent Neural Networks (RNN) and LSTM Basics")
    print("=" * 70)

    # 1. Visualize RNN unrolling
    visualize_rnn_unrolling()

    # 2. Compare RNN vs LSTM
    compare_rnn_vs_lstm()

    # 3. Visualize LSTM gates
    visualize_lstm_gates()

    # 4. Summary
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. RNN Core Ideas
   - Recurrent connection: Output feeds back into input
   - Hidden state: Memory of previous inputs
   - Parameter sharing: Same weights across time
   - Sequential processing: One step at a time

2. RNN Structure
   At each time step t:
       h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
       y_t = W_hy * h_t + b_y

   Hidden state h_t carries information forward in time

3. The Vanishing Gradient Problem
   Problem: RNN can't learn long-term dependencies
   Reason: Gradients vanish/explode through time
   Example: "The cat ... was" (forget "cat" after many steps)

4. LSTM Solution
   Key innovation: Cell state + 3 gates

   - Cell state (C_t): Long-term memory highway
   - Forget gate: Decide what to forget
   - Input gate: Decide what to add
   - Output gate: Decide what to expose

   Advantage: Gradients flow directly through cell state

5. LSTM Gates (All use sigmoid σ)
   Forget gate: f_t = σ(W_f · [h_{t-1}, x_t])
   Input gate:  i_t = σ(W_i · [h_{t-1}, x_t])
   Output gate: o_t = σ(W_o · [h_{t-1}, x_t])

   Cell update: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
                     ↑                ↑
                  Forget          Remember new

   Hidden state: h_t = o_t ⊙ tanh(C_t)

6. RNN vs LSTM
   RNN:
   - Simple structure, fewer parameters
   - Fast to train
   - Struggles with long sequences (vanishing gradient)

   LSTM:
   - Complex structure (4x parameters)
   - Slower to train
   - Handles long-term dependencies well

7. When to Use RNN/LSTM
   ✓ Sequential data: Text, time series, audio
   ✓ Variable-length inputs
   ✓ Context matters (order is important)
   ✓ Need memory of past events

   Applications:
   - Language modeling: Predict next word
   - Sentiment analysis: "This movie is great!" → positive
   - Machine translation: English → Chinese
   - Time series forecasting: Stock prices
   - User behavior modeling: Predict next action

8. RNN in Recommendation Systems
   ✓ User behavior sequences: [click, view, purchase, ...]
   ✓ Session-based recommendations
   ✓ DIN/DIEN models: Attention + RNN for user interest
   ✓ Sequential patterns: "Users who bought X then often buy Y"

9. Variants and Extensions
   - GRU (Gated Recurrent Unit): Simpler than LSTM, fewer parameters
   - Bidirectional RNN: Process sequence forward and backward
   - Stacked RNN: Multiple layers for deeper representation
   - Attention mechanism: Focus on relevant parts (next topic!)

10. Practical Tips
    - Sequence length: Keep < 100 for vanilla RNN, < 300 for LSTM
    - Gradient clipping: Prevent exploding gradients
    - Initialization: Orthogonal for recurrent weights
    - Regularization: Dropout between layers (not time steps!)
    - Batch size: Smaller for variable-length sequences
    """)


if __name__ == "__main__":
    main()

    print("\n💡 Practice Suggestions:")
    print("  1. Implement character-level text generation")
    print("  2. Try predicting stock prices with RNN vs LSTM")
    print("  3. Understand why LSTM solves vanishing gradient")
    print("  4. Visualize gate activations on your own data")
    print("  5. Think: How to use RNN for user behavior modeling?")
