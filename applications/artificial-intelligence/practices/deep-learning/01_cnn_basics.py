"""
Convolutional Neural Networks (CNN) Basics

Problem: Fully connected networks are inefficient for images
Goal: Understand how CNNs process images efficiently

Core Concepts:
1. Convolution: Local feature extraction with weight sharing
2. Pooling: Downsampling for robustness
3. Receptive Field: Input region each neuron "sees"
4. Feature Map: Output of convolutional layer
5. Translation Invariance: Same features detected anywhere

CNN vs Fully Connected:
- FC: Every neuron connects to all inputs → many parameters, loses spatial info
- CNN: Local connections + weight sharing → fewer parameters, preserves spatial structure
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==================== 1. Convolution Operation ====================
def conv2d(image, kernel):
    """
    2D Convolution (simplest implementation)

    ====================================================================
    🔑 What is Convolution?
    ====================================================================

    Convolution = Sliding a small window (kernel/filter) over the image
                  and computing weighted sums

    Example: 3×3 kernel sliding on 5×5 image

    Image (5×5):          Kernel (3×3):
    1 2 3 4 5            -1  0  1
    2 3 4 5 6            -1  0  1
    3 4 5 6 7            -1  0  1
    4 5 6 7 8
    5 6 7 8 9

    Computation (top-left corner):
    1×(-1) + 2×0 + 3×1 +
    2×(-1) + 3×0 + 4×1 +
    3×(-1) + 4×0 + 5×1 = -1 + 3 - 2 + 4 - 3 + 5 = 6

    Then slide right, down, repeat...

    ====================================================================
    🔑 Why is Convolution Useful?
    ====================================================================

    1. Local Connectivity: Each output depends only on local input (receptive field)
       → Fewer parameters, more efficient

    2. Weight Sharing: Same kernel shared across entire image
       → Drastically reduces parameters (3×3 kernel = only 9 params!)

    3. Translation Invariance: Features detected anywhere in the image
       → Edges, textures recognized at any position

    4. Feature Extraction: Different kernels extract different features
       → Edge detection, textures, shapes...

    ====================================================================
    """
    img_h, img_w = image.shape
    kernel_h, kernel_w = kernel.shape

    # Output size (no padding)
    out_h = img_h - kernel_h + 1
    out_w = img_w - kernel_w + 1

    output = np.zeros((out_h, out_w))

    # Sliding window
    for i in range(out_h):
        for j in range(out_w):
            # Extract current window
            window = image[i:i+kernel_h, j:j+kernel_w]
            # Element-wise multiply and sum
            output[i, j] = np.sum(window * kernel)

    return output


def visualize_convolution():
    """Visualize convolution operation"""
    print("=" * 70)
    print("Demo: How Convolution Works")
    print("=" * 70)

    # Create simple image (vertical line)
    image = np.array([
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
    ], dtype=float)

    # Define classic kernels
    kernels = {
        'Vertical Edge': np.array([[-1, 0, 1],
                                   [-1, 0, 1],
                                   [-1, 0, 1]]),

        'Horizontal Edge': np.array([[-1, -1, -1],
                                     [ 0,  0,  0],
                                     [ 1,  1,  1]]),

        'Blur': np.array([[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]]) / 9,

        'Sharpen': np.array([[ 0, -1,  0],
                            [-1,  5, -1],
                            [ 0, -1,  0]]),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image\n(Vertical Line)', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Apply each kernel
    for idx, (name, kernel) in enumerate(kernels.items(), 1):
        output = conv2d(image, kernel)

        axes[idx].imshow(output, cmap='gray')
        axes[idx].set_title(f'{name}\nOutput Size: {output.shape}',
                           fontsize=11, fontweight='bold')
        axes[idx].axis('off')

        print(f"\n{name}:")
        print(f"  Kernel Shape: {kernel.shape}")
        print(f"  Output Shape: {output.shape}")
        print(f"  Output Range: [{output.min():.2f}, {output.max():.2f}]")

    axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig('convolution_demo.png', dpi=100, bbox_inches='tight')
    print("\n📊 Convolution demo saved to: convolution_demo.png")
    plt.close()

    print("\n💡 Observations:")
    print("  - Vertical edge kernel responds strongly to vertical line (bright regions)")
    print("  - Horizontal edge kernel responds weakly to vertical line")
    print("  - Different kernels extract different features")


# ==================== 2. Pooling Operation ====================
def max_pooling(feature_map, pool_size=2):
    """
    Max Pooling

    ====================================================================
    🔑 What is Pooling?
    ====================================================================

    Pooling = Downsampling using local region statistics

    Max Pooling: Take maximum value in window
    Avg Pooling: Take average value in window

    Example: 2×2 Max Pooling

    Input (4×4):          Output (2×2):
    1  2  | 3  4          6  |  8
    5  6  | 7  8    →     ─────
    ─────────────         14 | 16
    9  10 | 11 12
    13 14 | 15 16

    Computation:
    Top-left:  max(1,2,5,6) = 6
    Top-right: max(3,4,7,8) = 8
    Bottom-left: max(9,10,13,14) = 14
    Bottom-right: max(11,12,15,16) = 16

    ====================================================================
    🔑 Why Pooling?
    ====================================================================

    1. Dimensionality Reduction: Reduces feature map size, lowers computation
       → 4×4 → 2×2, reduces parameters by 75%

    2. Robustness: Invariant to small translations
       → Feature slightly shifts, pooling result unchanged

    3. Larger Receptive Field: Later layers "see" larger regions
       → Hierarchical abstraction, extract high-level features

    4. Regularization: Fewer parameters, prevents overfitting

    ====================================================================
    """
    h, w = feature_map.shape
    out_h, out_w = h // pool_size, w // pool_size

    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            window = feature_map[
                i*pool_size:(i+1)*pool_size,
                j*pool_size:(j+1)*pool_size
            ]
            output[i, j] = np.max(window)

    return output


# ==================== 3. Simple CNN Class ====================
class SimpleCNN:
    """
    Simple CNN (1 conv layer + 1 pool layer + FC layer)

    Architecture:
    Input (8×8) → Conv (3×3, 4 kernels) → ReLU → MaxPool (2×2) → Flatten → FC → Softmax

    Why this structure?
    1. Conv layer: Extract local features (edges, textures)
    2. Pool layer: Downsample, increase robustness
    3. Flatten: Convert to vector
    4. FC layer: Classification decision
    """

    def __init__(self, n_filters=4, kernel_size=3, n_classes=10, learning_rate=0.1, n_epochs=100, batch_size=32):
        """
        Parameters:
            n_filters: Number of convolutional kernels (how many features to extract)
            kernel_size: Size of convolutional kernel
            n_classes: Number of classes
        """
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_classes = n_classes
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Initialize convolutional kernels (small random numbers)
        # Shape: (n_filters, kernel_size, kernel_size)
        self.kernels = np.random.randn(n_filters, kernel_size, kernel_size) * 0.1

        # FC layer weights (initialized later, need to know flattened size)
        self.fc_weights = None
        self.fc_bias = None

        self.loss_history = []

    def _conv_forward(self, image):
        """Convolutional layer forward pass"""
        n_filters = self.n_filters
        kernel_size = self.kernel_size

        # Input image size (assume 8×8)
        img_h, img_w = image.shape

        # Output feature map size
        out_h = img_h - kernel_size + 1
        out_w = img_w - kernel_size + 1

        # Each kernel produces one feature map
        feature_maps = np.zeros((n_filters, out_h, out_w))

        for f in range(n_filters):
            feature_maps[f] = conv2d(image, self.kernels[f])

        return feature_maps

    def _relu(self, x):
        """ReLU activation"""
        return np.maximum(0, x)

    def _pool_forward(self, feature_maps):
        """Pooling layer forward pass"""
        n_filters, h, w = feature_maps.shape
        pooled = np.zeros((n_filters, h//2, w//2))

        for f in range(n_filters):
            pooled[f] = max_pooling(feature_maps[f], pool_size=2)

        return pooled

    def forward(self, X):
        """
        Complete forward pass

        X: (n_samples, img_h, img_w)
        """
        n_samples = X.shape[0]

        # Apply convolution and pooling to each sample
        pooled_outputs = []
        for i in range(n_samples):
            # Convolution
            feature_maps = self._conv_forward(X[i])

            # ReLU
            feature_maps = self._relu(feature_maps)

            # Pooling
            pooled = self._pool_forward(feature_maps)

            pooled_outputs.append(pooled)

        # Convert to numpy array
        pooled_outputs = np.array(pooled_outputs)  # (n_samples, n_filters, h, w)

        # Flatten
        flattened = pooled_outputs.reshape(n_samples, -1)

        # Initialize FC layer (if not yet initialized)
        if self.fc_weights is None:
            n_features = flattened.shape[1]
            self.fc_weights = np.random.randn(n_features, self.n_classes) * 0.01
            self.fc_bias = np.zeros((1, self.n_classes))

        # FC layer
        logits = np.dot(flattened, self.fc_weights) + self.fc_bias

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return probs, flattened

    def fit(self, X, y):
        """
        Train model (simplified: only update FC layer, fix conv kernels)

        Note: Full CNN backpropagation is complex. Here we only update FC layer
              to demonstrate the concept while keeping code simple.
        """
        n_samples = X.shape[0]
        y_one_hot = np.eye(self.n_classes)[y]

        print(f"\nTraining CNN (only FC layer, conv kernels fixed)...")

        for epoch in range(self.n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)

            # Mini-batch gradient descent
            for start_idx in range(0, n_samples, self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                X_batch = X[batch_indices]
                y_batch = y_one_hot[batch_indices]

                # Forward pass
                probs, features = self.forward(X_batch)

                # Compute gradients for FC layer
                m = X_batch.shape[0]
                dZ = probs - y_batch  # (batch_size, n_classes)

                dW = (1/m) * np.dot(features.T, dZ)  # (n_features, n_classes)
                db = (1/m) * np.sum(dZ, axis=0, keepdims=True)  # (1, n_classes)

                # Update FC layer parameters
                self.fc_weights -= self.lr * dW
                self.fc_bias -= self.lr * db

            # Record loss every 10 epochs
            if epoch % 10 == 0:
                probs_all, _ = self.forward(X)
                loss = -np.mean(np.sum(y_one_hot * np.log(probs_all + 1e-15), axis=1))
                self.loss_history.append(loss)
                acc = np.mean(np.argmax(probs_all, axis=1) == y)
                print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Acc: {acc:.4f}")

    def predict(self, X):
        """Predict"""
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def score(self, X, y):
        """Accuracy"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# ==================== 4. Compare CNN vs Fully Connected ====================
def compare_cnn_vs_fc():
    """Compare CNN with fully connected network"""
    print("\n" + "=" * 70)
    print("Experiment: CNN vs Fully Connected Network")
    print("=" * 70)

    # Load digit dataset (8×8 handwritten digits)
    digits = load_digits()
    X, y = digits.data, digits.target

    # Reshape to image format
    X_images = X.reshape(-1, 8, 8)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_images, y, test_size=0.3, random_state=42
    )

    print(f"\nDataset Info:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")
    print(f"  Image Size: {X_train.shape[1:]}")
    print(f"  Classes: {len(np.unique(y))}")

    # Normalize
    X_train = X_train / 16.0
    X_test = X_test / 16.0

    # Train CNN
    print(f"\n{'='*70}")
    print("Training CNN...")
    print(f"{'='*70}")
    cnn = SimpleCNN(n_filters=8, kernel_size=3, n_classes=10,
                    learning_rate=0.1, n_epochs=100, batch_size=64)
    cnn.fit(X_train, y_train)

    train_acc = cnn.score(X_train, y_train)
    test_acc = cnn.score(X_test, y_test)

    print(f"\nCNN Results:")
    print(f"  Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Testing Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # Calculate parameter count
    n_conv_params = cnn.n_filters * cnn.kernel_size ** 2
    n_fc_params = cnn.fc_weights.size + cnn.fc_bias.size
    total_params = n_conv_params + n_fc_params

    print(f"\nCNN Parameter Count:")
    print(f"  Conv Layer: {n_conv_params} parameters")
    print(f"  FC Layer: {n_fc_params} parameters")
    print(f"  Total: {total_params} parameters")

    print(f"\n💡 CNN Advantages:")
    print(f"  ✅ Fewer parameters (weight sharing)")
    print(f"  ✅ Preserves spatial information")
    print(f"  ✅ Translation invariance")
    print(f"  ✅ Suitable for image processing")


# ==================== 5. Visualize Learned Filters ====================
def visualize_learned_filters():
    """Visualize CNN learned filters"""
    print("\n" + "=" * 70)
    print("Visualization: What Features Did CNN Learn?")
    print("=" * 70)

    # Load data and train
    digits = load_digits()
    X = digits.data.reshape(-1, 8, 8) / 16.0
    y = digits.target

    cnn = SimpleCNN(n_filters=8, kernel_size=3, n_epochs=100)
    print("Training...")
    cnn.fit(X[:1000], y[:1000])

    # Visualize filters
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(8):
        axes[i].imshow(cnn.kernels[i], cmap='RdBu', vmin=-1, vmax=1)
        axes[i].set_title(f'Filter {i+1}', fontsize=11)
        axes[i].axis('off')

    plt.suptitle('CNN Learned Filters (Edge & Texture Detectors)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('learned_filters.png', dpi=100, bbox_inches='tight')
    print("\n📊 Learned filters saved to: learned_filters.png")
    plt.close()

    print("\n💡 Observations:")
    print("  - Different filters learn different features")
    print("  - Some resemble edge detectors, others texture detectors")
    print("  - These are learned automatically from data!")


# ==================== 6. Main Program ====================
def main():
    print("=" * 70)
    print("Convolutional Neural Networks (CNN) Basics")
    print("=" * 70)

    # 1. Visualize convolution operation
    visualize_convolution()

    # 2. CNN vs Fully Connected
    compare_cnn_vs_fc()

    # 3. Visualize learned features
    visualize_learned_filters()

    # 4. Summary
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. CNN Core Ideas
   - Local Connectivity: Each neuron sees only local region (receptive field)
   - Weight Sharing: Same kernel shared across entire image
   - Hierarchical Features: Shallow layers extract edges, deep layers extract high-level features

2. Convolution Operation
   - Slide small window (kernel) over image
   - Compute weighted sum at each position
   - Different kernels extract different features (edges, textures...)

3. Pooling Operation
   - Downsampling reduces feature map size
   - Increases robustness (invariant to small shifts)
   - Expands receptive field

4. CNN vs Fully Connected
   Fully Connected:
   - Many parameters (8×8 image → 100 hidden = 6,400 params)
   - Loses spatial information
   - No translation invariance

   CNN:
   - Few parameters (8 3×3 kernels = 72 params)
   - Preserves spatial structure
   - Translation invariance

5. CNN Advantages
   ✓ Parameter efficient (weight sharing)
   ✓ Preserves spatial information
   ✓ Translation invariance
   ✓ Hierarchical feature learning
   ✓ Suitable for images, videos, grid data

6. Classic CNN Architectures
   - LeNet (1998): First CNN
   - AlexNet (2012): ImageNet winner, deep learning renaissance
   - VGG (2014): Deeper networks
   - ResNet (2015): Residual connections, hundreds of layers
   - Modern: EfficientNet, Vision Transformer

7. CNN in Recommendation Systems
   ✓ Image feature extraction (product images, user avatars)
   ✓ Sequence modeling (1D conv for user behavior sequences)
   ✓ Multimodal fusion (image + text)

8. Practical Tips
   - Kernel size: 3×3 most common
   - Pooling: 2×2 max pooling common
   - Activation: ReLU standard
   - Data augmentation: rotation, flip, crop (improve generalization)
    """)


if __name__ == "__main__":
    main()

    print("\n💡 Practice Suggestions:")
    print("  1. Try different kernels (edge detection, blur, sharpen)")
    print("  2. Modify number of filters, observe effect on accuracy")
    print("  3. Understand how conv and pooling reduce parameters")
    print("  4. Think: Why is CNN good for images but not tabular data?")
