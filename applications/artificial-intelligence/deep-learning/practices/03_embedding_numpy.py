"""
Embedding Techniques - From Discrete to Continuous

Problem: How to represent discrete symbols (words, items, users) in neural networks?
Goal: Map discrete symbols to continuous vectors that capture semantic similarity

Core Concepts:
1. Embedding: Mapping discrete IDs to dense vectors
2. Semantic Space: Similar items have similar vectors
3. Dimensionality: Usually 50-300 dimensions
4. Learning: Embeddings learned from data, not hand-crafted
5. Applications: NLP, Recommendation, Knowledge Graphs

Why Embedding?
- One-hot encoding: Sparse, high-dimensional, no similarity
- Embedding: Dense, low-dimensional, captures relationships
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# ==================== 1. One-hot vs Embedding ====================
def compare_onehot_vs_embedding():
    """
    Compare one-hot encoding with embedding

    ====================================================================
    🔑 The Problem with One-hot Encoding
    ====================================================================

    One-hot: Represent each item as a binary vector

    Example: 5 words
        "cat"    = [1, 0, 0, 0, 0]
        "dog"    = [0, 1, 0, 0, 0]
        "apple"  = [0, 0, 1, 0, 0]
        "banana" = [0, 0, 0, 1, 0]
        "car"    = [0, 0, 0, 0, 1]

    Problems:
    1. High dimensionality: vocab_size = 10,000 → 10,000 dimensions
    2. Sparse: 99.99% are zeros
    3. No similarity: distance("cat", "dog") = distance("cat", "apple")
       → Can't capture that cat and dog are both animals

    ====================================================================
    🔑 Embedding Solution
    ====================================================================

    Embedding: Map to dense, low-dimensional vectors

    Example: Same 5 words in 3D space
        "cat"    = [0.8, 0.3, 0.1]  ← Close to "dog"
        "dog"    = [0.7, 0.4, 0.2]  ← Close to "cat"
        "apple"  = [0.1, 0.9, 0.3]  ← Close to "banana"
        "banana" = [0.2, 0.8, 0.4]  ← Close to "apple"
        "car"    = [0.5, 0.1, 0.9]  ← Far from animals/fruits

    Advantages:
    1. Low dimensionality: 50-300 dimensions (vs 10,000+)
    2. Dense: Every value is meaningful
    3. Captures similarity: Similar items → Similar vectors
    4. Learned automatically: From data, not manual

    ====================================================================
    """
    print("=" * 70)
    print("Comparison: One-hot Encoding vs Embedding")
    print("=" * 70)

    # Define vocabulary
    vocab = ["cat", "dog", "tiger", "apple", "banana", "orange", "car", "bus"]
    vocab_size = len(vocab)

    # One-hot encoding
    print("\n" + "=" * 70)
    print("1. One-hot Encoding")
    print("=" * 70)

    onehot = np.eye(vocab_size)  # Identity matrix

    print(f"\nVocabulary size: {vocab_size}")
    print(f"One-hot dimension: {vocab_size}")
    print("\nOne-hot vectors:")
    for i, word in enumerate(vocab):
        print(f"  {word:8s}: {onehot[i]}")

    # Compute distances (all equal!)
    cat_idx, dog_idx, apple_idx = 0, 1, 3
    dist_cat_dog = np.linalg.norm(onehot[cat_idx] - onehot[dog_idx])
    dist_cat_apple = np.linalg.norm(onehot[cat_idx] - onehot[apple_idx])

    print(f"\nOne-hot distances:")
    print(f"  distance('cat', 'dog')   = {dist_cat_dog:.3f}")
    print(f"  distance('cat', 'apple') = {dist_cat_apple:.3f}")
    print(f"  → All distances are equal! No semantic meaning.")

    # Embedding (hand-crafted for demo)
    print("\n" + "=" * 70)
    print("2. Embedding (3D)")
    print("=" * 70)

    # Manually designed: [animal_score, fruit_score, vehicle_score]
    embeddings = np.array([
        [0.9, 0.1, 0.1],  # cat
        [0.8, 0.2, 0.1],  # dog
        [0.9, 0.0, 0.0],  # tiger
        [0.1, 0.9, 0.1],  # apple
        [0.1, 0.8, 0.2],  # banana
        [0.0, 0.9, 0.1],  # orange
        [0.1, 0.1, 0.9],  # car
        [0.2, 0.1, 0.8],  # bus
    ])

    print(f"\nEmbedding dimension: 3")
    print("\nEmbedding vectors:")
    for i, word in enumerate(vocab):
        print(f"  {word:8s}: {embeddings[i]}")

    # Compute distances (now meaningful!)
    emb_cat, emb_dog, emb_apple = embeddings[cat_idx], embeddings[dog_idx], embeddings[apple_idx]
    dist_cat_dog_emb = np.linalg.norm(emb_cat - emb_dog)
    dist_cat_apple_emb = np.linalg.norm(emb_cat - emb_apple)

    print(f"\nEmbedding distances:")
    print(f"  distance('cat', 'dog')   = {dist_cat_dog_emb:.3f}  ← Similar (both animals)")
    print(f"  distance('cat', 'apple') = {dist_cat_apple_emb:.3f}  ← Different")
    print(f"  → Distances capture semantic similarity!")

    # Visualize
    fig = plt.figure(figsize=(14, 6))

    # 2D projection of embeddings
    ax = fig.add_subplot(121, projection='3d')

    colors = ['red', 'red', 'red', 'green', 'green', 'green', 'blue', 'blue']
    labels_unique = ['Animals', 'Fruits', 'Vehicles']

    for i, (word, emb, color) in enumerate(zip(vocab, embeddings, colors)):
        ax.scatter(emb[0], emb[1], emb[2], c=color, s=200, alpha=0.6)
        ax.text(emb[0], emb[1], emb[2], word, fontsize=10, fontweight='bold')

    ax.set_xlabel('Animal Dimension', fontsize=10)
    ax.set_ylabel('Fruit Dimension', fontsize=10)
    ax.set_zlabel('Vehicle Dimension', fontsize=10)
    ax.set_title('Embedding Space (3D)\nSimilar items cluster together',
                 fontsize=12, fontweight='bold')

    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Animals'),
                      Patch(facecolor='green', label='Fruits'),
                      Patch(facecolor='blue', label='Vehicles')]
    ax.legend(handles=legend_elements, loc='upper right')

    # Distance comparison
    ax2 = fig.add_subplot(122)

    methods = ['One-hot', 'Embedding']
    cat_dog_distances = [dist_cat_dog, dist_cat_dog_emb]
    cat_apple_distances = [dist_cat_apple, dist_cat_apple_emb]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax2.bar(x - width/2, cat_dog_distances, width,
                    label='cat ↔ dog (similar)', alpha=0.7, color='#3498db')
    bars2 = ax2.bar(x + width/2, cat_apple_distances, width,
                    label='cat ↔ apple (different)', alpha=0.7, color='#e74c3c')

    ax2.set_ylabel('Distance', fontsize=11)
    ax2.set_title('Distance Comparison\nEmbedding captures similarity!',
                  fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('onehot_vs_embedding.png', dpi=100, bbox_inches='tight')
    print("\n📊 Comparison saved to: onehot_vs_embedding.png")
    plt.close()

    print("\n💡 Key Takeaway:")
    print("  - One-hot: No semantic meaning, all items equally distant")
    print("  - Embedding: Similar items → Similar vectors")


# ==================== 2. Word2Vec (Skip-gram) ====================
class SimpleWord2Vec:
    """
    Simplified Word2Vec (Skip-gram model)

    ====================================================================
    🔑 What is Word2Vec?
    ====================================================================

    Word2Vec: Learn word embeddings from text corpus

    Core Idea: Words appearing in similar contexts have similar meanings
               "You shall know a word by the company it keeps"

    Example:
        "The cat sat on the mat"
        "The dog sat on the mat"

        → "cat" and "dog" appear in similar contexts
        → Should have similar embeddings

    ====================================================================
    🔑 Skip-gram Model
    ====================================================================

    Task: Given a center word, predict context words

    Example: Window size = 2
        Sentence: "The cat sat on the mat"
                      ↑
                  center word

        Input:  "cat"
        Output: ["The", "sat"]  (words within window)

    Architecture:
        cat (one-hot) → Embedding → W₁ → hidden → W₂ → softmax → [The, sat]
                         ↑                                ↑
                    Learn this!                    Predict context

    Training:
        For each word in corpus:
            1. Sample center word
            2. Get context words (within window)
            3. Predict context from center
            4. Update embeddings via backprop

    Result: Words with similar contexts get similar embeddings

    ====================================================================
    🔑 Why It Works
    ====================================================================

    Mathematical Insight:
        Maximize: P(context | center)
        Equivalent to: Making similar words close in embedding space

    Example:
        "cat" often appears near: pet, animal, meow, cute
        "dog" often appears near: pet, animal, bark, cute

        → Both share many context words (pet, animal, cute)
        → Optimization pushes their embeddings close together

    ====================================================================
    """

    def __init__(self, vocab_size, embedding_dim=50, learning_rate=0.01):
        """
        Parameters:
            vocab_size: Number of unique words
            embedding_dim: Dimension of embedding vectors
            learning_rate: Learning rate
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate

        # Embedding matrix: (vocab_size, embedding_dim)
        # Each row is a word's embedding
        self.W1 = np.random.randn(vocab_size, embedding_dim) * 0.01

        # Output weights: (embedding_dim, vocab_size)
        self.W2 = np.random.randn(embedding_dim, vocab_size) * 0.01

    def softmax(self, x):
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def forward(self, word_idx):
        """
        Forward pass

        word_idx: Index of center word
        Returns: Probability distribution over vocabulary
        """
        # Get embedding (lookup)
        hidden = self.W1[word_idx]  # (embedding_dim,)

        # Compute scores
        scores = np.dot(hidden, self.W2)  # (vocab_size,)

        # Softmax
        probs = self.softmax(scores)

        return hidden, probs

    def backward(self, word_idx, context_idx, hidden, probs):
        """
        Backward pass (simplified)

        word_idx: Center word index
        context_idx: True context word index
        hidden: Hidden layer output
        probs: Predicted probabilities
        """
        # Compute gradient
        d_scores = probs.copy()
        d_scores[context_idx] -= 1  # Derivative of cross-entropy

        # Update W2
        d_W2 = np.outer(hidden, d_scores)  # (embedding_dim, vocab_size)
        self.W2 -= self.lr * d_W2

        # Update W1 (embedding)
        d_hidden = np.dot(self.W2, d_scores)  # (embedding_dim,)
        self.W1[word_idx] -= self.lr * d_hidden

    def train(self, center_context_pairs, epochs=100):
        """
        Train on center-context pairs

        center_context_pairs: List of (center_idx, context_idx) tuples
        """
        n_pairs = len(center_context_pairs)

        for epoch in range(epochs):
            total_loss = 0

            for center_idx, context_idx in center_context_pairs:
                # Forward
                hidden, probs = self.forward(center_idx)

                # Loss (cross-entropy)
                loss = -np.log(probs[context_idx] + 1e-10)
                total_loss += loss

                # Backward
                self.backward(center_idx, context_idx, hidden, probs)

            if epoch % 20 == 0:
                avg_loss = total_loss / n_pairs
                print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f}")

    def get_embedding(self, word_idx):
        """Get embedding for a word"""
        return self.W1[word_idx]

    def most_similar(self, word_idx, top_k=5):
        """
        Find most similar words using cosine similarity

        Cosine similarity: cos(θ) = (A · B) / (||A|| ||B||)
        Range: [-1, 1], where 1 = identical, 0 = orthogonal, -1 = opposite
        """
        word_emb = self.W1[word_idx]

        # Compute cosine similarity with all words
        # Normalize embeddings
        word_emb_norm = word_emb / (np.linalg.norm(word_emb) + 1e-10)
        all_emb_norm = self.W1 / (np.linalg.norm(self.W1, axis=1, keepdims=True) + 1e-10)

        # Dot product = cosine similarity (when normalized)
        similarities = np.dot(all_emb_norm, word_emb_norm)

        # Get top-k (excluding the word itself)
        similarities[word_idx] = -np.inf  # Exclude self
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return top_indices, similarities[top_indices]


def demo_word2vec():
    """Demonstrate Word2Vec on a toy corpus"""
    print("\n" + "=" * 70)
    print("Demo: Word2Vec (Skip-gram)")
    print("=" * 70)

    # Toy corpus
    corpus = [
        "cat likes fish",
        "dog likes bone",
        "cat likes milk",
        "dog likes meat",
        "bird likes seeds",
        "cat and dog are pets",
        "fish and bone are food",
    ]

    print(f"\nCorpus ({len(corpus)} sentences):")
    for i, sent in enumerate(corpus, 1):
        print(f"  {i}. {sent}")

    # Build vocabulary
    words = []
    for sent in corpus:
        words.extend(sent.split())

    vocab = sorted(set(words))
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    print(f"\nVocabulary ({len(vocab)} words):")
    print(f"  {vocab}")

    # Generate training pairs (center, context)
    window_size = 2
    pairs = []

    for sent in corpus:
        word_indices = [word2idx[w] for w in sent.split()]

        for i, center_idx in enumerate(word_indices):
            # Get context indices (within window)
            context_start = max(0, i - window_size)
            context_end = min(len(word_indices), i + window_size + 1)

            for j in range(context_start, context_end):
                if j != i:  # Don't include center word itself
                    pairs.append((center_idx, word_indices[j]))

    print(f"\nTraining pairs: {len(pairs)}")
    print(f"  Example: ('{idx2word[pairs[0][0]]}' → '{idx2word[pairs[0][1]]}')")

    # Train Word2Vec
    print("\n" + "=" * 70)
    print("Training Word2Vec...")
    print("=" * 70)

    vocab_size = len(vocab)
    embedding_dim = 10

    w2v = SimpleWord2Vec(vocab_size, embedding_dim, learning_rate=0.1)
    w2v.train(pairs, epochs=200)

    # Test similarity
    print("\n" + "=" * 70)
    print("Word Similarities")
    print("=" * 70)

    test_words = ["cat", "dog", "fish"]

    for word in test_words:
        if word in word2idx:
            word_idx = word2idx[word]
            similar_indices, similarities = w2v.most_similar(word_idx, top_k=3)

            print(f"\nMost similar to '{word}':")
            for idx, sim in zip(similar_indices, similarities):
                print(f"  {idx2word[idx]:10s}: {sim:.4f}")

    # Visualize embeddings
    print("\n" + "=" * 70)
    print("Visualizing Embeddings...")
    print("=" * 70)

    # Get all embeddings
    embeddings = w2v.W1

    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color code by category
    categories = {
        'animals': ['cat', 'dog', 'bird'],
        'food': ['fish', 'bone', 'milk', 'meat', 'seeds'],
        'other': ['likes', 'and', 'are', 'pets', 'food']
    }

    colors = {'animals': 'red', 'food': 'green', 'other': 'gray'}

    for category, words_in_cat in categories.items():
        indices = [word2idx[w] for w in words_in_cat if w in word2idx]
        if indices:
            x = embeddings_2d[indices, 0]
            y = embeddings_2d[indices, 1]
            ax.scatter(x, y, c=colors[category], s=200, alpha=0.6, label=category.capitalize())

    # Add labels
    for word, idx in word2idx.items():
        x, y = embeddings_2d[idx]
        ax.annotate(word, (x, y), fontsize=11, fontweight='bold',
                   ha='center', va='center')

    ax.set_xlabel('PCA Component 1', fontsize=11)
    ax.set_ylabel('PCA Component 2', fontsize=11)
    ax.set_title('Word2Vec Embeddings (2D Projection)\nSimilar words cluster together',
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('word2vec_embeddings.png', dpi=100, bbox_inches='tight')
    print("\n📊 Embeddings visualization saved to: word2vec_embeddings.png")
    plt.close()

    print("\n💡 Observations:")
    print("  - Words with similar contexts (cat, dog) are close")
    print("  - Words appearing together (cat, fish) are close")
    print("  - Unrelated words are far apart")


# ==================== 3. Embedding in Recommendation Systems ====================
def demo_item_embedding():
    """
    Item Embedding for Recommendation

    ====================================================================
    🔑 Embedding in RecSys
    ====================================================================

    Same idea as Word2Vec, but for items/users:

    1. User Embedding:
       - Represent each user as a dense vector
       - Similar users → Similar vectors

    2. Item Embedding:
       - Represent each item as a dense vector
       - Similar items → Similar vectors

    3. Prediction:
       score = user_embedding · item_embedding (dot product)
       High score → User likely to like item

    ====================================================================
    🔑 Learning Embeddings
    ====================================================================

    Method 1: Matrix Factorization
        User-Item matrix R → U (user embeddings) × I (item embeddings)

    Method 2: Neural Networks
        Learn embeddings as first layer of neural network

    Method 3: Implicit Feedback
        User clicked items → Learn from co-occurrence (like Word2Vec)

    ====================================================================
    """
    print("\n" + "=" * 70)
    print("Demo: Item Embedding for Recommendation")
    print("=" * 70)

    # Simulated user-item interactions
    # users × items, 1 = interaction, 0 = no interaction
    user_item_matrix = np.array([
        [1, 1, 0, 0, 0, 0],  # User 0: likes item 0, 1 (action movies)
        [1, 1, 1, 0, 0, 0],  # User 1: likes item 0, 1, 2
        [0, 0, 0, 1, 1, 0],  # User 2: likes item 3, 4 (romance)
        [0, 0, 0, 1, 1, 1],  # User 3: likes item 3, 4, 5
        [1, 0, 0, 0, 1, 0],  # User 4: mixed preference
    ])

    items = ["Action-1", "Action-2", "Action-3", "Romance-1", "Romance-2", "Romance-3"]
    n_users, n_items = user_item_matrix.shape
    embedding_dim = 3

    print(f"\nUser-Item Interaction Matrix ({n_users} users, {n_items} items):")
    print(f"{'':10s}", end='')
    for item in items:
        print(f"{item:12s}", end='')
    print()
    for i in range(n_users):
        print(f"User {i:3d}:  ", end='')
        for j in range(n_items):
            print(f"{user_item_matrix[i, j]:12d}", end='')
        print()

    # Simple embedding learning (matrix factorization)
    print("\n" + "=" * 70)
    print("Learning Embeddings via Matrix Factorization...")
    print("=" * 70)

    # Initialize
    user_embeddings = np.random.randn(n_users, embedding_dim) * 0.1
    item_embeddings = np.random.randn(n_items, embedding_dim) * 0.1

    # Train (simplified)
    lr = 0.01
    n_epochs = 500

    for epoch in range(n_epochs):
        total_loss = 0

        for i in range(n_users):
            for j in range(n_items):
                if user_item_matrix[i, j] > 0:  # Only train on observed interactions
                    # Predict
                    pred = np.dot(user_embeddings[i], item_embeddings[j])

                    # Loss (MSE)
                    error = user_item_matrix[i, j] - pred
                    total_loss += error ** 2

                    # Update embeddings (gradient descent)
                    user_embeddings[i] += lr * error * item_embeddings[j]
                    item_embeddings[j] += lr * error * user_embeddings[i]

        if epoch % 100 == 0:
            avg_loss = total_loss / np.sum(user_item_matrix > 0)
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f}")

    print(f"\nLearned Item Embeddings ({embedding_dim}D):")
    for i, item in enumerate(items):
        print(f"  {item:12s}: {item_embeddings[i]}")

    # Compute item similarities
    print("\n" + "=" * 70)
    print("Item Similarities (Cosine)")
    print("=" * 70)

    # Normalize embeddings
    item_emb_norm = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)

    # Compute similarity matrix
    similarity_matrix = np.dot(item_emb_norm, item_emb_norm.T)

    print(f"\n{'':12s}", end='')
    for item in items:
        print(f"{item:12s}", end='')
    print()

    for i in range(n_items):
        print(f"{items[i]:12s}", end='')
        for j in range(n_items):
            print(f"{similarity_matrix[i, j]:12.3f}", end='')
        print()

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Item embeddings (2D projection)
    pca = PCA(n_components=2)
    item_emb_2d = pca.fit_transform(item_embeddings)

    colors = ['red', 'red', 'red', 'blue', 'blue', 'blue']
    for i, (item, color) in enumerate(zip(items, colors)):
        axes[0].scatter(item_emb_2d[i, 0], item_emb_2d[i, 1],
                       c=color, s=300, alpha=0.6)
        axes[0].annotate(item, (item_emb_2d[i, 0], item_emb_2d[i, 1]),
                        fontsize=10, fontweight='bold', ha='center', va='center')

    axes[0].set_xlabel('PCA Component 1', fontsize=11)
    axes[0].set_ylabel('PCA Component 2', fontsize=11)
    axes[0].set_title('Item Embeddings (2D Projection)\nSimilar items cluster',
                     fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Action'),
                      Patch(facecolor='blue', label='Romance')]
    axes[0].legend(handles=legend_elements)

    # Plot 2: Similarity heatmap
    im = axes[1].imshow(similarity_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[1].set_xticks(range(n_items))
    axes[1].set_yticks(range(n_items))
    axes[1].set_xticklabels(items, rotation=45, ha='right')
    axes[1].set_yticklabels(items)
    axes[1].set_title('Item Similarity Matrix\nGreen = Similar, Red = Dissimilar',
                     fontsize=12, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1])
    cbar.set_label('Cosine Similarity', fontsize=10)

    # Add text annotations
    for i in range(n_items):
        for j in range(n_items):
            text = axes[1].text(j, i, f'{similarity_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)

    plt.tight_layout()
    plt.savefig('item_embeddings.png', dpi=100, bbox_inches='tight')
    print("\n📊 Item embeddings visualization saved to: item_embeddings.png")
    plt.close()

    print("\n💡 Observations:")
    print("  - Action movies cluster together (high similarity)")
    print("  - Romance movies cluster together")
    print("  - Cross-category similarity is low")
    print("  - Can use embeddings to recommend similar items!")


# ==================== 4. Main Program ====================
def main():
    print("=" * 70)
    print("Embedding Techniques: From Discrete to Continuous")
    print("=" * 70)

    # 1. One-hot vs Embedding
    compare_onehot_vs_embedding()

    # 2. Word2Vec demo
    demo_word2vec()

    # 3. Item embedding for RecSys
    demo_item_embedding()

    # 4. Summary
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. The Embedding Concept
   - Discrete symbols (words, items, users) → Dense vectors
   - Similar items → Similar vectors (close in space)
   - Low dimensional (50-300D vs 10,000+D one-hot)
   - Learned from data, not hand-crafted

2. Why Embedding Works
   One-hot Encoding:
   - "cat" = [1, 0, 0, ..., 0]  (10,000 dimensions)
   - "dog" = [0, 1, 0, ..., 0]
   - distance("cat", "dog") = √2  (same as any other pair)
   - No semantic meaning

   Embedding:
   - "cat" = [0.8, 0.3, 0.1, ...]  (50 dimensions)
   - "dog" = [0.7, 0.4, 0.2, ...]
   - distance("cat", "dog") = 0.2  (close!)
   - Captures similarity

3. Word2Vec (Skip-gram)
   Idea: Words in similar contexts have similar meanings

   Training:
   - Given center word, predict context words
   - "The [cat] sat on" → Predict: "The", "sat"
   - Update embeddings via gradient descent

   Result:
   - Similar words → Similar embeddings
   - Can capture analogies: king - man + woman ≈ queen

4. Embedding in Recommendation Systems
   User Embedding:
   - Each user → Dense vector
   - Similar users → Similar vectors

   Item Embedding:
   - Each item → Dense vector
   - Similar items → Similar vectors

   Prediction:
   - score = user_embedding · item_embedding
   - High score → Recommend item to user

5. Learning Embeddings
   Method 1: Matrix Factorization
   - Factor user-item matrix: R ≈ U × I^T
   - U: user embeddings, I: item embeddings

   Method 2: Neural Networks
   - First layer: Embedding lookup
   - Train end-to-end with task objective

   Method 3: Word2Vec Style
   - Learn from item co-occurrence
   - "Users who liked A also liked B"

6. Similarity Metrics
   Cosine Similarity: cos(θ) = (A · B) / (||A|| ||B||)
   - Range: [-1, 1]
   - 1 = identical, 0 = orthogonal, -1 = opposite
   - Most common in NLP and RecSys

   Euclidean Distance: d = ||A - B||
   - Range: [0, ∞)
   - 0 = identical, larger = more different

   Dot Product: A · B
   - Range: (-∞, +∞)
   - Simple, fast, used in retrieval

7. Applications
   NLP:
   ✓ Word embeddings (Word2Vec, GloVe, FastText)
   ✓ Sentence embeddings (BERT, Sentence-BERT)
   ✓ Semantic search, similarity, clustering

   Recommendation:
   ✓ User/item embeddings in two-tower models
   ✓ Collaborative filtering
   ✓ Cold-start handling (content-based embeddings)

   Other:
   ✓ Knowledge graphs (entity embeddings)
   ✓ Social networks (node embeddings)
   ✓ Biology (protein/gene embeddings)

8. Dimensionality Choice
   Trade-off:
   - Too low (e.g., 10D): Can't capture all patterns
   - Too high (e.g., 1000D): Overfitting, slow, memory

   Common choices:
   - Word2Vec: 50-300D
   - Two-tower RecSys: 64-512D
   - BERT: 768D (base), 1024D (large)

9. Visualization
   Problem: Embeddings are high-dimensional (50-300D)
   Solution: Dimensionality reduction to 2D/3D

   PCA (Principal Component Analysis):
   - Linear projection
   - Fast, deterministic
   - Preserves global structure

   t-SNE (t-Distributed Stochastic Neighbor Embedding):
   - Non-linear projection
   - Slow, stochastic
   - Preserves local structure (clusters)

10. Two-Tower Model Connection
    This is the foundation for recommendation systems!

    User Tower:                  Item Tower:
    User ID → Embedding (64D)    Item ID → Embedding (64D)
         ↓                            ↓
    User Vector                  Item Vector
         ↓                            ↓
         └──────── Dot Product ───────┘
                      ↓
                   Score (relevance)

    Training:
    - Positive pairs (user clicked item): High score
    - Negative pairs (user didn't click): Low score
    - Learn embeddings via gradient descent

11. Practical Tips
    - Initialize randomly (small values, e.g., N(0, 0.01))
    - Normalize embeddings for cosine similarity
    - Use pretrained embeddings when possible (Word2Vec, GloVe)
    - Fine-tune embeddings on your specific task
    - Regularize to prevent overfitting (L2 penalty)
    - Monitor embedding norm (shouldn't explode)
    """)


if __name__ == "__main__":
    main()

    print("\n💡 Practice Suggestions:")
    print("  1. Implement GloVe (co-occurrence matrix factorization)")
    print("  2. Train Word2Vec on a larger corpus (e.g., Wikipedia)")
    print("  3. Visualize embeddings with t-SNE")
    print("  4. Build a simple movie recommender using item embeddings")
    print("  5. Explore pretrained embeddings (GloVe, FastText)")
    print("  6. Think: How do two-tower models use embeddings?")
