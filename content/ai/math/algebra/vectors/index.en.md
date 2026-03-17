---
weight: 1
title: "Vectors, Scalars and Vector Spaces"
authors:
  - jnonino
description: >
  Master vectors, scalars & vector spaces from scratch. Step-by-step math derivations, runnable Python code, and research-grade insights for aspiring Machine Learning (ML) scientists.
date: 2026-03-06
tags: ["AI", "Maths", "Algebra", "Vectors"]
---

In 2017, researchers at Google Brain published [**Attention Is All You Need**](https://arxiv.org/abs/1706.03762), introducing the Transformer architecture that now underlies GPT-4, Gemini, and virtually every state-of-the-art language model. At the heart of that architecture (and of every neural network, recommendation system, and computer vision model) is a deceptively simple object: the **vector**.

When a language model reads the word *"bank"*, it doesn't see a string. It sees a vector in a 4096-dimensional space where *"bank (financial)"* and *"bank (riverbank)"* occupy measurably different regions. When a search engine decides that your query matches a document, it is computing an angle between two vectors. When a neural network learns, it is moving vectors through space in response to a gradient, itself a vector.

This article builds your working foundation for all of that. By the end, you will be able to:

- Formally define vectors, scalars, and vector spaces, and explain *why* the axioms matter.
- Compute norms, dot products, and inter-vector angles both by hand and in *NumPy*.
- Reason geometrically about high-dimensional data, a non-negotiable skill for Machine Learning research.
- Read a research paper that uses vector notation without losing the thread.

No fluff, let's start.

## Prerequisites

Before reading this article, you should be comfortable with:

- **High school algebra:** variables, functions, the coordinate plane.
- **Python basics**: lists, loops, functions, importing libraries.
- **Basic calculus intuition** (helpful but not required): the idea that a derivative points in the direction of steepest increase.

## Intuition first

### The programmer's analogy: vectors as typed arrays with geometric soul

As a developer, you've used arrays your whole career. A Python list `[3.0, -1.5, 7.2]` stores three numbers. A vector is superficially the same thing, but with a crucial additional structure: **position in space and the geometry that connects positions**.

Think of it this way. If you have two dictionaries in Python:

```python
user_A = {"age": 28, "purchase_freq": 5, "avg_spend": 120.0}
user_B = {"age": 29, "purchase_freq": 6, "avg_spend": 115.0}
```

As dictionaries, they're just data blobs. You can read values, but *"how similar are these users?"* is not a question the dictionary can answer natively. Now convert them to vectors:

```
A = [28, 5, 120.0]
B = [29, 6, 115.0]
```

Suddenly you have geometry. You can measure the *distance* between them, the *angle* they form relative to the origin, and whether one is a *scaled version* of another. This is the leap vectors make over plain arrays: **they live in a space equipped with rules for measuring, comparing, and transforming**.

### Geometric picture: vectors as arrows

Picture a standard 2D coordinate system. The vector \(\mathbf{v} = [3, 2]\) is an arrow starting at the origin \((0, 0)\) and ending at the point \((3, 2)\). Two things define it completely: its **magnitude** (how long the arrow is) and its **direction** (which way it points).

{{< figure
    src="images/vector.png"
    alt="Visual representation of a two-dimension vector"
    caption="Visual representation of a two-dimension vector"
    >}}

This geometric interpretation is not just visual sugar. In Machine Learning, a data point (a row in your dataset) *is* a vector, it's an arrow in feature space. Two similar data points are arrows pointing in roughly the same direction. An outlier is an arrow pointing somewhere unexpected. Dimensionality reduction (PCA, UMAP) is the art of finding a lower-dimensional space where those arrows still tell roughly the same story.

### Scalars: the simplest case

A **scalar** is just a single number, no direction, no components. Temperature, loss value, learning rate: all scalars. When you multiply a vector by a scalar, you stretch or shrink the arrow without rotating it:

Same direction vector, twice as long.
$$
2 \cdot \begin{pmatrix} 3 \\ 2 \end{pmatrix} = \begin{pmatrix} 2 \cdot 3 \\ 2 \cdot 2 \end{pmatrix} = \begin{pmatrix} 6 \\ 4 \end{pmatrix}
$$
```python
2 x [3, 2] = [6, 4]
```

Same direction vector, flipped (180°).
$$
-1 \cdot \begin{pmatrix} 3 \\ 2 \end{pmatrix} = \begin{pmatrix} -1 \cdot 3 \\ -1 \cdot 2 \end{pmatrix} = \begin{pmatrix} -3 \\ -2 \end{pmatrix}
$$
```python
-1 x [3, 2] = [-3, -2]
```

This operation, **scalar multiplication**, is one of the two foundational operations that define a vector space.

{{< callout type="important" >}}
When debugging a neural network and the loss explodes, it often means vectors (activations or gradients) are being scaled up by factors much greater than 1.0 each layer. Understanding scalar multiplication geometrically helps you see *why* gradient clipping or batch normalisation restores stability: they're renormalizing the length of those arrows.
{{< /callout >}}

## Mathematical derivation

### Formal definitions

#### Scalar

An element of a field \(\mathbb{F}\), for our purposes, a real number \(\mathbb{R}\) or complex number \(\mathbb{C}\). Denoted with standard italics and usually greek characters: \(\alpha\), \(\beta\), \(\lambda\).

#### Vector

An ordered tuple of scalars from \(\mathbb{F}\). An \(n\)-dimensional real vector is an element of the space \(\mathbb{R}^n\):

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n$$

{{< callout type="info" >}}
In plain English: \(\mathbf{v}\) is a column of \(n\) real numbers. The subscript tells you which *"slot"* you're in.
{{< /callout >}}

#### Vectorial space

A set \(V\) of vectors over a field \(\mathbb{F}\), equipped with two operations, for which it is ***closed***.

**Vector addition**: \(\mathbf{u} + \mathbf{v} \in V\) for all \(\mathbf{u}, \mathbf{v} \in V\). Internal operation with the following properties:
- *Associativity*: \(\mathbf{u} + (\mathbf{v} + \mathbf{w}) = (\mathbf{u} + \mathbf{v}) + \mathbf{w}\).
- *Commutativity*: \(\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}\).
- *Identity element*: there exists a vector \(\mathbf{0} \in V\), called zero vector, such that \(\mathbf{v} + \mathbf{0} = \mathbf{v}\) for all \(\mathbf{v} \in V\).
- *Inverse element*: for all \(\mathbf{v} \in V\), there exist an element \(\mathbf{-v} \in V\) such as \(\mathbf{v} + (\mathbf{-v}) = \mathbf{0}\).

**Scalar multiplication**: \(\alpha \mathbf{v} \in V\) for all \(\alpha \in \mathbb{F}, \mathbf{v} \in V\).  External operation with the following properties:
- *Associativity*: \(\alpha (\beta \mathbf{v}) = (\alpha \beta) \mathbf{v}\).
- *Identity element*: there exists an scalar \(\alpha\), such as \(\alpha \mathbf{v} = \mathbf{v} \alpha = \mathbf{v}\) para todo \(\mathbf{v} \in V\).
- *Distributivity of scalar multiplication with respect to vector addition*: For any scalar \(\alpha\), it is true that \(\alpha (\mathbf{u} + \mathbf{v}) = \alpha \mathbf{u} + \alpha \mathbf{v}\) for all \(\mathbf{u}, \mathbf{v} \in V\).
- *Distributivity of scalar multiplication with respect to scalars addition*: For any two scalars \(\alpha\) and \(\beta\), it is true that \((\alpha + \beta) \mathbf{v} = \alpha \mathbf{v} + \beta \mathbf{v}\) for every \(\mathbf{v} \in V\).

{{< callout >}}
What does it mean that the vector space is closed for those operations?

It means that they produce results that are in the same vector space.
- If \(\mathbf{u} \in V\) and \(\mathbf{v} \in V\) then \((\mathbf{u} + \mathbf{v}) \in V\).
- For any scalar \(\alpha \in \mathbb{R}\) and vector \(\mathbf{v} \in V\), then \(\alpha \mathbf{v} \in V\).
{{< /callout >}}

### Vector addition

Given \(\mathbf{u} = [u_1, u_2, \ldots, u_n]^T\) and \(\mathbf{v} = [v_1, v_2, \ldots, v_n]^T\):

$$
\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{bmatrix}
$$

{{< callout type="info" >}}
In plain English: add element-by-element. Geometrically, place the tail of \(\mathbf{v}\) at the head of \(\mathbf{u}\), the result is the arrow from start to finish following the [parallelogram law](https://en.wikipedia.org/wiki/Parallelogram_law).
{{< /callout >}}

### Vector norms

The **norm** of a vector measures its length. The most common is the **Euclidean norm** (\(L^2\) norm):

$$
\|\mathbf{v}\|_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \sqrt{\sum_{i=1}^{n} v_i^2}
$$

{{< callout type="info" >}}
In plain English: square each component, sum them, take the square root. This is precisely the Pythagorean theorem generalized to \(n\) dimensions.
{{< /callout >}}

The general family is the **\(L^p\) norm**:

$$
\|\mathbf{v}\|_p = \left( \sum_{i=1}^{n} |v_i|^p \right)^{1/p}
$$

Two special cases appear constantly in Machine Learning (ML):

- **\(L^1\) norm (Manhattan)**:

    Used in LASSO regularisation because it induces *sparsity*, it penalizes any non-zero component equally.
    $$
    \|\mathbf{v}\|_1 = \sum_{i=1}^{n} |v_i|
    $$

- **\(L^\infty\) norm (max norm)**:

    Useful when you care about the single largest activation.
    $$
    \|\mathbf{v}\|_\infty = \max_i |v_i|
    $$

To dig more about Vector norms, check [this](https://en.wikipedia.org/wiki/Norm_(mathematics)) Wikipedia article.

### The dot product

The **dot product** (inner product) of two vectors is:

$$
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n
$$

{{< callout type="info" >}}
In plain English: multiply corresponding components and sum the results. The output is a scalar, a single number that encodes how much the two vectors ***"align"***.
{{< /callout >}}

{{< callout >}}
The dot product of a vector and itself results in its magnitude squared.
$$
\mathbf{v} \cdot \mathbf{v} = \|\mathbf{v}\|^2
$$
{{< /callout >}}

### The angle between vectors

Here is where geometry and algebra merge beautifully. From Euclidean geometry, the [**Law of Cosines**](https://en.wikipedia.org/wiki/Law_of_cosines) states that for a triangle with sides \(a\), \(b\), \(c\) and angle \(\theta\) opposite side \(c\):

$$
c^2 = a^2 + b^2 - 2 a b \cos(\theta)
$$

Now apply this to vectors. Let \(\mathbf{u}\) and \(\mathbf{v}\) be two vectors. The third side of the triangle they form is \(\mathbf{u} - \mathbf{v}\).

Substituting \(a = \|\mathbf{u}\|\), \(b = \|\mathbf{v}\|\), \(c = \|\mathbf{u} - \mathbf{v}\|\):

$$
\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2 \|\mathbf{u}\| \|\mathbf{v}\|\ cos(\theta)
$$

Expanding the left side:

$$
\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2 \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)
$$

$$
(\mathbf{u} - \mathbf{v}) \cdot (\mathbf{u} - \mathbf{v}) = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2 \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)
$$

$$
\|\mathbf{u}\|^2 - 2 (\mathbf{u} \cdot \mathbf{v}) + \|\mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2 \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)
$$

Cancelling \(\|\mathbf{u}\|^2\) and \(\|\mathbf{v}\|^2\) from both sides:
$$
-2 (\mathbf{u} \cdot \mathbf{v}) = -2 \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)
$$

Dividing both sides by \(-2 \|\mathbf{u}\| \|\mathbf{v}\|\) (assuming neither vector is zero):
$$
\frac{-2 (\mathbf{u} \cdot \mathbf{v})}{-2 \|\mathbf{u}\| \|\mathbf{v}\|} = \frac{-2 \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)}{-2 \|\mathbf{u}\| \|\mathbf{v}\|}
$$

$$
\boxed{\frac{(\mathbf{u} \cdot \mathbf{v})}{\|\mathbf{u}\| \|\mathbf{v}\|} = \cos(\theta)}
$$

{{< callout type="info" >}}
In plain English: the cosine of the angle between two vectors equals their dot product divided by the product of their lengths. This formula is ***foundational***, it gives us **cosine similarity**, one of the most ubiquitous distance metrics in Machine Learning.
{{< /callout >}}

Key interpretations:
- \(\cos\theta = 1\) (\(\theta = 0°\)): vectors point in the **same direction** (identical topics in a document embedding).
- \(\cos\theta = 0\) (\(\theta = 90°\)): vectors are **orthogonal**, completely unrelated.
- \(\cos\theta = -1\) (\(\theta = 180°\)): vectors point in **opposite directions** (antonyms in a well-trained embedding space).

{{< callout >}}
In the original [Word2Vec](https://arxiv.org/abs/1301.3781) paper (Mikolov et al., 2013), the famous analogy:
$$
king − man + woman ≈ queen
$$
works precisely because of this geometry. Semantic relationships are encoded as *directions* in vector space, and finding `queen` means finding the vector whose cosine similarity to the query vector is maximized. Every modern embedding model (BERT, GPT, sentence-transformers) inherits this geometric philosophy. Next time you read something about word representation in vector spaces, remember they are talking about the same geometry we just derived.
{{< /callout >}}

### The cross product

The dot product takes two vectors and returns a **scalar**. The **cross product** takes two vectors in \(\mathbb{R}^3\) and returns a **vector**, one that is perpendicular to both inputs. It is defined only in three (and seven) dimensions, which makes it more geometrically specialised than the dot product.

Given \(\mathbf{u} = [u_1, u_2, u_3]^T\) and \(\mathbf{v} = [v_1, v_2, v_3]^T\), the cross product \(\mathbf{u} \times \mathbf{v}\) is computed by expanding the following symbolic determinant:

$$
\mathbf{u} \times \mathbf{v} = \begin{vmatrix} \mathbf{e}_1 & \mathbf{e}_2 & \mathbf{e}_3 \\ u_1 & u_2 & u_3 \\ v_1 & v_2 & v_3 \end{vmatrix}
$$

Expanding along the first row:

$$
\mathbf{u} \times \mathbf{v} = \mathbf{e}_1(u_2 v_3 - u_3 v_2) - \mathbf{e}_2(u_1 v_3 - u_3 v_1) + \mathbf{e}_3(u_1 v_2 - u_2 v_1)
$$

$$
\boxed{\mathbf{u} \times \mathbf{v} = \begin{bmatrix} u_2 v_3 - u_3 v_2 \\ u_3 v_1 - u_1 v_3 \\ u_1 v_2 - u_2 v_1 \end{bmatrix}}
$$

{{< callout type="info" >}}
In plain English: each component of the result is a \(2 \times 2\) determinant built from the other two components of the inputs. The pattern is cyclic: \((2,3)\), \((3,1)\), \((1,2)\).
{{< /callout >}}

**Two geometric facts define the cross product completely**:

**Direction:** \(\mathbf{u} \times \mathbf{v}\) is always orthogonal to both \(\mathbf{u}\) and \(\mathbf{v}\). You can verify this directly: \((\mathbf{u} \times \mathbf{v}) \cdot \mathbf{u} = 0\) and \((\mathbf{u} \times \mathbf{v}) \cdot \mathbf{v} = 0\). The orientation follows the **right-hand rule**: curl the fingers of your right hand from \(\mathbf{u}\) toward \(\mathbf{v}\), and your thumb points in the direction of \(\mathbf{u} \times \mathbf{v}\).

**Magnitude:** The length of the result equals the area of the parallelogram spanned by \(\mathbf{u}\) and \(\mathbf{v}\), which can be expressed as:

$$\|\mathbf{u} \times \mathbf{v}\| = \|\mathbf{u}\|\|\mathbf{v}\|\sin\theta$$


{{< callout type="info" >}}
When \(\mathbf{u}\) and \(\mathbf{v}\) are parallel (\(\theta = 0°\)), the parallelogram is flat and the cross product is the zero vector. When they are perpendicular (\(\theta = 90°\)), the parallelogram has maximum area and \(\|\mathbf{u} \times \mathbf{v}\|\) is maximised. This is the exact opposite behaviour to the dot product, which is maximised when vectors are parallel and zero when perpendicular.
{{< /callout >}}

**Key algebraic property, anticommutativity**:

$$
\mathbf{u} \times \mathbf{v} = -(\mathbf{v} \times \mathbf{u})
$$

Swapping the order flips the sign and the direction. This means the cross product is **not commutative**, unlike the dot product.

## Python implementation

Let's implement everything from scratch, first in pure Python, then verify with NumPy.

{{< tabs >}}
  {{< tab name="Pure Python" >}}
    {{< details title="Code" closed="true" >}}
      {{< codeimporter
          url="https://raw.githubusercontent.com/learn-software-engineering/examples/refs/heads/main/ai/math/algebra/vectors/vector_pure_python.en.py"
          type="python"
          >}}
    {{< /details >}}
    {{< details title="Output" closed="true" >}}
      ```bash
      >  python vector_pure_python.en.py
      ==================================
      Vector Operations with pure Python
      ==================================
      u = [1.0, 2.0, 3.0]
      v = [4.0, 0.0, -1.0]
      Addition (u + v)        -> [5.0, 2.0, 2.0]
      Scalar multiply (2 * u) -> [2.0, 4.0, 6.0]
      L2 norm of u (||u||₂)   -> 3.7417
      L1 norm of u (||u||₁)   -> 6.0000
      Dot product (u · v)     -> 1.0000
      Cosine similarity       -> 0.0648
      Angle between u, v.     -> 86.28°
      Unit vector of u (û)    -> [0.2673, 0.5345, 0.8018]
      Verify `||û||₂ = 1`     -> 1.000000
      =============================
      Semantic similarity mini-demo
      =============================
      In real NLP, these would be word embeddings. Here we illustrate
      the principle with handcrafted feature vectors.
      Features: [royalty_score, masculinity, age, power]
      king  = [0.9, 0.9, 0.8, 0.9]
      queen = [0.9, 0.1, 0.8, 0.9]
      man   = [0.0, 0.9, 0.5, 0.4]
      woman = [0.0, 0.1, 0.5, 0.4]
      The famous analogy: king - man + woman ≈ queen
      king − man + woman -> [0.9, 0.1, 0.8, 0.9]
      Cosine similarity to 'queen': 1.0000
      Cosine similarity to 'king':  0.8902
      ==> The analogy vector is closer to 'queen' than 'king'.
      ```
    {{< /details >}}
  {{< /tab >}}
  {{< tab name="Numpy" >}}
    {{< details title="Code" closed="true" >}}
      {{< codeimporter
          url="https://raw.githubusercontent.com/learn-software-engineering/examples/refs/heads/main/ai/math/algebra/vectors/vector_numpy.en.py"
          type="python"
          >}}
    {{< /details >}}
    {{< details title="Output" closed="true" >}}
      ```bash
      >  python vector_numpy.en.py
      ============================
      Vector Operations with Numpy
      ============================
      u = [1. 2. 3.]
      v = [ 4.  0. -1.]
      Addition (u + v)        -> [5. 2. 2.]
      Scalar multiply (2 * u) -> [2. 4. 6.]
      L2 norm of u (||u||₂)   -> 3.7417
      L1 norm of u (||u||₁)   -> 6.0000
      Dot product (u · v)     -> 1.0000
      Cosine similarity       -> 0.0648
      Angle between u, v.     -> 86.28°
      Unit vector of u (û)    -> [np.float64(0.2673), np.float64(0.5345), np.float64(0.8018)]
      Verify `||û||₂ = 1`     -> 1.000000
      =============================
      Semantic similarity mini-demo
      =============================
      In real NLP, these would be word embeddings. Here we illustrate
      the principle with handcrafted feature vectors.
      Features: [royalty_score, masculinity, age, power]
      king  = [0.9 0.9 0.8 0.9]
      queen = [0.9 0.1 0.8 0.9]
      man   = [0.  0.9 0.5 0.4]
      woman = [0.  0.1 0.5 0.4]
      The famous analogy: king - man + woman ≈ queen
      king − man + woman -> [np.float64(0.9), np.float64(0.1), np.float64(0.8), np.float64(0.9)]
      Cosine similarity to 'queen': 1.0000
      Cosine similarity to 'king':  0.8902
      ==> The analogy vector is closer to 'queen' than 'king'.
      ```
    {{< /details >}}
  {{< /tab >}}
{{< /tabs >}}

## Machine Learning and AI perspective

Vectors are not a preliminary concept you'll graduate from, they are the *lingua franca* of modern AI research. Here are three ways they appear in the context of Artificial Intelligence and Machine Learning.

**Embedding spaces and representation learning**. Every modern deep learning model learns to represent inputs as vectors. The Transformer's token embeddings described in the [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) paper are vectors in \(\mathbb{R}^{d_{model}}\) (typically between 512 and 4096 dimensions). The entire training process can be viewed as optimizing the geometry of these vectors so that semantically similar inputs cluster together. Research on [*contrastive learning*](https://arxiv.org/abs/2002.05709) explicitly frames the learning objective in terms of `pushing similar sample vectors together and dissimilar sample vectors apart in embedding space`.

**Retrieval-Augmented Generation (RAG) and vector databases**. With the explosion of LLMs, a major applied-research direction is efficient nearest-neighbour search over billions of vectors. In the paper, [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html), Lewis et al. showed that augmenting generation with retrieved document vectors dramatically improves factuality. The entire retrieval step is cosine similarity search, the formula we saw before.

**Dimensionality**. The geometry of high-dimensional spaces is deeply counter-intuitive, a phenomenon called the *curse of dimensionality*. In very high dimensions, almost all pairs of vectors become nearly orthogonal (\(\cos\theta \approx 0\)), which can degrade cosine similarity as a meaningful metric. Understanding *when* cosine similarity breaks down and what geometric alternatives exist ([hyperbolic spaces](https://en.wikipedia.org/wiki/Hyperbolic_space), [Riemannian manifolds](https://en.wikipedia.org/wiki/Riemannian_manifold)) is an active research area. If this interests you, look up [*Poincaré Embeddings*](https://arxiv.org/abs/1705.08039).

## Common pitfalls and debugging

1. **Forgetting to normalize before computing cosine similarity, and when NOT to**. Cosine similarity measures direction only, discarding magnitude. If two documents are similar but one is ten times longer, cosine similarity ignores the length difference. Sometimes that's a bug (when magnitude matters), sometimes a feature (sentiment classification where you want topic, not verbosity). Always ask: should magnitude matter here? If yes, use Euclidean distance instead.

2. **Dimension mismatch silently producing wrong results**. In NumPy, `np.dot(u, v)` will raise an error if dimensions don't match for 1D arrays, but with 2D arrays (matrices), NumPy may broadcast in ways that give a result with the wrong shape. Always `.assert u.shape == v.shape` before dot products in research code, or use `np.einsum` with explicit index notation for clarity.

3. **The zero-vector edge case**. `cosine_similarity([0,0,0], [1,2,3])` is mathematically undefined (you're dividing by zero). In production systems that compute embeddings, a zero vector usually signals a bug upstream: an empty input, a bad tokenisation, or a collapsed network layer. If you see `NaN` losses, check your embedding norms first.

4. **Floating-point precision in `arccos`**. Due to floating-point arithmetic, dot products can occasionally yield cosine values slightly outside \([-1, 1]\) (e.g., `1.0000000002`). Passing this directly to `math.acos()` raises a `ValueError`. Always clamp: `cos_theta = max(-1.0, min(1.0, cos_theta))` before calling `arccos`.

5. **Confusing \(L^1\) and \(L^2\) regularisation effects**. \(L^2\) regularisation (weight decay) shrinks all weights proportionally, it never drives weights exactly to zero. \(L^1\) regularisation does drive weights to zero, creating sparse models. This is a direct consequence of the geometry of \(L^1\) vs \(L^2\) norm balls. Choosing the wrong regulariser is a common source of models that are either too dense (wasted computation) or not sparse enough (poor interpretability).

## Summary and what's next

**Key takeaways from this article**:

- A **vector** is an ordered tuple of scalars that lives in a geometric space, it has both magnitude and direction.
- A **vector space** is defined by closure under addition and scalar multiplication, this is why linear operations in neural networks are so well-behaved.
- The **\(L^2\) norm** measures Euclidean length. The **\(L^1\) norm** measures Manhattan distance and promotes sparsity.
- The **dot product** (\(\mathbf{u} \cdot \mathbf{v}\)) measures alignment. Dividing by both norms gives **cosine similarity**, the angle-based similarity metric at the core of retrieval, embeddings, and attention mechanisms.
- The formula \(\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}\) is derived from the Law of Cosines, it's not arbitrary, it's geometry.
- High-dimensional vectors are the language in which all of modern Machine Learning is written. Fluency here compounds across every future topic.

**Coming up in (Matrices & Linear Transformations)**: We generalize from single vectors to collections of vectors, introduce matrices as functions that transform vector spaces, and derive the rules of matrix multiplication from first principles. You'll see exactly why a fully-connected neural network layer is nothing more than a matrix multiplication followed by a nonlinearity.

---

{{< callout icon="sparkles" >}}
Cheers for making it this far! I hope this journey through the AI universe has been as fascinating for you as it was for me to write down.

We're keen to hear your thoughts, so don't be shy, drop your comments, suggestions, and those bright ideas you're bound to have.

You'll find all the code and projects in our GitHub repository [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Thanks for being part of this learning community. Keep coding and exploring new territories in this captivating world of computing!
{{< /callout >}}
