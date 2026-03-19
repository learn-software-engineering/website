---
weight: 2
title: "Matrices: operations and properties"
authors:
  - jnonino
description: >
  Master matrix arithmetic, linear transformations, determinants, rank, and inverses from scratch. Step-by-step math derivations, runnable Python code, and research-grade insights for aspiring Machine Learning (ML) scientists.
date: 2026-03-13
tags: ["AI", "Maths", "Algebra", "Linear Algebra", "Matrices"]
---

GPT-3 stores its *"intelligence"* in approximately 1800 weight matrices. Every token you type triggers a cascade of multiplications across those matrices (projections, attention scores, feed-forward expansions) before a single output probability is produced. The entire 175-billion-parameter model is, at its computational core, an intricate composition of matrix operations applied in sequence.

Matrices are where the abstract idea of a vector space becomes an engine you can actually run. A vector tells you where something is; a matrix tells you how to move it. Weight matrices in neural networks encode learned transformations of feature space. Covariance matrices in statistics encode the shape of a data distribution. The Jacobian matrix of a function encodes how its outputs change with respect to its inputs, it is the object that makes *backpropagation* possible.

The trouble is that most introductions to matrices treat them as glorified spreadsheets: grids of numbers with arithmetic rules bolted on. That framing makes the operations feel arbitrary. This article takes the opposite approach: we start from the geometry, derive every operation from first principles, and by the end you will see matrices not as tables but as **functions**, with all the richness that implies: composition, invertibility, image, kernel, and rank.

By the end of this article, you will be able to:

- Formally define a matrix and a linear transformation, and explain why the axioms matter.
- Execute and derive all fundamental matrix operations: addition, multiplication, transpose, trace, and inversion.
- Compute the determinant geometrically and algebraically, and understand what a zero determinant means.
- Reason about column space, row space, null space, and rank. And state and apply the Rank-Nullity Theorem.
- Implement these operations from scratch in pure Python and NumPy.
- Connect these operations to contemporary research in deep learning.

Let's start.

## Prerequisites

Before reading this article, you should be comfortable with:

- **Vectors and vector spaces**: formal definitions, dot products, norms, span, and linear combinations.
- **Basic Python and NumPy**: array creation, indexing, shape manipulation.
- **Function notation**: understanding \(f: \mathbb{R}^m \rightarrow \mathbb{R}^n\) style notation.

If you can define what it means for a set to be *closed* under an operation, you have enough scaffolding.

## Intuition first

### The programmer's analogy: matrices as functions

As a developer, you have used functions your whole career. A function `transform(x)` takes an input and maps it to an output. A matrix \(\mathbf{A}\) is exactly that, a **function** that takes a vector as input and produces a vector as output. The key constraint is that this function must be *linear*, which imposes a specific geometric structure on what transformations are permitted.

Think of it this way. Imagine a data pipeline where user feature vectors pass through processing stages:

```python
# Stage 1: expand 3 raw features into 5 derived features
stage1 = transform_3_to_5(user_features)   # shape: (5,)

# Stage 2: compress 5 derived features into 2 latent dimensions
stage2 = transform_5_to_2(stage1)           # shape: (2,)

# Combined: can we do both in one step?
combined = transform_3_to_2(user_features)  # Yes — matrix multiplication
```

This is exactly what matrix multiplication computes: the **composition** of two linear transformations into one. A \(5 \times 3\) matrix (stage 1) composed with a \(2 \times 5\) matrix (stage 2) produces a single \(2 \times 3\) matrix that does both steps at once. Every layer of a neural network is one stage of this pipeline.

### Geometric picture: matrices as space transformations

Picture a 2D coordinate system. The [standard basis](https://en.wikipedia.org/wiki/Standard_basis) vectors are:

$$
\hat{e}_1 = [1,0] \qquad \text{(points right along x-axis)}
$$

$$
\hat{e}_2 = [0,1] \qquad \text{(points up along y-axis)}
$$

Apply the matrix \(\mathbf{A} = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}\). After the transformation: \(\hat{e}_1\) maps to \([2,0]\) (stretched right by factor 2) and \(\hat{e}_2\) maps to \([0,3]\) (stretched up by factor 3). The unit square (with area 1), becomes a \(2 \times 3\) rectangle with area 6. The **determinant** of \(\mathbf{A}\), which we will derive shortly, equals exactly 6. This is not a coincidence: the determinant *is* the volume scaling factor.

Compare this to a rotation matrix:

$$
\mathbf{R}(\theta) = \begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{bmatrix}
$$

This rotates every vector by angle $\theta$ without stretching, so it preserves all areas — and its determinant is always $1$.

The columns of any matrix tell you exactly where the basis vectors land. Since every vector is a linear combination of the basis vectors, knowing where basis vectors go tells you where *every* vector in the space goes. This insight explains why matrix multiplication is non-commutative (applying transformation A then B is geometrically different from B then A), and why the columns of weight matrices in neural networks carry semantic meaning that researchers actively analyse.

{{< callout type="important" >}}
The columns of a matrix are not just numbers, they are the *images of the basis vectors* under the transformation. When you look at the weight matrix \(\mathbf{W}\) of a trained neural network layer, each column tells you how that layer responds to one standard input direction. This is the foundation of feature visualization research, where practitioners interpret what each neuron *"detects*" by examining the directions in weight matrices.
{{< /callout >}}

## Mathematical derivation

### Formal definition

An **\(m \times n\) matrix** over \(\mathbb{R}\) is a rectangular array of real numbers arranged in \(m\) rows and \(n\) columns:

$$
\mathbf{A} = \begin{bmatrix} A_{11} & A_{12} & \cdots & A_{1n} \\ A_{21} & A_{22} & \cdots & A_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ A_{m1} & A_{m2} & \cdots & A_{mn} \end{bmatrix} \in \mathbb{R}^{m \times n}
$$

The entry \(A_{ij}\) denotes the element in **row \(i\), column \(j\)**. We write the set of all \(m \times n\) real matrices as \(\mathbb{R}^{m \times n}\).

{{< callout type="info" >}}
In plain English: a matrix is a 2D table of numbers. The notation \(\mathbb{R}^{m \times n}\) is the *"type"* of the matrix (how many rows and columns it has). This is exactly the type signature you would attach to a 2D array in a statically typed language: `Array[Float, m, n]`.
{{< /callout >}}

**Special cases** you will encounter constantly:

- **Column vector**: a matrix of shape \(n \times 1\), this is just a vector \(mathbf{v} \in \mathbb{R}^n\).
- **Row vector**: a matrix of shape \(1 \times n\).
- **Square matrix**: a matrix where \(m = n\).
- **Identity matrix** \(\mathbf{I}_n\): the \(n \times n\) matrix with \(I_{ij} = 1\) if \(i = j\), else \(0\).
- **Zero matrix** \(\mathbf{0}\): all entries are zero.
- **Diagonal matrix**: a square matrix where \(A_{ij} = 0\) for all \(i \neq j\).

### Matrices as linear transformations

The most important property of a matrix is that it defines a **linear transformation** \(T: \mathbb{R}^n \rightarrow \mathbb{R}^m\) via \(T(\mathbf{x}) = \mathbf{A}\mathbf{x}\).

A function \(T\) is called **linear** if and only if it satisfies two axioms for all vectors \(\mathbf{u}, \mathbf{v} \in \mathbb{R}^n\) and all scalars \(\alpha \in \mathbb{R}\):

- *Additivity*: \(T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})\)
- *Homogeneity*: \(T(\alpha \mathbf{u}) = \alpha T(\mathbf{u})\)

These two axioms are equivalent to the single condition:

$$
T(\alpha \mathbf{u} + \beta \mathbf{v}) = \alpha T(\mathbf{u}) + \beta T(\mathbf{v})
$$

{{< callout type="info" >}}
In plain English: a linear transformation preserves the structure of the vector space, it doesn't matter whether you add two vectors first and then transform, or transform each separately and add. This is why stacking multiple linear (un-activated) layers in a neural network collapses to a single matrix product: \(\mathbf{W}_2(\mathbf{W}_1\mathbf{x}) = (\mathbf{W}_2\mathbf{W}_1)\mathbf{x}\).
{{< /callout >}}

A foundational theorem from linear algebra states that **every** linear transformation between finite-dimensional spaces can be represented as a matrix, and conversely every matrix defines a linear transformation. The matrix and the linear transformation are, for our purposes, the same object.

### Matrix addition and scalar multiplication

The set \(\mathbb{R}^{m \times n}\) of all \(m \times n\) real matrices forms a **vector space** under the following operations.

**Matrix addition**: Given \(\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}\):

$$
(\mathbf{A} + \mathbf{B})_{ij} = A_{ij} + B_{ij}
$$

This satisfies:
- *Commutativity*: \(\mathbf{A} + \mathbf{B} = \mathbf{B} + \mathbf{A}\)
- *Associativity*: \((\mathbf{A} + \mathbf{B}) + \mathbf{C} = \mathbf{A} + (\mathbf{B} + \mathbf{C})\)
- *Identity element*: there exists a zero matrix \(\mathbf{0}\) such that \(\mathbf{A} + \mathbf{0} = \mathbf{A}\)
- *Inverse element*: for each \(\mathbf{A}\), there exists \(-\mathbf{A}\) such that \(\mathbf{A} + (-\mathbf{A}) = \mathbf{0}\)

**Scalar multiplication**: Given \(\alpha \in \mathbb{R}\) and \(\mathbf{A} \in \mathbb{R}^{m \times n}\):

$$(\alpha \mathbf{A})_{ij} = \alpha \cdot A_{ij}$$

This satisfies:
- *Associativity*: \(\alpha(\beta \mathbf{A}) = (\alpha\beta)\mathbf{A}\)
- *Identity element*: \(1 \cdot \mathbf{A} = \mathbf{A}\)
- *Distributivity over matrix addition*: \(\alpha(\mathbf{A} + \mathbf{B}) = \alpha\mathbf{A} + \alpha\mathbf{B}\)
- *Distributivity over scalar addition*: \((\alpha + \beta)\mathbf{A} = \alpha\mathbf{A} + \beta\mathbf{A}\)

{{< callout >}}
If these axioms look familiar, they should. They are exactly the vector space axioms from our previous lesson about vectors. Matrices *"are"* vectors in a higher-dimensional space. A \(3 \times 4\) matrix is just a vector with \(12\) components arranged in a grid. This means every theorem we prove about vector spaces applies to matrices too.
{{< /callout >}}

### Matrix multiplication

Given \(\mathbf{A} \in \mathbb{R}^{m \times k}\) and \(\mathbf{B} \in \mathbb{R}^{k \times n}\), their product \(\mathbf{C} = \mathbf{A}\mathbf{B} \in \mathbb{R}^{m \times n}\) is defined as:

$$\boxed{C_{ij} = \sum_{l=1}^{k} A_{il} \cdot B_{lj}}$$

To compute entry \(C_{ij}\), the element in row \(i\), column \(j\) of the result, take the **dot product of row \(i\) of \(\mathbf{A}\) with column \(j\) of \(\mathbf{B}\)**. The inner dimensions must match:

$$
\underbrace{\mathbf{A}}_{m \times k} \cdot \underbrace{\mathbf{B}}_{k \times n} = \underbrace{\mathbf{C}}_{m \times n}
$$

{{< callout type="info" >}}
In plain English: matrix multiplication is function composition. Applying transformation \(\mathbf{B}\) first, then \(\mathbf{A}\), gives the same result as the single composed transformation \(\mathbf{AB}\). The inner dimensions must agree because the output dimension of the first transformation must equal the input dimension of the second, exactly like composing functions.
{{< /callout >}}

Matrix multiplication satisfies:
- *Associativity*: \((\mathbf{AB})\mathbf{C} = \mathbf{A}(\mathbf{BC})\)
- *Left distributivity*: \(\mathbf{A}(\mathbf{B} + \mathbf{C}) = \mathbf{AB} + \mathbf{AC}\)
- *Right distributivity*: \((\mathbf{A} + \mathbf{B})\mathbf{C} = \mathbf{AC} + \mathbf{BC}\)
- *Identity*: \(\mathbf{A}\mathbf{I} = \mathbf{I}\mathbf{A} = \mathbf{A}\) (for compatible \(\mathbf{I}\))
- **Non-commutativity**: \(\mathbf{AB} \neq \mathbf{BA}\) in general

{{< callout >}}
The non-commutativity is not a flaw but a fundamental feature of transformations. Rotating a figure then reflecting it is geometrically different from reflecting then rotating.
{{< /callout >}}

### The transpose

The **transpose** of \(\mathbf{A} \in \mathbb{R}^{m \times n}\) is the matrix \(\mathbf{A}^\top \in \mathbb{R}^{n \times m}\) defined by:

$$
\boxed{(\mathbf{A}^\top)_{ij} = A_{ji}}
$$

{{< callout type="info" >}}
In plain English: flip the matrix along its main diagonal, rows become columns and columns become rows. A \(3 \times 5\) matrix becomes a \(5 \times 3\) matrix. Geometrically, the transpose corresponds to the *adjoint* of the transformation, the transformation that "undoes the rotation part" while keeping the scaling.
{{< /callout >}}

The transpose satisfies the following properties:

- *Double transpose*: \((\mathbf{A}^\top)^\top = \mathbf{A}\)
- *Linearity*: \((\mathbf{A} + \mathbf{B})^\top = \mathbf{A}^\top + \mathbf{B}^\top\) and \((\alpha\mathbf{A})^\top = \alpha\mathbf{A}^\top\)
- *Product reversal*: \((\mathbf{AB})^\top = \mathbf{B}^\top \mathbf{A}^\top\)

The product reversal property is **critical** and appears in every backpropagation derivation. Let us prove it completely. We want to show \([(\mathbf{AB})^\top]_{ij} = [\mathbf{B}^\top \mathbf{A}^\top]_{ij}\).

Starting from the definition of transpose:

$$
[(\mathbf{AB})^\top]_{ij} = [\mathbf{AB}]_{ji}
$$

Applying the definition of matrix multiplication to \([\mathbf{AB}]_{ji}\):

$$
[\mathbf{AB}]_{ji} = \sum_{l} A_{jl} \cdot B_{li}
$$

Recognising each factor using the definition of transpose (\(A_{jl} = [\mathbf{A}^\top]_{lj}\) and \(B_{li} = [\mathbf{B}^\top]_{il}\)):

$$\sum_{l} A_{jl} B_{li} = \sum_{l} [\mathbf{B}^\top]_{il} \cdot [\mathbf{A}^\top]_{lj}$$

Recognising the right-hand side as the definition of matrix multiplication \( [\mathbf{B}^\top \mathbf{A}^\top]_{ij} \):

$$
\boxed{(\mathbf{AB})^\top = \mathbf{B}^\top \mathbf{A}^\top}
$$

**Symmetric and skew-symmetric matrices** are important special cases:

- \(\mathbf{A}\) is **symmetric** if \(\mathbf{A}^\top = \mathbf{A}\), equivalently \(A_{ij} = A_{ji}\) for all \(i, j\). Covariance matrices and kernel matrices in ML are always symmetric.
- \(\mathbf{A}\) is **skew-symmetric** if \(\mathbf{A}^\top = -\mathbf{A}\), equivalently \(A_{ij} = -A_{ji}\) and all diagonal entries are zero.

{{< callout >}}
Any square matrix \(\mathbf{A}\) can be uniquely decomposed into a symmetric part and a skew-symmetric part:

$$
\mathbf{A} = \underbrace{\frac{\mathbf{A} + \mathbf{A}^\top}{2}}_{\text{symmetric}} + \underbrace{\frac{\mathbf{A} - \mathbf{A}^\top}{2}}_{\text{skew-symmetric}}
$$

This decomposition has applications in physics (strain vs. rotation analysis) and in recent research on attention mechanism structure.
{{< /callout >}}

### The trace

For a square matrix \(\mathbf{A} \in \mathbb{R}^{n \times n}\), the **trace** is the sum of its diagonal entries:

$$
\text{tr}(\mathbf{A}) = \sum_{i=1}^{n} A_{ii}
$$

{{< callout type="info" >}}
In plain English: add up the main diagonal. The trace captures how much the transformation *"expands"* space on average.
{{< /callout >}}

Key properties:
- *Linearity*: \(\text{tr}(\mathbf{A} + \mathbf{B}) = \text{tr}(\mathbf{A}) + \text{tr}(\mathbf{B})\) and \(\text{tr}(\alpha\mathbf{A}) = \alpha\,\text{tr}(\mathbf{A})\)
- *Transpose invariance*: \(\text{tr}(\mathbf{A}^\top) = \text{tr}(\mathbf{A})\)
- **Cyclic property**: \(\text{tr}(\mathbf{ABC}) = \text{tr}(\mathbf{BCA}) = \text{tr}(\mathbf{CAB})\)

The cyclic property is everywhere in gradient derivations. Let us prove the base case \(\text{tr}(\mathbf{AB}) = \text{tr}(\mathbf{BA})\).

Starting from the definition of trace:

$$\text{tr}(\mathbf{AB}) = \sum_{i} [\mathbf{AB}]_{ii} = \sum_{i} \sum_{j} A_{ij} B_{ji}$$

Swapping the order of summation (valid since both sums are finite):

$$\sum_{i} \sum_{j} A_{ij} B_{ji} = \sum_{j} \sum_{i} B_{ji} A_{ij}$$

Recognising the right-hand side as the diagonal entry \([\mathbf{BA}]_{jj}\):

$$
\sum_{j} \sum_{i} B_{ji} A_{ij} = \sum_{j} [\mathbf{BA}]_{jj} = \text{tr}(\mathbf{BA})
$$

$$
\boxed{\text{tr}(\mathbf{AB}) = \text{tr}(\mathbf{BA})}
$$

{{< callout >}}
Note that this holds even when \(\mathbf{AB}\) and \(\mathbf{BA}\) have different shapes (e.g., \(\mathbf{A} \in \mathbb{R}^{m \times n}\), \(\mathbf{B} \in \mathbb{R}^{n \times m}\)), both traces are scalar and equal.
{{< /callout >}}

### The determinant

Before computing a single formula, it is worth asking: why does the determinant exist? What problem does it solve?

When a matrix encodes a linear transformation, the most fundamental question you can ask about it is: **does it destroy information?** A transformation that squashes a 2D plane into a 1D line has lost all information in the collapsed dimension, it cannot be undone. A transformation that rotates or stretches the plane preserves all information and can be reversed. The determinant is the single number that answers this question, and by how much.

More precisely, the determinant answers three interconnected questions simultaneously.

{{< details title="How does the transformation scale volumes?" closed="true" >}}
In two dimensions, the two columns of a matrix are simply two vectors drawn from the origin. Two vectors from the same origin always enclose a parallelogram — that is not a choice, it is a geometric inevitability. The determinant is the *signed area* of that parallelogram. If \(|\det(\mathbf{A})| = 6\), every region in the plane has its area multiplied by 6 after the transformation. In 3D, it becomes the signed volume of the parallelepiped formed by the three column vectors.

{{< figure
    src="images/volume-scale.en.svg"
    alt="Escala de volúmenes por el determinante"
    caption="The two columns of a \(2\times2\) matrix are vectors that always form a parallelogram. The determinant is its signed area."
    >}}
{{< /details >}}

{{< details title="Does the transformation preserve or reverse orientation?" closed="true" >}}
The word *signed* matters. Two vectors always form a parallelogram, but the *order* in which you traverse them determines the sign of the area. If the rotation from \(\mathbf{a}_1\) to \(\mathbf{a}_2\) is **anticlockwise**, the determinant is positive. If it is **clockwise**, the determinant is negative. The geometric content is identical (same shape, same area) but the sign encodes whether the transformation preserves or flips the handedness of space.

{{< figure
    src="images/orientation.en.svg"
    alt="Preservation or inversion of orientation by the determinant"
    caption="The order of the columns determines the sign of the determinant: positive if the rotation from the first column to the second is anticlockwise, negative if clockwise."
    >}}
{{< /details >}}

{{< details title="Is the transformation invertible?" closed="true" >}}
\(\det(\mathbf{A}) = 0\) means the two column vectors are parallel, they lie on the same line, spanning zero area. The transformation collapses a 2D plane into a 1D line. Information is irreversibly lost: infinitely many different input vectors produce the same output, so you cannot determine which input you came from. The matrix is **singular** and has no inverse.

{{< figure
    src="images/invertible.en.svg"
    alt="Invertibility of the transformation"
    caption="A matrix is invertible if and only if its determinant is non-zero."
    >}}
{{< /details >}}

{{< callout type="important" >}}
In ML, checking near-zero determinants is not a mathematical formality, it is a practical debugging step. Covariance matrices that are nearly singular cause numerical instability in [Gaussian processes](https://en.wikipedia.org/wiki/Gaussian_process), linear regression normal equations, and dimensionality reduction. When a model produces `NaN` outputs or wildly large predictions after a matrix inversion, a near-zero determinant is usually the culprit.
{{< /callout >}}

#### The 2x2 case, derivation from geometry

Let \(\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}\). The two column vectors are \(\mathbf{a}_1 = \begin{bmatrix}a \\ c\end{bmatrix}\) and \(\mathbf{a}_2 = \begin{bmatrix}b \\ d\end{bmatrix}\).

The determinant is the signed area of the parallelogram they span which has four vertices: \(O = (0,0)\), \(A = (a,c)\), \(B = (b,d)\), and \(C = (a+b,\, c+d)\).

$$
\boxed{\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc}
$$

We derive this in two complementary ways: first geometrically via bounding-box subtraction, then algebraically via the *Shoelace* formula.

{{< details title="Bounding-box subtraction" closed="true" >}}
Enclose the parallelogram in the smallest axis-aligned rectangle that contains it: a box of width \(a+b\) and height \(c+d\). The area of this bounding rectangle is \((a+b)(c+d)\).

The region between the rectangle and the parallelogram consists of exactly six pieces:

- **Two rectangles** with sides \(b\) and \(c\) each (top-left and bottom-right corners).
- **Two triangles** with legs \(a\) and \(c\) each (top and bottom triangles).
- **Two triangles** with legs \(b\) and \(d\) each (left and right triangles).

{{< figure
    src="images/parallelogram-area.en.svg"
    alt="Area of the parallelogram"
    caption="The area of the parallelogram is the difference between the bounding rectangle and the area of the triangles."
    >}}

Subtracting all of this from the bounding rectangle we'll get the area of the parallelogram:

$$
\begin{aligned}
  S &= (a+b)(c+d) - 2(bc) - 2\left(\frac{1}{2}ac\right) - 2\left(\frac{1}{2}bd\right) \\
  S &= ac + ad + bc + bd - 2(bc) - 2\left(\frac{1}{2}ac\right) - 2\left(\frac{1}{2}bd\right) \\
  S &= ac + ad + bc + bd - 2(bc) - ac - bd \\
  S &= ad - bc
\end{aligned}
$$
{{< /details >}}

{{< details title="Shoelace formula" closed="true" >}}
The **Shoelace formula** gives the signed area of any polygon whose vertices are listed in order. For a polygon with \(n\) vertices \((x_1, y_1), \ldots, (x_n, y_n)\), traversed anticlockwise:

$$
\text{área con signo} = \frac{1}{2}\sum_{i=1}^{n}\bigl(x_i y_{i+1} - x_{i+1} y_{i}\bigr)
$$

where indices are cyclical, so \(x_{n+1} = x_1\), \(y_{n+1} = y_1\), etc. and \(n\) is the number of vertices.

The intuition behind each term \(x_i y_{i+1} - x_{i+1} y_i\) is that it measures the signed area of the triangle formed by the origin, vertex \(i\), and vertex \(i+1\). Summing these triangles around the polygon gives the total signed area, positive for anticlockwise traversal, negative for clockwise.

Apply this to the parallelogram with vertices in anticlockwise order:

$$
O = (0,0), \quad A = (a,c), \quad B = (b,d), \quad C = (a+b,\, c+d)
$$

Expanding the sum term by term:

**Term 1** (\(O \to A\))
$$
\begin{aligned}
  x_{\mathbf{O}}\, y_A - x_A\, y_{\mathbf{O}} &= 0 \cdot c - a \cdot 0 \\
  x_{\mathbf{O}}\, y_A - x_A\, y_{\mathbf{O}} &= 0
\end{aligned}
$$

**Term 2** (\(A \to C\))
$$
\begin{aligned}
  x_A\, y_C - x_C\, y_A &= a(c+d) - (a+b)c \\
  x_A\, y_C - x_C\, y_A &= ac + ad - ac - bc \\
  x_A\, y_C - x_C\, y_A &= ad - bc
\end{aligned}
$$

**Term 3** (\(C \to B\))
$$
\begin{aligned}
  x_C\, y_B - x_B\, y_C &= (a+b)d - b(c+d) \\
  x_C\, y_B - x_B\, y_C &= ad + bd - bc - bd \\
  x_C\, y_B - x_B\, y_C &= ad - bc
\end{aligned}
$$

**Term 4** (\(B \to O\))
$$
\begin{aligned}
  x_B\, y_O - x_O\, y_B &= b \cdot 0 - 0 \cdot d \\
  x_B\, y_O - x_O\, y_B &= 0
\end{aligned}
$$

Summing and applying the \(\tfrac{1}{2}\) factor:
$$
\begin{aligned}
  \text{signed area} &= \frac{1}{2}\bigl[0 + (ad - bc) + (ad - bc) + 0\bigr] \\
  \text{signed area} &= \frac{1}{2} \cdot 2(ad-bc) \\
  \text{signed area} &= ad - bc
\end{aligned}
$$

This derivation makes no assumption about the signs of \(a, b, c, d\), it holds for any real entries. The result is positive when the traversal \(\mathbf{O} \to A \to C \to B\) is anticlockwise (i.e., when \(ad > bc\)), and negative when it is clockwise. This is precisely the sign that encodes orientation.
{{< /details >}}

### Linear independence and column/row spaces

Understanding a matrix means understanding the *spaces* it creates and destroys.

A set of vectors \(\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}\) is **linearly independent** if the only solution to:

$$
\alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \cdots + \alpha_k \mathbf{v}_k = \mathbf{0}
$$

is \(\alpha_1 = \alpha_2 = \cdots = \alpha_k = 0\). In other words, no vector in the set can be written as a combination of the others. Linearly dependent columns are the signature of a rank-deficient matrix.

For \(\mathbf{A} \in \mathbb{R}^{m \times n}\), three fundamental subspaces are defined:

**Column space** (range), \(\text{col}(\mathbf{A}) \subseteq \mathbb{R}^m\) is the space generated by the columns of \(\mathbf{A}\). Equivalently, \(\text{col}(\mathbf{A}) = \{\mathbf{A}\mathbf{x} : \mathbf{x} \in \mathbb{R}^n\}\).

**Row space**, \(\text{row}(\mathbf{A}) \subseteq \mathbb{R}^n\): the span of the rows of \(\mathbf{A}\). Equivalently, \(\text{row}(\mathbf{A}) = \text{col}(\mathbf{A}^\top)\).

**Null space** (kernel), \(\text{null}(\mathbf{A}) \subseteq \mathbb{R}^n\): all input vectors that \(\mathbf{A}\) maps to zero:

$$\text{null}(\mathbf{A}) = \{\mathbf{x} \in \mathbb{R}^n : \mathbf{A}\mathbf{x} = \mathbf{0}\}$$

{{< callout type="info" >}}
In plain English: the null space is the "blind spot" of the transformation. Any vector in the null space is invisible to the matrix — it produces zero output signal regardless of how large it is. In a neural network context, directions in the null space of a weight matrix receive no gradient from that layer and will never be updated by that layer's gradients alone.
{{< /callout >}}
