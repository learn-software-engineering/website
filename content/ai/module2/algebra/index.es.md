---
weight: 1
series: ["Matemática para Machine Learning"]
series_order: 1
title: "Álgebra Lineal para Machine Learning: vectores y matrices que todo AI engineer debe conocer"
description: "Domina los fundamentos matemáticos del machine learning: vectores, matrices y transformaciones lineales explicados desde la perspectiva del programador."
authors:
  - jnonino
date: 2025-09-02
tags: ["Inteligencia Artificial", "Aprendizaje Automático", "Machine Learning", "Matemática", "Álgebra Lineal", "Vectores", "Matrices"]
---
{{< katex >}}

{{< lead >}}
En el módulo anterior exploramos los fundamentos conceptuales de la Inteligencia Artificial. Ahora es momento de sumergirnos en una de las áreas de la matemática que hacen posible que los algoritmos de Machine Learning funcionen: el **álgebra lineal**.

Si eres como la mayoría de los ingenieros de software, probablemente te preguntes: "¿por qué necesito álgebra lineal para programar IA?" La respuesta es simple pero profunda: **el álgebra lineal es el lenguaje nativo del machine learning**.

{{< /lead >}}

---

## ¿Por qué el Álgebra Lineal es crucial en IA?

Imagina que estás desarrollando un sistema de recomendaciones para Netflix. Cada usuario tiene preferencias (acción, comedia, drama) que pueden representarse como un vector. Cada película también tiene características (género, año, rating) que forman otro vector. El problema de recomendar películas se convierte en encontrar similitudes entre vectores: **álgebra lineal**.

O considera una red neuronal procesando una imagen de \(224x224\) píxeles. Esa imagen se convierte en un vector de \(50176\) elementos. Las operaciones de la red (convoluciones, transformaciones) son multiplicaciones de matrices. El entrenamiento optimiza estas matrices, otra vez: **álgebra lineal**.

### Los tres pilares del ML que dependen del Álgebra Lineal

1. **Representación de Datos**: Todo en ML se convierte en vectores y matrices
2. **Transformaciones**: Los algoritmos manipulan datos mediante operaciones lineales
3. **Optimización**: Los métodos de entrenamiento usan gradientes (derivadas de operaciones matriciales)

Como programadores, estamos acostumbrados a pensar en estructuras de datos como arrays, listas o objetos. En machine learning, pensamos en **vectores** y **matrices**. En este módulo aprenderemos a hacer esa transición mental.

---

## Vectores: más que arrays

Un vector no es simplemente un array de números. Es una entidad matemática que representa tanto **magnitud** como **dirección**. En el contexto de machine learning, un vector es una forma de codificar información.

### Definición formal

Un vector \(v\) en el espacio \(R^n\) es una tupla ordenada de \(n\) números reales:

$$v = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}$$

Pero más importante que la definición formal es la **interpretación práctica**:

- **En un sistema de recomendaciones**:
$$v = \begin{pmatrix} rating_{accion} \\ rating_{comedia} \\ rating_{drama} \end{pmatrix}$$
- **En procesamiento de texto**:
$$v = \begin{pmatrix} frecuencia_{palabra1} \\ frecuencia_{palabra2} \\ frecuencia_{palabra3} \\ \vdots \end{pmatrix}$$
- **En visión por computadora**:
$$v = \begin{pmatrix} pixel_1 \\ pixel_2 \\ pixel_3 \\ \vdots \end{pmatrix}$$

### Interpretación geométrica

Un vector en dos dimensiones (2D) se puede visualizar como una flecha desde el origen \((0,0)\) hasta el punto \((v_1, v_2)\). Esta visualización es clave para entender las operaciones vectoriales.

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_vector(vector, color='blue', label='Vector'):
    plt.quiver(0, 0, vector[0], vector[1],
               angles='xy', scale_units='xy', scale=1,
               color=color, label=label, width=0.005)
    plt.xlim(-1, 5)
    plt.ylim(-1, 5)
    plt.grid(True)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)

# Ejemplo: vector que representa preferencias de usuario
user_preferences = np.array([3, 4])  # [acción: 3, comedia: 4]
plt.figure(figsize=(8, 6))
plot_vector(user_preferences, 'blue', 'Preferencias Usuario')
plt.xlabel('Rating Acción')
plt.ylabel('Rating Comedia')
plt.title('Vector de Preferencias de Usuario')
plt.legend()
plt.show()
```

### Operaciones vectoriales fundamentales

#### Suma de vectores

La suma vectorial es **componente a componente**:

$$\mathbf{u} + \mathbf{v} = \begin{pmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{pmatrix}$$

*Interpretación en ML*: Si tenemos las preferencias de dos usuarios similares, podemos promediar sus vectores para encontrar preferencias "típicas" de ese segmento.

#### Multiplicación por escalar

$$c \cdot \mathbf{v} = \begin{pmatrix} c \cdot v_1 \\ c \cdot v_2 \\ \vdots \\ c \cdot v_n \end{pmatrix}$$

*Interpretación en ML*: Amplificar o reducir la importancia de ciertas características.

#### Producto Punto

El producto punto es quizás la operación más importante en ML:

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = u_1 v_1 + u_2 v_2 + ... + u_n v_n$$

**¿Por qué es tan importante?**
- **Similitud**: Vectores similares tienen productos punto altos
- **Proyección**: Mide cuánto un vector "apunta" en la dirección de otro
- **Redes neuronales**: La base de las operaciones en cada neurona

La interpretación geométrica es crucial, el producto punto es igual al producto entre las magnitudes de cada vector y el coseno del ángulo entre ellos:

$$\mathbf{u} \cdot \mathbf{v} = ||\mathbf{u}|| \cdot ||\mathbf{v}|| \cdot \cos(\theta)$$

Donde \(\theta\) es el ángulo entre los vectores.

O de otra manera:

$$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{||\mathbf{u}|| \cdot ||\mathbf{v}||}$$

$$\theta = \arccos(\frac{\mathbf{u} \cdot \mathbf{v}}{||\mathbf{u}|| \cdot ||\mathbf{v}||})$$

Conocer el ángulo entre los vectores permite determinar que tan alineados están.

### Implementación desde cero: clase Vector

Antes de usar `NumPy`, implementemos nuestras propias operaciones vectoriales para entender qué sucede en el detrás de escena:

{{< codeimporter url="https://github.com/learn-software-engineering/examples/blob/main/ai/module2/vector.py" type="python" >}}

---

## Matrices: transformaciones de datos

Si los vectores representan datos, las **matrices representan transformaciones** de esos datos. Una matriz es una tabla rectangular de números organizados en filas y columnas.

$$\mathbf{A} = \begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{pmatrix}$$

**En machine learning, las matrices son omnipresentes:**

- **Dataset**: Cada fila es un ejemplo, cada columna una característica
- **Pesos de red neuronal**: Transforman entradas en salidas
- **Transformaciones**: Rotación, escalado, proyección de datos

### Operaciones matriciales fundamentales

#### Multiplicación Matriz-Vector

Esta es la operación más común en ML. Transforma un vector usando una matriz:

$$\mathbf{A}\mathbf{v} = \begin{pmatrix}
\sum_{i=1}^{n} a_{1i} v_i \\
\sum_{i=1}^{n} a_{2i} v_i \\
\sum_{i=1}^{n} a_{3i} v_i \\
\vdots \\
\sum_{i=1}^{n} a_{mi} v_i \\
\end{pmatrix}$$

$$\mathbf{A}\mathbf{v} = \begin{pmatrix}
a_{11} & a_{12} & a_{13} & \cdots & a_{1n} \\
a_{21} & a_{22} & a_{33} & \cdots & a_{2n} \\
a_{31} & a_{32} & a_{33} & \cdots & a_{3n} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & a_{m3} & \cdots & a_{mn} \\
\end{pmatrix} \cdot \begin{pmatrix} v_1 \\ v_2 \\ v_3 \\ \vdots \\ v_n \end{pmatrix}$$

$$\mathbf{A}\mathbf{v} = \begin{pmatrix}
a_{11} \cdot v_1 + a_{12} \cdot v_2 + a_{13} \cdot v_3 + \cdots + a_{1n} \cdot v_n \\
a_{21} \cdot v_1 + a_{22} \cdot v_2 + a_{23} \cdot v_3 + \cdots + a_{2n} \cdot v_n \\
a_{31} \cdot v_1 + a_{32} \cdot v_2 + a_{33} \cdot v_3 + \cdots + a_{3n} \cdot v_n \\
\vdots \\
a_{m1} \cdot v_1 + a_{m2} \cdot v_2 + a_{m3} \cdot v_3 + \cdots + a_{mn} \cdot v_n \\
\end{pmatrix}$$

Cuando se multiplica una matriz por un vector, es necesario que el número de elementos del vector coincida con el número de columnas de la matriz. Si no es así, la multiplicación no está definida.

**Ejemplo práctico**: En una red neuronal, cada capa aplica una transformación lineal:
```
salida = pesos × entrada + sesgo
```

#### Multiplicación Matriz-Matriz

Para que el producto de dos matrices \(A\) y \(B\) es decir, \(AB\) esté definido, la matriz \(A\) debe tener el mismo número de columnas que la matriz \(B\) tenga de filas. Si \(A\) es de tamaño \(m x n\) y \(B\) es de tamaño \(n x p\), entonces el resultado \(C = AB\) será una matriz de tamaño \(m x p\).

$$\mathbf{C} = \mathbf{A}\mathbf{B} =
\begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
a_{31} & a_{32} & \cdots & a_{3n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn} \\
\end{pmatrix}
\cdot
\begin{pmatrix}
b_{11} & b_{12} & \cdots & b_{1p} \\
b_{21} & b_{22} & \cdots & b_{2p} \\
b_{31} & b_{32} & \cdots & b_{3p} \\
\vdots & \vdots & \ddots & \vdots \\
b_{n1} & b_{n2} & \cdots & b_{np} \\
\end{pmatrix}$$

$$\mathbf{C} = \mathbf{A}\mathbf{B} = \begin{pmatrix}
\sum_{k=1}^{n} a_{1k} b_{k1} & \sum_{k=1}^{n} a_{1k} b_{k2} & \cdots & \sum_{k=1}^{n} a_{1k} b_{kp} \\
\sum_{k=1}^{n} a_{2k} b_{k1} & \sum_{k=1}^{n} a_{2k} b_{k2} & \cdots & \sum_{k=1}^{n} a_{2k} b_{kp} \\
\sum_{k=1}^{n} a_{3k} b_{k1} & \sum_{k=1}^{n} a_{3k} b_{k2} & \cdots & \sum_{k=1}^{n} a_{3k} b_{kp} \\
\vdots & \vdots & \ddots & \vdots \\
\sum_{k=1}^{n} a_{mk} b_{k1} & \sum_{k=1}^{n} a_{mk} b_{k2} & \cdots & \sum_{k=1}^{n} a_{mk} b_{kp} \\
\end{pmatrix}$$

Es decir, cada elemento \(ij\) de la matriz resultado (\(C\)) será:

$$\mathbf{C}_{ij} = (\mathbf{AB})_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}$$

Esta operación permite componer transformaciones lineales.

### Implementación desde cero: clase Matrix

{{< codeimporter url="https://github.com/learn-software-engineering/examples/blob/main/ai/module2/matrix.py" type="python" >}}

---

## Espacios Vectoriales y Transformaciones Lineales

Un espacio vectorial (o espacio lineal) es un conjunto no vacío de vectores, en el que se han definido dos operaciones: la suma de vectores y la multiplicación de un vector por un escalar (número real o complejo). Para que un conjunto sea considerado un espacio vectorial, debe cumplir con ciertos axiomas fundamentales.

1. Conmutatividad: \(u + v = v + u\)

2. Asociatividad: \((u + v) + w = u + (v + w)\)

3. Existencia del vector nulo: \(\exists \ v_0 \in V \ \;|\; \ v_0 + u = u \ \forall \ u \in V \)

4. Existencia del opuesto: \(\forall \ v_i \in V \ \exists \ -v_i \in V \ \;|\; \ v_i + (-v_i) = 0\)

5. Distributividad del producto respecto a la suma vectorial: \(\alpha (u + v) = \alpha u + \alpha v\)

6. Distributividad del producto respecto a la suma escalar: \((\alpha + \beta) u = \alpha u + \beta u\)

7. Asociatividad del producto de escalares: \(\alpha (\beta u) = (\alpha \beta) u\)

8. Elemento neutro: \(1 u = u \ \forall \ u \in V\)

Entre algunos ejemplos de espacios vectoriales podemos mencionar:
- Vectores en el plano: Los vectores en \(\mathbb{R}^2\) son un ejemplo clásico de espacio vectorial, donde cada vector se representa como un par ordenado \((x,y)\)
- Vectores en el espacio tridimensional: En \(\mathbb{R}^3\), un vector se puede escribir como \(V = \alpha i + \beta j + \gamma k \) donde \(i\), \(j\) y \(k\) son vectores base.

Los espacios vectoriales son fundamentales en diversas áreas, incluyendo matemáticas, física, ingeniería y ciencias de la computación, ya que permiten modelar y resolver problemas complejos mediante el uso de vectores y matrices.

**¿Por qué importa en Machine Learning?**
- **Características**: Cada dataset define un espacio vectorial
- **Modelos**: Los algoritmos de Machine Learning operan en estos espacios
- **Transformaciones**: Cambiamos de un espacio a otro para facilitar el aprendizaje

### Transformaciones lineales

Una transformación \(T: \mathbb{R}^n \rightarrow \mathbb{R}^m\) se define como una función que asigna a cada vector \(v\) en un espacio vectorial \(V\) un único vector \(w\) en otro espacio vectorial \(W\).

Para que \(T\) sea considerada lineal, debe cumplir dos condiciones fundamentales:
- *Adición*: Para cualquier par de vectores \(u\) y \(v\) en \(V\), se cumple que:
$$ T(u + v) = T(u) + T(v) $$
- *Homogeneidad*: Para cualquier escalar \(c\) y cualquier vector \(v\) en \(V\) se cumple que:
$$ T(c \ v) = c \ T(v) $$

*Toda transformación lineal se puede representar como una matriz.*

### Vectores y valores propios

Los valores propios o autovalores y los vectores propios o autovectores revelan las direcciones *"especiales"* de una transformación lineal.

Los vectores propios o autovectores de una transformación lineal son los vectores no nulos que, cuando son transformados, dan lugar a un múltiplo escalar de sí mismos, con lo que no cambian su dirección. Este escalar \(\lambda\) recibe el nombre de valor propio o autovalor. En muchos casos, una transformación queda completamente determinada por sus vectores propios y valores propios. Un espacio propio o autoespacio asociado al valor propio \(\lambda\) es el conjunto de vectores propios con un valor propio común.

Para una transformación lineal representada por la matriz \(\mathbf{A}\), un vector \(\mathbf{v}\) es un **vector propio** con **valor propio** \(\lambda\) si:

$$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$$

**Interpretación**: La transformación \(\mathbf{A}\) solo escala el vector \(\mathbf{v}\) por el factor \(\lambda\), sin cambiar su dirección.

> {{< figure
    src="img/Mona_Lisa_with_eigenvector.png"
    alt="Mona Lisa con auto vector"
    caption="Imagen de [J. Finkelstein](https://en.wikipedia.org/wiki/User:J._Finkelstein) y [Vb](https://en.wikipedia.org/wiki/User:Vb) on [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Mona_Lisa_with_eigenvector.png), dominio público"
    >}}
> En esta transformación de la Mona Lisa, la imagen se ha deformado de tal forma que su eje vertical no ha cambiado. El vector azul, representado por la flecha azul que va desde el pecho hasta el hombro, ha cambiado de dirección, mientras que el rojo, representado por la flecha roja, no ha cambiado. El vector rojo es entonces un vector propio o autovector de la transformación, mientras que el azul no lo es. Dado que el vector rojo no ha cambiado de longitud, su valor propio o autovalor es \(1\). Todos los vectores de esta misma dirección son vectores propios, con el mismo valor propio. Forman un subespacio del espacio propio de este valor propio.
>
> — <cite>Vector, valor y espacio propios. [Wikipedia](https://es.wikipedia.org/w/index.php?title=Vector,_valor_y_espacio_propios&oldid=159651633)</cite>

#### Aplicaciones en Machine Learning

1. **PCA (Análisis de Componentes Principales)**: Los vectores propios de la matriz de covarianza muestran las direcciones de mayor varianza en los datos.

2. **Reducción de dimensionalidad**: Proyectar datos en los vectores propios principales.

3. **Estabilidad de sistemas**: Los valores propios indican si un sistema dinámico es estable.

```python
def visualize_eigenvectors():
    """
    Visualiza conceptualmente los eigenvectores.
    Esta es una simplificación para matrices 2×2.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Matriz de ejemplo
    A = np.array([[3, 1], [0, 2]])

    # Calcular eigenvalores y eigenvectores usando NumPy
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Crear varios vectores para mostrar la transformación
    angles = np.linspace(0, 2*np.pi, 16)
    original_vectors = np.array([[np.cos(a), np.sin(a)] for a in angles])
    transformed_vectors = np.array([A @ v for v in original_vectors])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Vectores originales
    ax1.set_aspect('equal')
    for v in original_vectors:
        ax1.arrow(0, 0, v[0], v[1], head_width=0.05, head_length=0.1,
                 fc='blue', ec='blue', alpha=0.6)

    # Eigenvectores originales
    for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        ax1.arrow(0, 0, vec[0], vec[1], head_width=0.1, head_length=0.15,
                 fc='red', ec='red', linewidth=3,
                 label=f'Eigenvector {i+1}')

    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_title('Vectores Originales')
    ax1.grid(True)
    ax1.legend()

    # Vectores transformados
    ax2.set_aspect('equal')
    for v in transformed_vectors:
        ax2.arrow(0, 0, v[0], v[1], head_width=0.05, head_length=0.1,
                 fc='green', ec='green', alpha=0.6)

    # Eigenvectores transformados (escalados por eigenvalor)
    for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        transformed_eigenvec = val * vec
        ax2.arrow(0, 0, transformed_eigenvec[0], transformed_eigenvec[1],
                 head_width=0.1, head_length=0.15, fc='red', ec='red',
                 linewidth=3, label=f'λ{i+1}={val:.1f} × eigenvec{i+1}')

    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.set_title('Vectores Transformados por A')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print(f"Eigenvalores: {eigenvalues}")
    print(f"Eigenvectores:\n{eigenvectors}")

# visualize_eigenvectors()  # Descomentar para ejecutar
```

---

## Implementación práctica con NumPy

Ahora que entendemos los conceptos fundamentales, veamos cómo NumPy optimiza estas operaciones:

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import time

class OptimizedLinearAlgebra:
    """
    Comparación entre implementaciones caseras y NumPy optimizado.
    """

    @staticmethod
    def performance_comparison():
        """
        Compara el rendimiento entre implementaciones caseras y NumPy.
        """
        print("=== Comparación de Rendimiento ===")

        # Crear datos de prueba
        size = 1000

        # Datos para implementación casera
        vector1_data = [float(i) for i in range(size)]
        vector2_data = [float(i * 2) for i in range(size)]

        # Crear vectores caseros
        custom_v1 = Vector(vector1_data)
        custom_v2 = Vector(vector2_data)

        # Crear vectores NumPy
        numpy_v1 = np.array(vector1_data)
        numpy_v2 = np.array(vector2_data)

        # Probar producto punto - implementación casera
        start_time = time.time()
        custom_result = custom_v1.dot_product(custom_v2)
        custom_time = time.time() - start_time

        # Probar producto punto - NumPy
        start_time = time.time()
        numpy_result = np.dot(numpy_v1, numpy_v2)
        numpy_time = time.time() - start_time

        print(f"Tamaño del vector: {size}")
        print(f"Implementación casera: {custom_time:.6f} segundos")
        print(f"NumPy optimizado: {numpy_time:.6f} segundos")
        print(f"Speedup: {custom_time / numpy_time:.1f}x más rápido")
        print(f"Resultados iguales: {abs(custom_result - numpy_result) < 1e-10}")

        return custom_time, numpy_time

    @staticmethod
    def numpy_vector_operations():
        """
        Demuestra operaciones vectoriales optimizadas con NumPy.
        """
        print("\n=== Operaciones Vectoriales con NumPy ===")

        # Crear vectores NumPy
        user_preferences = np.array([4.5, 2.1, 5.0, 3.2])  # ratings
        movie_features = np.array([4.0, 3.5, 4.8, 2.9])   # características

        print(f"Preferencias usuario: {user_preferences}")
        print(f"Características película: {movie_features}")

        # Operaciones vectoriales
        similarity = np.dot(user_preferences, movie_features)
        print(f"Similitud (producto punto): {similarity:.2f}")

        # Normas (magnitudes)
        user_norm = np.linalg.norm(user_preferences)
        movie_norm = np.linalg.norm(movie_features)

        # Similitud coseno
        cosine_similarity = similarity / (user_norm * movie_norm)
        print(f"Similitud coseno: {cosine_similarity:.3f}")

        # Operaciones elemento a elemento
        preference_diff = user_preferences - movie_features
        print(f"Diferencia de preferencias: {preference_diff}")

        # Normalización
        normalized_prefs = user_preferences / user_norm
        print(f"Preferencias normalizadas: {normalized_prefs}")

        return similarity, cosine_similarity

    @staticmethod
    def numpy_matrix_operations():
        """
        Demuestra operaciones matriciales con NumPy.
        """
        print("\n=== Operaciones Matriciales con NumPy ===")

        # Dataset: 5 usuarios, 4 películas
        ratings_matrix = np.array([
            [5, 3, 0, 1],  # Usuario 1
            [4, 0, 0, 1],  # Usuario 2
            [1, 1, 0, 5],  # Usuario 3
            [1, 0, 0, 4],  # Usuario 4
            [0, 1, 5, 4],  # Usuario 5
        ])

        print("Matriz de ratings (usuarios × películas):")
        print(ratings_matrix)
        print(f"Forma: {ratings_matrix.shape}")

        # Transpuesta: películas × usuarios
        movies_users = ratings_matrix.T
        print(f"\nMatriz transpuesta (películas × usuarios):")
        print(movies_users)

        # Matriz de similitud entre usuarios (usuarios × usuarios)
        user_similarity = ratings_matrix @ ratings_matrix.T
        print(f"\nMatriz de similitud entre usuarios:")
        print(user_similarity)

        # Matriz de similitud entre películas (películas × películas)
        movie_similarity = movies_users @ movies_users.T
        print(f"\nMatriz de similitud entre películas:")
        print(movie_similarity)

        # Eigenvalores y eigenvectores de la matriz de similitud de usuarios
        eigenvals, eigenvecs = np.linalg.eig(user_similarity)
        print(f"\nEigenvalores de similitud de usuarios:")
        print(eigenvals)

        # El primer eigenvector (componente principal)
        principal_component = eigenvecs[:, 0]
        print(f"Componente principal (eigenvector dominante):")
        print(principal_component)

        return ratings_matrix, user_similarity, eigenvals, eigenvecs

def advanced_ml_applications():
    """
    Aplicaciones avanzadas de álgebra lineal en ML.
    """
    print("\n=== Aplicaciones Avanzadas en ML ===")

    # 1. Reducción de dimensionalidad con PCA
    print("1. Análisis de Componentes Principales (PCA)")

    # Generar datos de ejemplo: dataset 2D con correlación
    np.random.seed(42)
    n_samples = 100

    # Datos correlacionados
    x1 = np.random.randn(n_samples)
    x2 = x1 + 0.5 * np.random.randn(n_samples)  # x2 correlacionado con x1
    data = np.column_stack([x1, x2])

    print(f"Datos originales shape: {data.shape}")

    # Centrar los datos
    data_centered = data - np.mean(data, axis=0)

    # Calcular matriz de covarianza
    cov_matrix = np.cov(data_centered.T)
    print(f"Matriz de covarianza:")
    print(cov_matrix)

    # Eigendescomposición
    eigenvals, eigenvecs = np.linalg.eig(cov_matrix)

    # Ordenar por eigenvalores descendientes
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    print(f"Eigenvalores: {eigenvals}")
    print(f"Proporción de varianza explicada: {eigenvals / np.sum(eigenvals)}")

    # Proyectar datos en el primer componente principal
    first_pc = eigenvecs[:, 0]
    projected_data = data_centered @ first_pc

    print(f"Datos proyectados en 1D: shape {projected_data.shape}")

    # 2. Sistemas de recomendación con factorización matricial
    print(f"\n2. Factorización Matricial para Recomendaciones")

    # Matrix factorization simple usando SVD
    ratings = np.array([
        [5, 3, 0, 1, 4],
        [4, 0, 0, 1, 3],
        [1, 1, 0, 5, 2],
        [1, 0, 0, 4, 1],
        [0, 1, 5, 4, 0],
    ])

    # SVD: R = U × S × V^T
    U, s, Vt = np.linalg.svd(ratings, full_matrices=False)

    print(f"Forma original: {ratings.shape}")
    print(f"U shape: {U.shape}, S shape: {s.shape}, V^T shape: {Vt.shape}")

    # Reconstruir con menos componentes (reducción de dimensionalidad)
    k = 2  # Usar solo 2 componentes principales
    ratings_reconstructed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

    print(f"Ratings originales vs reconstruidos:")
    print(f"Original:\n{ratings}")
    print(f"Reconstruido (k={k}):\n{ratings_reconstructed}")

    # Calcular error de reconstrucción
    reconstruction_error = np.linalg.norm(ratings - ratings_reconstructed)
    print(f"Error de reconstrucción: {reconstruction_error:.3f}")

    return data, eigenvals, eigenvecs, ratings_reconstructed

def neural_network_linear_algebra():
    """
    Demuestra cómo el álgebra lineal potencia las redes neuronales.
    """
    print("\n=== Álgebra Lineal en Redes Neuronales ===")

    # Red neuronal simple: 3 entradas → 4 ocultas → 2 salidas
    np.random.seed(42)

    # Datos de entrada (batch de 5 ejemplos)
    X = np.random.randn(5, 3)  # 5 ejemplos, 3 características
    print(f"Entrada X shape: {X.shape}")
    print(f"Primeros 2 ejemplos:\n{X[:2]}")

    # Capa oculta: 3 → 4
    W1 = np.random.randn(3, 4) * 0.5  # Pesos
    b1 = np.zeros((1, 4))             # Sesgos

    # Forward pass capa 1
    Z1 = X @ W1 + b1  # Transformación lineal
    A1 = np.maximum(0, Z1)  # Activación ReLU

    print(f"\nCapa oculta:")
    print(f"Pesos W1 shape: {W1.shape}")
    print(f"Salida Z1 shape: {Z1.shape}")
    print(f"Salida activada A1 shape: {A1.shape}")

    # Capa de salida: 4 → 2
    W2 = np.random.randn(4, 2) * 0.5
    b2 = np.zeros((1, 2))

    # Forward pass capa 2
    Z2 = A1 @ W2 + b2
    A2 = 1 / (1 + np.exp(-Z2))  # Activación sigmoide

    print(f"\nCapa de salida:")
    print(f"Pesos W2 shape: {W2.shape}")
    print(f"Salida final A2 shape: {A2.shape}")
    print(f"Predicciones:\n{A2}")

    # Análisis de gradientes (simplificado)
    print(f"\n=== Análisis de Gradientes ===")

    # Gradiente ficticio desde la función de pérdida
    dA2 = np.random.randn(5, 2) * 0.1  # Gradiente de la pérdida respecto a A2

    # Backpropagation: calcular gradientes usando álgebra lineal
    # dW2 = A1^T @ dZ2
    dZ2 = dA2 * A2 * (1 - A2)  # Derivada de sigmoide
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    print(f"Gradiente dW2 shape: {dW2.shape}")
    print(f"Gradiente db2 shape: {db2.shape}")

    # Propagación hacia atrás a la capa anterior
    dA1 = dZ2 @ W2.T
    dZ1 = dA1.copy()
    dZ1[A1 <= 0] = 0  # Derivada de ReLU

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    print(f"Gradiente dW1 shape: {dW1.shape}")
    print(f"Gradiente db1 shape: {db1.shape}")

    return W1, W2, dW1, dW2

def visualize_transformations():
    """
    Visualiza transformaciones lineales para entender su efecto geométrico.
    """
    print("\n=== Visualización de Transformaciones Lineales ===")

    # Crear un conjunto de puntos para visualizar
    theta = np.linspace(0, 2*np.pi, 20)
    unit_circle = np.array([np.cos(theta), np.sin(theta)]).T

    # Definir varias transformaciones
    transformations = {
        'Identidad': np.eye(2),
        'Escalado': np.array([[2, 0], [0, 0.5]]),
        'Rotación 45°': np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                  [np.sin(np.pi/4), np.cos(np.pi/4)]]),
        'Cizalladura': np.array([[1, 0.5], [0, 1]]),
        'Reflexión': np.array([[1, 0], [0, -1]]),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, transform) in enumerate(transformations.items()):
        if idx >= len(axes):
            break

        # Aplicar transformación
        transformed = unit_circle @ transform.T

        # Plotear
        ax = axes[idx]
        ax.plot(unit_circle[:, 0], unit_circle[:, 1], 'b-', label='Original', alpha=0.7)
        ax.plot(transformed[:, 0], transformed[:, 1], 'r-', label='Transformado', linewidth=2)

        # Calcular y mostrar eigenvalores/eigenvectores
        try:
            eigenvals, eigenvecs = np.linalg.eig(transform)
            for i, (val, vec) in enumerate(zip(eigenvals, eigenvecs.T)):
                if np.isreal(val) and val > 0:
                    # Dibujar eigenvector
                    ax.arrow(0, 0, vec[0], vec[1], head_width=0.1,
                            head_length=0.1, fc='green', ec='green', linewidth=2)
                    ax.text(vec[0]*1.2, vec[1]*1.2, f'λ={val:.1f}',
                           fontsize=10, color='green')
        except:
            pass

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{name}')
        ax.legend()

    # Ocultar el último subplot si no se usa
    if len(transformations) < len(axes):
        axes[-1].set_visible(False)

    plt.tight_layout()
    plt.savefig('transformaciones_lineales.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Análisis de determinantes
    print(f"\nAnálisis de Determinantes (cambio de área):")
    for name, transform in transformations.items():
        det = np.linalg.det(transform)
        print(f"{name}: determinante = {det:.2f}")
        if abs(det) < 1e-10:
            print(f"  → Transformación singular (colapsa dimensiones)")
        elif abs(det) < 1:
            print(f"  → Contrae el área por factor {abs(det):.2f}")
        elif abs(det) > 1:
            print(f"  → Expande el área por factor {abs(det):.2f}")

        if det < 0:
            print(f"  → Invierte orientación")

# Ejecutar todas las demostraciones
def run_all_demonstrations():
    """
    Ejecuta todas las demostraciones de álgebra lineal.
    """
    print("=== ÁLGEBRA LINEAL PARA MACHINE LEARNING ===")
    print("Demostraciones prácticas de conceptos fundamentales\n")

    # Comparación de rendimiento
    optimizer = OptimizedLinearAlgebra()
    optimizer.performance_comparison()

    # Operaciones vectoriales y matriciales con NumPy
    optimizer.numpy_vector_operations()
    optimizer.numpy_matrix_operations()

    # Aplicaciones avanzadas
    advanced_ml_applications()

    # Redes neuronales
    neural_network_linear_algebra()

    # Visualizaciones (comentado para evitar problemas de display)
    # visualize_transformations()

    print("\n=== FIN DE DEMOSTRACIONES ===")
    print("¡Has completado el tour completo por el álgebra lineal para ML!")

if __name__ == "__main__":
    run_all_demonstrations()
```

---

## Proyecto Semanal: Sistema de Recomendación con Álgebra Lineal

Para consolidar los conceptos aprendidos, implementaremos un sistema de recomendación completo usando solo álgebra lineal.

### Objetivo del Proyecto

Crear un sistema que:
1. Represente usuarios y productos como vectores
2. Calcule similitudes usando productos punto
3. Haga recomendaciones basadas en usuarios similares
4. Visualice los resultados en espacios de menor dimensión

### Implementación Completa

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json

class RecommendationSystem:
    """
    Sistema de recomendación basado en álgebra lineal.

    Este proyecto demuestra cómo el álgebra lineal es fundamental
    en sistemas de machine learning reales.
    """

    def __init__(self):
        self.users: Dict[str, np.ndarray] = {}
        self.items: Dict[str, np.ndarray] = {}
        self.ratings_matrix: Optional[np.ndarray] = None
        self.user_similarity_matrix: Optional[np.ndarray] = None
        self.item_features: Optional[np.ndarray] = None
        self.user_names: List[str] = []
        self.item_names: List[str] = []

    def add_user_ratings(self, user_name: str, ratings: Dict[str, float]):
        """
        Agrega las calificaciones de un usuario.

        Args:
            user_name: Nombre del usuario
            ratings: Diccionario {item_name: rating}
        """
        # Convertir ratings a vector (usar 0 para items no calificados)
        if not self.item_names:
            self.item_names = list(ratings.keys())

        rating_vector = np.zeros(len(self.item_names))
        for i, item in enumerate(self.item_names):
            rating_vector[i] = ratings.get(item, 0.0)

        self.users[user_name] = rating_vector
        if user_name not in self.user_names:
            self.user_names.append(user_name)

    def build_ratings_matrix(self):
        """
        Construye la matriz de calificaciones usuarios × items.
        """
        n_users = len(self.user_names)
        n_items = len(self.item_names)

        self.ratings_matrix = np.zeros((n_users, n_items))

        for i, user in enumerate(self.user_names):
            if user in self.users:
                self.ratings_matrix[i] = self.users[user]

        print(f"Matriz de ratings construida: {self.ratings_matrix.shape}")
        return self.ratings_matrix

    def calculate_user_similarity(self, method='cosine'):
        """
        Calcula la matriz de similitud entre usuarios.

        Args:
            method: 'cosine', 'dot_product', o 'euclidean'
        """
        if self.ratings_matrix is None:
            self.build_ratings_matrix()

        n_users = self.ratings_matrix.shape[0]
        self.user_similarity_matrix = np.zeros((n_users, n_users))

        for i in range(n_users):
            for j in range(n_users):
                if method == 'cosine':
                    similarity = self._cosine_similarity(
                        self.ratings_matrix[i],
                        self.ratings_matrix[j]
                    )
                elif method == 'dot_product':
                    similarity = np.dot(
                        self.ratings_matrix[i],
                        self.ratings_matrix[j]
                    )
                elif method == 'euclidean':
                    similarity = -np.linalg.norm(
                        self.ratings_matrix[i] - self.ratings_matrix[j]
                    )

                self.user_similarity_matrix[i, j] = similarity

        return self.user_similarity_matrix

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcula similitud coseno entre dos vectores."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def recommend_items(self, target_user: str, n_recommendations: int = 3) -> List[Tuple[str, float]]:
        """
        Recomienda items basado en usuarios similares.

        Args:
            target_user: Usuario para quien hacer recomendaciones
            n_recommendations: Número de recomendaciones

        Returns:
            Lista de tuplas (item_name, predicted_rating)
        """
        if target_user not in self.user_names:
            raise ValueError(f"Usuario {target_user} no encontrado")

        if self.user_similarity_matrix is None:
            self.calculate_user_similarity()

        user_idx = self.user_names.index(target_user)
        user_ratings = self.ratings_matrix[user_idx]

        # Encontrar usuarios similares (excluyendo al usuario mismo)
        similarities = self.user_similarity_matrix[user_idx].copy()
        similarities[user_idx] = 0  # Excluir auto-similitud

        # Predecir ratings para items no calificados
        predictions = []

        for item_idx, current_rating in enumerate(user_ratings):
            if current_rating == 0:  # Item no calificado
                # Calcular predicción basada en usuarios similares
                numerator = 0
                denominator = 0

                for other_user_idx, similarity in enumerate(similarities):
                    if similarity > 0 and self.ratings_matrix[other_user_idx, item_idx] > 0:
                        numerator += similarity * self.ratings_matrix[other_user_idx, item_idx]
                        denominator += similarity

                if denominator > 0:
                    predicted_rating = numerator / denominator
                    predictions.append((self.item_names[item_idx], predicted_rating))

        # Ordenar por rating predicho descendente
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n_recommendations]

    def analyze_with_pca(self, n_components: int = 2):
        """
        Analiza los datos usando PCA para visualización.

        Args:
            n_components: Número de componentes principales

        Returns:
            Datos proyectados, componentes principales, varianza explicada
        """
        if self.ratings_matrix is None:
            self.build_ratings_matrix()

        # Centrar los datos
        data_centered = self.ratings_matrix - np.mean(self.ratings_matrix, axis=0)

        # Calcular matriz de covarianza
        cov_matrix = np.cov(data_centered.T)

        # Eigendescomposición
        eigenvals, eigenvecs = np.linalg.eig(cov_matrix)

        # Ordenar por eigenvalores descendentes
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        # Seleccionar componentes principales
        principal_components = eigenvecs[:, :n_components]

        # Proyectar datos
        projected_data = data_centered @ principal_components

        # Calcular varianza explicada
        explained_variance = eigenvals[:n_components] / np.sum(eigenvals)

        return projected_data, principal_components, explained_variance

    def visualize_users_2d(self):
        """
        Visualiza usuarios en espacio 2D usando PCA.
        """
        projected_data, components, explained_var = self.analyze_with_pca(2)

        plt.figure(figsize=(10, 8))

        # Scatter plot de usuarios
        plt.scatter(projected_data[:, 0], projected_data[:, 1],
                   s=100, alpha=0.7, c='blue')

        # Etiquetar usuarios
        for i, user in enumerate(self.user_names):
            plt.annotate(user, (projected_data[i, 0], projected_data[i, 1]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)

        plt.xlabel(f'Componente Principal 1 ({explained_var[0]:.1%} varianza)')
        plt.ylabel(f'Componente Principal 2 ({explained_var[1]:.1%} varianza)')
        plt.title('Usuarios en Espacio 2D (PCA)')
        plt.grid(True, alpha=0.3)

        # Mostrar vectores de componentes principales
        plt.arrow(0, 0, components[0, 0]*2, components[0, 1]*2,
                 head_width=0.1, head_length=0.1, fc='red', ec='red',
                 label='PC1')
        plt.arrow(0, 0, components[1, 0]*2, components[1, 1]*2,
                 head_width=0.1, head_length=0.1, fc='green', ec='green',
                 label='PC2')

        plt.legend()
        plt.tight_layout()
        plt.show()

        return projected_data, components, explained_var

    def print_analysis_report(self):
        """
        Imprime un reporte completo del análisis.
        """
        print("=== REPORTE DEL SISTEMA DE RECOMENDACIÓN ===")

        if self.ratings_matrix is not None:
            print(f"\n1. DATOS:")
            print(f"   - Usuarios: {len(self.user_names)}")
            print(f"   - Items: {len(self.item_names)}")
            print(f"   - Matriz de ratings: {self.ratings_matrix.shape}")

            # Estadísticas de la matriz
            non_zero_ratings = np.count_nonzero(self.ratings_matrix)
            total_possible = self.ratings_matrix.size
            sparsity = 1 - (non_zero_ratings / total_possible)

            print(f"   - Ratings dados: {non_zero_ratings}/{total_possible}")
            print(f"   - Sparsity: {sparsity:.1%}")

            print(f"\n2. MATRIZ DE RATINGS:")
            print("   Usuarios × Items:")
            print("   " + "\t".join(f"{item[:8]}" for item in self.item_names))
            for i, user in enumerate(self.user_names):
                ratings_str = "\t".join(f"{r:.1f}" for r in self.ratings_matrix[i])
                print(f"   {user[:8]}\t{ratings_str}")

        if self.user_similarity_matrix is not None:
            print(f"\n3. SIMILITUD ENTRE USUARIOS:")
            print("   " + "\t".join(f"{user[:8]}" for user in self.user_names))
            for i, user in enumerate(self.user_names):
                sim_str = "\t".join(f"{s:.3f}" for s in self.user_similarity_matrix[i])
                print(f"   {user[:8]}\t{sim_str}")

        # PCA Analysis
        projected, components, explained_var = self.analyze_with_pca(2)
        print(f"\n4. ANÁLISIS DE COMPONENTES PRINCIPALES:")
        print(f"   - Componente 1 explica: {explained_var[0]:.1%} de la varianza")
        print(f"   - Componente 2 explica: {explained_var[1]:.1%} de la varianza")
        print(f"   - Total explicado: {sum(explained_var):.1%}")

        print(f"\n   Usuarios en espacio 2D:")
        for i, user in enumerate(self.user_names):
            print(f"   {user}: ({projected[i, 0]:.2f}, {projected[i, 1]:.2f})")

# Crear y probar el sistema de recomendación
def test_recommendation_system():
    """
    Prueba completa del sistema de recomendación.
    """
    print("=== PRUEBA DEL SISTEMA DE RECOMENDACIÓN ===")

    # Crear sistema
    rec_sys = RecommendationSystem()

    # Datos de ejemplo: usuarios y sus calificaciones de películas
    user_ratings = {
        'Ana': {
            'Acción_1': 5.0, 'Comedia_1': 2.0, 'Drama_1': 4.0,
            'Acción_2': 4.0, 'Comedia_2': 1.0, 'Drama_2': 5.0,
            'Acción_3': 0.0, 'Comedia_3': 0.0, 'Drama_3': 0.0
        },
        'Bob': {
            'Acción_1': 4.0, 'Comedia_1': 3.0, 'Drama_1': 3.0,
            'Acción_2': 5.0, 'Comedia_2': 2.0, 'Drama_2': 4.0,
            'Acción_3': 4.0, 'Comedia_3': 0.0, 'Drama_3': 0.0
        },
        'Carlos': {
            'Acción_1': 1.0, 'Comedia_1': 5.0, 'Drama_1': 2.0,
            'Acción_2': 2.0, 'Comedia_2': 4.0, 'Drama_2': 1.0,
            'Acción_3': 0.0, 'Comedia_3': 5.0, 'Drama_3': 0.0
        },
        'Diana': {
            'Acción_1': 2.0, 'Comedia_1': 1.0, 'Drama_1': 5.0,
            'Acción_2': 1.0, 'Comedia_2': 0.0, 'Drama_2': 4.0,
            'Acción_3': 0.0, 'Comedia_3': 0.0, 'Drama_3': 5.0
        },
        'Elena': {
            'Acción_1': 4.0, 'Comedia_1': 4.0, 'Drama_1': 3.0,
            'Acción_2': 3.0, 'Comedia_2': 4.0, 'Drama_2': 3.0,
            'Acción_3': 0.0, 'Comedia_3': 0.0, 'Drama_3': 0.0
        }
    }

    # Agregar usuarios al sistema
    for user, ratings in user_ratings.items():
        rec_sys.add_user_ratings(user, ratings)

    # Construir matriz y calcular similitudes
    rec_sys.build_ratings_matrix()
    rec_sys.calculate_user_similarity('cosine')

    # Generar reporte
    rec_sys.print_analysis_report()

    # Hacer recomendaciones para cada usuario
    print(f"\n5. RECOMENDACIONES:")
    for user in rec_sys.user_names:
        try:
            recommendations = rec_sys.recommend_items(user, n_recommendations=2)
            print(f"\n   Recomendaciones para {user}:")
            if recommendations:
                for item, rating in recommendations:
                    print(f"   - {item}: {rating:.2f} (predicho)")
            else:
                print(f"   - No hay recomendaciones disponibles")
        except Exception as e:
            print(f"   - Error generando recomendaciones: {e}")

    # Visualización (comentado para evitar issues con display)
    # rec_sys.visualize_users_2d()

    return rec_sys


---

## Ejercicios Prácticos

Para consolidar tu comprensión, completa estos ejercicios progresivos:

### Ejercicio 1: Implementación Básica (⭐)

Implementa las siguientes funciones sin usar NumPy:

```python
def vector_operations_exercise():
    """
    Ejercicio 1: Operaciones vectoriales básicas.
    """
    print("=== EJERCICIO 1: OPERACIONES VECTORIALES ===")

    # TODO: Implementar estas funciones
    def dot_product_manual(vec1: List[float], vec2: List[float]) -> float:
        """Calcula el producto punto manualmente."""
        # Tu código aquí
        pass

    def vector_magnitude_manual(vec: List[float]) -> float:
        """Calcula la magnitud de un vector manualmente."""
        # Tu código aquí
        pass

    def cosine_similarity_manual(vec1: List[float], vec2: List[float]) -> float:
        """Calcula la similitud coseno manualmente."""
        # Tu código aquí
        pass

    # Datos de prueba
    user1_prefs = [4.0, 2.0, 5.0, 1.0]  # [acción, comedia, drama, terror]
    user2_prefs = [3.0, 4.0, 2.0, 2.0]
    user3_prefs = [4.5, 1.5, 4.8, 0.5]

    # Probar tus implementaciones
    print(f"Usuario 1: {user1_prefs}")
    print(f"Usuario 2: {user2_prefs}")
    print(f"Usuario 3: {user3_prefs}")

    # Calcular similitudes
    sim_1_2 = cosine_similarity_manual(user1_prefs, user2_prefs)
    sim_1_3 = cosine_similarity_manual(user1_prefs, user3_prefs)
    sim_2_3 = cosine_similarity_manual(user2_prefs, user3_prefs)

    print(f"\nSimilitudes:")
    print(f"Usuario 1 - Usuario 2: {sim_1_2:.3f}")
    print(f"Usuario 1 - Usuario 3: {sim_1_3:.3f}")
    print(f"Usuario 2 - Usuario 3: {sim_2_3:.3f}")

    # ¿Qué usuarios son más similares y por qué?
    print(f"\nAnálisis:")
    print(f"Usuarios más similares: {'1 y 3' if sim_1_3 > max(sim_1_2, sim_2_3) else '1 y 2' if sim_1_2 > sim_2_3 else '2 y 3'}")
```

### Ejercicio 2: Manipulación de Matrices (⭐⭐)

```python
def matrix_operations_exercise():
    """
    Ejercicio 2: Operaciones matriciales.
    """
    print("=== EJERCICIO 2: OPERACIONES MATRICIALES ===")

    # Dataset de ratings (usuarios × películas)
    ratings = [
        [5, 3, 0, 1, 4, 0],  # Usuario A
        [4, 0, 0, 1, 3, 5],  # Usuario B
        [1, 1, 0, 5, 2, 4],  # Usuario C
        [1, 0, 0, 4, 1, 5],  # Usuario D
        [0, 1, 5, 4, 0, 3],  # Usuario E
    ]

    # TODO: Implementar estas funciones usando tu clase Matrix

    def calculate_user_similarity_matrix(ratings_matrix):
        """
        Calcula la matriz de similitud usuario-usuario.
        Resultado: matriz 5×5 donde element (i,j) = similitud entre usuario i y j
        """
        # Tu código aquí
        pass

    def calculate_movie_similarity_matrix(ratings_matrix):
        """
        Calcula la matriz de similitud película-película.
        Pista: usa la transpuesta de la matriz de ratings
        """
        # Tu código aquí
        pass

    def predict_missing_ratings(ratings_matrix, user_similarities):
        """
        Predice los ratings faltantes usando collaborative filtering.
        """
        # Tu código aquí
        pass

    # Probar tus implementaciones
    ratings_matrix = Matrix(ratings)
    print("Matriz de ratings original:")
    print(ratings_matrix)

    # Calcular similitudes
    user_similarities = calculate_user_similarity_matrix(ratings_matrix)
    movie_similarities = calculate_movie_similarity_matrix(ratings_matrix)

    print("Matriz de similitud entre usuarios:")
    print(user_similarities)

    # Predecir ratings faltantes
    predicted_ratings = predict_missing_ratings(ratings_matrix, user_similarities)
    print("Matriz con ratings predichos:")
    print(predicted_ratings)
```

### Ejercicio 3: Análisis de Componentes Principales (⭐⭐⭐)

```python
def pca_exercise():
    """
    Ejercicio 3: Implementar PCA desde cero.
    """
    print("=== EJERCICIO 3: PCA DESDE CERO ===")

    # Datos de ejemplo: preferencias de usuarios en 4 géneros
    user_data = np.array([
        [5, 1, 4, 2],  # Usuario que prefiere acción y drama
        [4, 2, 5, 1],  # Similar al anterior
        [1, 5, 2, 4],  # Usuario que prefiere comedia y terror
        [2, 4, 1, 5],  # Similar al anterior
        [3, 3, 3, 3],  # Usuario promedio
        [4, 1, 3, 2],  # Variante del primer tipo
        [2, 4, 2, 4],  # Variante del segundo tipo
    ])

    def implement_pca_steps(data, n_components=2):
        """
        Implementa PCA paso a paso para entender el proceso.

        Pasos:
        1. Centrar los datos (restar la media)
        2. Calcular matriz de covarianza
        3. Encontrar eigenvalores y eigenvectores
        4. Ordenar por eigenvalores descendentes
        5. Seleccionar n_components principales
        6. Proyectar datos originales

        Returns:
            projected_data, principal_components, explained_variance_ratio
        """

        print(f"Datos originales shape: {data.shape}")
        print(f"Primeros 3 usuarios:")
        print(data[:3])

        # Paso 1: Centrar los datos
        # TODO: Implementar
        data_centered = None  # Tu código aquí
        print(f"\nDatos centrados (primeros 3):")
        print(data_centered[:3])

        # Paso 2: Matriz de covarianza
        # TODO: Implementar
        cov_matrix = None  # Tu código aquí
        print(f"\nMatriz de covarianza:")
        print(cov_matrix)

        # Paso 3: Eigenvalores y eigenvectores
        # TODO: Implementar usando np.linalg.eig
        eigenvals, eigenvecs = None, None  # Tu código aquí

        # Paso 4: Ordenar por eigenvalores
        # TODO: Implementar
        idx = None  # Tu código aquí
        eigenvals_sorted = None  # Tu código aquí
        eigenvecs_sorted = None  # Tu código aquí

        print(f"\nEigenvalores ordenados: {eigenvals_sorted}")
        print(f"Eigenvectores (columnas):")
        print(eigenvecs_sorted)

        # Paso 5: Seleccionar componentes principales
        principal_components = eigenvecs_sorted[:, :n_components]
        print(f"\nComponentes principales (primeros {n_components}):")
        print(principal_components)

        # Paso 6: Proyectar datos
        projected_data = data_centered @ principal_components
        print(f"\nDatos proyectados shape: {projected_data.shape}")
        print(f"Primeros 3 usuarios en 2D:")
        print(projected_data[:3])

        # Varianza explicada
        explained_variance_ratio = eigenvals_sorted[:n_components] / np.sum(eigenvals_sorted)
        print(f"\nVarianza explicada por cada componente:")
        for i, ratio in enumerate(explained_variance_ratio):
            print(f"  PC{i+1}: {ratio:.1%}")
        print(f"  Total: {np.sum(explained_variance_ratio):.1%}")

        return projected_data, principal_components, explained_variance_ratio

    # Ejecutar PCA
    projected, components, variance_ratios = implement_pca_steps(user_data, 2)

    # Análisis adicional
    print(f"\n=== ANÁLISIS DE RESULTADOS ===")
    print(f"Los componentes principales representan:")
    print(f"PC1 (explica {variance_ratios[0]:.1%}): direcciones de máxima varianza")
    print(f"PC2 (explica {variance_ratios[1]:.1%}): segunda dirección de máxima varianza")

    # TODO: Interpretar los componentes principales
    # ¿Qué significan en términos de preferencias de usuarios?

    return projected, components
```

### Ejercicio 4: Red Neuronal Simple (⭐⭐⭐⭐)

```python
def neural_network_exercise():
    """
    Ejercicio 4: Implementar una red neuronal usando solo álgebra lineal.
    """
    print("=== EJERCICIO 4: RED NEURONAL CON ÁLGEBRA LINEAL ===")

    class SimpleNeuralNetwork:
        """
        Red neuronal simple: entrada → capa oculta → salida
        Todo implementado con operaciones matriciales.
        """

        def __init__(self, input_size: int, hidden_size: int, output_size: int):
            # TODO: Inicializar pesos aleatoriamente
            np.random.seed(42)  # Para reproducibilidad

            # Pesos capa 1: entrada → oculta
            self.W1 = None  # Tu código aquí: shape (input_size, hidden_size)
            self.b1 = None  # Tu código aquí: shape (1, hidden_size)

            # Pesos capa 2: oculta → salida
            self.W2 = None  # Tu código aquí: shape (hidden_size, output_size)
            self.b2 = None  # Tu código aquí: shape (1, output_size)

        def forward(self, X):
            """
            Forward pass: calcula la salida de la red.

            Args:
                X: datos de entrada, shape (n_samples, input_size)

            Returns:
                output: salida de la red, shape (n_samples, output_size)
            """
            # TODO: Implementar forward pass

            # Capa 1: transformación lineal + activación ReLU
            self.z1 = None  # X @ W1 + b1
            self.a1 = None  # ReLU(z1) = max(0, z1)

            # Capa 2: transformación lineal + activación sigmoide
            self.z2 = None  # a1 @ W2 + b2
            self.a2 = None  # sigmoide(z2) = 1/(1 + exp(-z2))

            return self.a2

        def backward(self, X, y, output):
            """
            Backward pass: calcula gradientes usando regla de la cadena.

            Args:
                X: datos de entrada
                y: etiquetas verdaderas
                output: salida de forward pass
            """
            m = X.shape[0]  # número de ejemplos

            # TODO: Implementar backward pass

            # Gradientes capa de salida
            dz2 = output - y  # Derivada de pérdida cuadrática + sigmoide
            dW2 = None  # Tu código aquí
            db2 = None  # Tu código aquí

            # Gradientes capa oculta
            da1 = None  # Tu código aquí
            dz1 = None  # Tu código aquí (considerar derivada de ReLU)
            dW1 = None  # Tu código aquí
            db1 = None  # Tu código aquí

            return dW1, db1, dW2, db2

        def train_step(self, X, y, learning_rate=0.01):
            """Un paso de entrenamiento: forward + backward + actualización."""
            # Forward pass
            output = self.forward(X)

            # Calcular pérdida
            loss = np.mean((output - y) ** 2)

            # Backward pass
            dW1, db1, dW2, db2 = self.backward(X, y, output)

            # Actualizar pesos (gradient descent)
            # TODO: Implementar actualización de pesos
            self.W1 -= None  # Tu código aquí
            self.b1 -= None  # Tu código aquí
            self.W2 -= None  # Tu código aquí
            self.b2 -= None  # Tu código aquí

            return loss

    # Generar datos de ejemplo: clasificación binaria simple
    np.random.seed(42)
    X = np.random.randn(100, 3)  # 100 ejemplos, 3 características
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)  # Etiqueta binaria

    print(f"Datos de entrenamiento:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Primeros 5 ejemplos:")
    print(f"X[:5]: {X[:5]}")
    print(f"y[:5]: {y[:5].flatten()}")

    # TODO: Crear y entrenar la red
    net = SimpleNeuralNetwork(input_size=3, hidden_size=4, output_size=1)

    # Entrenar por varias épocas
    epochs = 1000
    losses = []

    for epoch in range(epochs):
        loss = net.train_step(X, y, learning_rate=0.1)
        losses.append(loss)

        if epoch % 100 == 0:
            print(f"Época {epoch}, Pérdida: {loss:.4f}")

    # Evaluación final
    final_output = net.forward(X)
    final_predictions = (final_output > 0.5).astype(int)
    accuracy = np.mean(final_predictions == y)

    print(f"\nResultados finales:")
    print(f"Pérdida final: {losses[-1]:.4f}")
    print(f"Precisión: {accuracy:.2%}")

    # TODO: Analizar qué aprendió la red
    print(f"\nPesos aprendidos:")
    print(f"W1 (entrada → oculta):")
    print(net.W1)
    print(f"W2 (oculta → salida):")
    print(net.W2)

    return net, losses

# neural_network_exercise()  # Descomentar para ejecutar
```

---

## Conexión con la Próxima Semana: Estadística y Probabilidad

El álgebra lineal que acabas de dominar es la base computacional del machine learning. La próxima semana exploraremos **estadística y probabilidad**, que proporcionan el marco teórico para entender:

### Lo que Veremos en la Semana 3

1. **Distribuciones de Probabilidad**: Cómo modelar la incertidumbre en los datos
2. **Estadística Bayesiana**: El fundamento teórico de muchos algoritmos de ML
3. **Inferencia Estadística**: Cómo hacer predicciones con confianza
4. **Correlación vs Causalidad**: Evitar trampas comunes en análisis de datos

### Conexiones con Álgebra Lineal

- **Vectores aleatorios**: Las distribuciones multivariadas usan vectores
- **Matrices de covarianza**: Describen relaciones estadísticas entre variables
- **Transformaciones**: Cambios de variables usando matrices
- **PCA**: Tiene interpretación estadística profunda

### Proyecto de Transición

Para prepararte para la próxima semana, piensa en estas preguntas sobre tu sistema de recomendación:

1. **¿Qué tan confiables son nuestras predicciones?**
2. **¿Cómo manejar la incertidumbre en los ratings?**
3. **¿Qué probabilidad hay de que a un usuario le guste una película?**
4. **¿Cómo incorporar la confianza en nuestras recomendaciones?**

Estas preguntas nos llevan naturalmente al mundo de la estadística y probabilidad.

---

## Resumen y Puntos Clave

### Lo que Aprendiste Esta Semana

✅ **Conceptos Fundamentales**
- Vectores como representación de datos
- Matrices como transformaciones
- Espacios vectoriales y transformaciones lineales
- Eigenvalores y eigenvectores

✅ **Operaciones Clave**
- Producto punto para similitud
- Multiplicación matriz-vector para transformaciones
- Descomposición de matrices para análisis

✅ **Aplicaciones Prácticas**
- Sistema de recomendación completo
- Reducción de dimensionalidad con PCA
- Fundamentos de redes neuronales

✅ **Herramientas**
- Implementación desde cero para comprensión
- NumPy para optimización
- Visualización de conceptos

### Puntos Clave para Recordar

1. **El álgebra lineal es el lenguaje del ML**: Todo se reduce a vectores y matrices
2. **Las operaciones tienen significado**: No son solo cálculos, representan conceptos
3. **La geometría importa**: Visualizar ayuda a entender
4. **La optimización es crucial**: NumPy vs implementación casera
5. **La teoría guía la práctica**: Entender el "por qué" antes del "cómo"

### Para Profundizar

Si quieres explorar más, considera estos recursos:

- **Libros**: "Linear Algebra and Its Applications" de Gilbert Strang
- **Videos**: Curso de álgebra lineal de 3Blue1Brown
- **Práctica**: Implementar más algoritmos desde cero
- **Aplicaciones**: Explorar computer vision y NLP

---

La próxima semana nos sumergiremos en **estadística y probabilidad**, donde aprenderemos a cuantificar y manejar la incertidumbre - un componente esencial en cualquier sistema de IA robusto.

¿Estás listo para dar el siguiente paso en tu transformación de software engineer a AI engineer?

{{< alert >}}
💡 **Tip del AI Engineer**: El álgebra lineal no es solo matemática abstracta - es la herramienta que permite a las máquinas "pensar" con datos. Cada operación vectorial que domines te acerca más a entender cómo funciona la inteligencia artificial.
{{< /alert >}}

---

*Este artículo es parte de la serie "De Software Engineer a AI Engineer" - 16 semanas de transformación práctica. ¿Perdiste la semana anterior? [Revisa los Fundamentos de IA](../semana-1-fundamentos-ia). ¿Listo para continuar? [Explora Estadística y Probabilidad](../semana-3-estadistica-probabilidad).*
