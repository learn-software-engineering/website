---
weight: 1
series: ["Matemática para Machine Learning"]
series_order: 1
title: "Álgebra Lineal para Machine Learning: vectores y matrices que todo ingeniero en IA debe conocer"
description: "Domina los fundamentos matemáticos del machine learning: vectores, matrices y transformaciones lineales explicados desde la perspectiva del programador."
authors:
  - jnonino
date: 2025-09-12
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

O considera una red neuronal procesando una imagen de \(224x224\) píxeles. Esa imagen se convierte en una matriz de \(50176\) elementos. Las operaciones de la red (convoluciones, transformaciones) son multiplicaciones de matrices. El entrenamiento optimiza estas matrices, otra vez: **álgebra lineal**.

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

{{< codeimporter url="https://raw.githubusercontent.com/learn-software-engineering/examples/refs/heads/main/ai/module2/algebra/interpretacion_geometrica.py" type="python" >}}

Al ejecutar el código anterior obtenemos:

{{< figure
  src="img/interpretacion_geometrica.png"
  alt="Interpretación geométrica de un vector en Machine Learning"
  caption="Interpretación geométrica de un vector en Machine Learning"
  >}}

### Suma de vectores

La suma vectorial es **componente a componente**:

$$\mathbf{u} + \mathbf{v} = \begin{pmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{pmatrix}$$

*Interpretación en ML*: Si tenemos las preferencias de dos usuarios similares, podemos promediar sus vectores para encontrar preferencias "típicas" de ese segmento.

### Multiplicación por escalar

$$c \cdot \mathbf{v} = \begin{pmatrix} c \cdot v_1 \\ c \cdot v_2 \\ \vdots \\ c \cdot v_n \end{pmatrix}$$

*Interpretación en ML*: Amplificar o reducir la importancia de ciertas características.

### Producto Punto

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

{{< codeimporter url="https://raw.githubusercontent.com/learn-software-engineering/examples/refs/heads/main/ai/module2/algebra/vector.py" type="python" >}}

Al ejecutar el código anterior obtenemos:
```bash
>  python vector.py
###################################
Ejemplos de Operaciones Vectoriales
###################################

=== Ejemplo 1: Preferencias de usuarios ===
   Cada usuario se corresponde con un vector que mapea sus preferencias en películas
   Vector([acción, comedia, drama])

   Usuario 0: Vector([4, 2, 5])
      Cálculos de similitud con el usuario 1
         Suma: preferencias combinadas: Vector([7, 6, 7])
         Similitud (producto punto): 30
         Similitud coseno: 0.830
      Cálculos de similitud con el usuario 2
         Suma: preferencias combinadas: Vector([13, 3, 7])
         Similitud (producto punto): 48
         Similitud coseno: 0.772
      Cálculos de similitud con el usuario 3
         Suma: preferencias combinadas: Vector([7, 10, 6])
         Similitud (producto punto): 33
         Similitud coseno: 0.572
      Cálculos de similitud con el usuario 4
         Suma: preferencias combinadas: Vector([5, 4, 14])
         Similitud (producto punto): 53
         Similitud coseno: 0.852
   Usuario 1: Vector([3, 4, 2])
      Cálculos de similitud con el usuario 2
         Suma: preferencias combinadas: Vector([12, 5, 4])
         Similitud (producto punto): 35
         Similitud coseno: 0.701
      Cálculos de similitud con el usuario 3
         Suma: preferencias combinadas: Vector([6, 12, 3])
         Similitud (producto punto): 43
         Similitud coseno: 0.928
      Cálculos de similitud con el usuario 4
         Suma: preferencias combinadas: Vector([4, 6, 11])
         Similitud (producto punto): 29
         Similitud coseno: 0.581
   Usuario 2: Vector([9, 1, 2])
      Cálculos de similitud con el usuario 3
         Suma: preferencias combinadas: Vector([12, 9, 3])
         Similitud (producto punto): 37
         Similitud coseno: 0.464
      Cálculos de similitud con el usuario 4
         Suma: preferencias combinadas: Vector([10, 3, 11])
         Similitud (producto punto): 29
         Similitud coseno: 0.337
   Usuario 3: Vector([3, 8, 1])
      Cálculos de similitud con el usuario 4
         Suma: preferencias combinadas: Vector([4, 10, 10])
         Similitud (producto punto): 28
         Similitud coseno: 0.351
   Usuario 4: Vector([1, 2, 9])

=== Ejemplo 2: Análisis de Documentos ===
   Cada documento se corresponde con un vector que mapea las frecuencias de las palabras que contiene
   Vector([frecuencia_palabra_1, frecuencia_palabra_2, frecuencia_palabra_3, frecuencia_palabra_4])
   Documento 1: Vector([2, 1, 0, 3])
   Documento 2: Vector([1, 2, 1, 2])
      Similitud (producto punto): 10
      Similitud entre documentos (coseno): 0.845
```

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

### Multiplicación Matriz-Vector

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

{{< alert icon="lightbulb" >}}
Si aún no lo notaste, se puede establecer una conexión entre la multiplicación de una matriz por un vector y el producto punto entre vectores.

La conexión es directa: **multiplicar una matriz por un vector es, en el fondo, hacer varios productos punto seguidos**.

Si \(A\) es una matriz de \(m \times n\) y \(v\) es un vector de dimensión \(n\), el resultado de \(A \ v\) es un vector de dimensión \(m\) donde cada componente se obtiene haciendo el producto punto de una fila de la matriz con el vector.

$$(A \ v)_j = fila_j(A) \cdot v$$
{{< /alert >}}

### Multiplicación Matriz-Matriz

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

{{< alert icon="lightbulb" >}}
Nuevamente, la multiplicación de matrices está muy relacionada con el producto punto de vectores.

En la multiplicación entre las matrices \(A(m \times n)\) por \(B(n \times p)\), el elemento \(c_{ij}\) de la matriz resultado \(C\) se obtiene como:

$$\mathbf{C}_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}$$

Esto es exactamente el **producto punto** entre la fila \(i\) de \(A\) y la columna \(j\) de \(B\).

$$\mathbf{C}_{ij} = a_{i} \cdot b_{j}$$
{{< /alert >}}

### Implementación desde cero: clase Matriz

{{< codeimporter url="https://raw.githubusercontent.com/learn-software-engineering/examples/refs/heads/main/ai/module2/algebra/matriz.py" type="python" >}}

Al ejecutar el código anterior obtenemos:
```bash
>  python matriz.py
###################################
Ejemplos de Operaciones Matriciales
###################################

=== Ejemplo 1: Transponer ===
Matriz(
  [   1.000    2.000    3.000]
  [   4.000    5.000    6.000]
  [   7.000    8.000    9.000]
)
Transponer...
Matriz(
  [   1.000    4.000    7.000]
  [   2.000    5.000    8.000]
  [   3.000    6.000    9.000]
)

=== Ejemplo 2: Multiplicación matriz por escalar ===
Matriz(
  [   1.000    2.000    3.000]
  [   4.000    5.000    6.000]
  [   7.000    8.000    9.000]
)
Multiplicación por el escalar: 3...
Matriz(
  [   3.000    6.000    9.000]
  [  12.000   15.000   18.000]
  [  21.000   24.000   27.000]
)

=== Ejemplo 3: Multiplicación matriz por vector ===
Matriz(
  [   1.000    2.000    3.000]
  [   4.000    5.000    6.000]
  [   7.000    8.000    9.000]
)
Multiplicación por el vector: Vector([1.0, 2.0, 3.0])...
Vector([14.0, 32.0, 50.0])

=== Ejemplo 4: Multiplicación matriz por matriz ===
Matriz(
  [   1.000    2.000    3.000]
  [   4.000    5.000    6.000]
  [   7.000    8.000    9.000]
)
Multiplicación por la matriz: Matriz(
  [   9.000    8.000    7.000]
  [   6.000    5.000    4.000]
  [   3.000    2.000    1.000]
)...
   Forma matriz A: (3, 3)
   Forma matriz B: (3, 3)
Matriz(
  [  30.000   24.000   18.000]
  [  84.000   69.000   54.000]
  [ 138.000  114.000   90.000]
)

=== Ejemplo 5: Rotación de un vector en 2D ===

Vector original en 2D: Vector([1.0, 0.0])
Matriz de rotacion en 45 grados: Matriz(
  [   0.707   -0.707]
  [   0.707    0.707]
)
Vector rotado en 45 grados: Vector([0.7071067811865476, 0.7071067811865475])

Vector original en 2D: Vector([1.0, 0.0])
Matriz de rotacion en 90 grados: Matriz(
  [   0.000   -1.000]
  [   1.000    0.000]
)
Vector rotado en 90 grados: Vector([6.123233995736766e-17, 1.0])

Vector original en 2D: Vector([1.0, 0.0])
Matriz de rotacion en 180 grados: Matriz(
  [  -1.000   -0.000]
  [   0.000   -1.000]
)
Vector rotado en 180 grados: Vector([-1.0, 1.2246467991473532e-16])
```

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

Toda transformación lineal entre espacios vectoriales de dimensión finita puede representarse mediante una matriz, por ejemplo:

Imaginemos una transformación \(T: \mathbb{R}^2 \rightarrow \mathbb{R}^2\) definida por:

$$T(x,y) = (2x+y,3x-4y)$$

En base canónica:

- \(T(1,0) = (2,3) \rightarrow \) primera columna \(\begin{pmatrix} 2 \\ 3 \end{pmatrix}\)
- \(T(0,1) = (1,-4) \rightarrow \) segunda columna \(\begin{pmatrix} 1 \\ -4 \end{pmatrix}\)

La matriz asociada a la transformación \(T\) es:

$$|\mathbf{T}| = \begin{pmatrix}
2 & 1 \\
3 & -4
\end{pmatrix}$$

> La base canónica es un conjunto de vectores que forma una base ortonormal en un espacio vectorial. En el plano, la base canónica está compuesta por los vectores \(i\) y \(j\), que representan las direcciones de los ejes \(x\) e \(y\), respectivamente. Estos vectores se utilizan para expresar otros vectores como combinaciones lineales de la base canónica. Además, la base canónica es fundamental para entender la dimensión y la estructura de los espacios vectoriales.

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

#### Visualizando valores y vectores propios

Crea un archivo Python con el siguiente contenido:

{{< codeimporter url="https://raw.githubusercontent.com/learn-software-engineering/examples/refs/heads/main/ai/module2/algebra/valores_vectores_propios.py" type="python" >}}

Lo ejecutamos...

```bash
virtualenv venv
source venv/bin/activate
pip install numpy
pip install matplotlib
python valores_vectores_propios.py
```

y obtenemos:

{{< figure
    src="img/vectores_valores_propios.png"
    alt="Visualizando vectores propios con Python"
    caption="Visualizando vectores propios con Python"
    >}}

---

## Implementación práctica: un sistema de recomendaciones usando álgebra lineal

En el siguiente artículo de este módulo, vamos a actualizar nuestro sistema de recomendaciones para utilizar los conceptos de álgebra lineal que aprendimos hasta acá.

¡Nos vemos allí! 🚀

---

{{< alert icon="comment" >}}
¡Gracias por haber llegado hasta acá!

Si te gustó el artículo, por favor ¡no olvides compartirlo con tu familia, amigos y colegas!

Y si puedes, envía tus comentarios, sugerencias, críticas a nuestro mail o por redes sociales, nos ayudarías a generar mejor contenido y sobretodo más relevante para vos.

[{{< icon "email" >}}](mailto:learn.software.eng@gmail.com)
[{{< icon "github" >}}](https://github.com/learn-software-engineering)
[{{< icon "patreon" >}}](https://patreon.com/learnsoftwareeng)
[{{< icon "linkedin" >}}](https://linkedin.com/company/learn-software)
[{{< icon "instagram" >}}](https://www.instagram.com/learnsoftwareeng)
[{{< icon "facebook" >}}](https://www.facebook.com/learn.software.eng)
[{{< icon "x-twitter" >}}](https://x.com/software45687)
{{< /alert >}}
