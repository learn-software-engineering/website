---
weight: 1
title: "Vectores, Escalares y Espacios Vectoriales"
authors:
  - jnonino
description: >
  Domina vectores y espacios vectoriales para machine learning: intuición geométrica, producto punto, similitud coseno, normas, código NumPy y perspectiva de investigación para científicos en Machine Learning (ML).
date: 2026-03-06
tags: ["AI", "Maths", "Algebra", "Vectors"]
---

En 2017, investigadores de Google Brain publicaron [**Attention Is All You Need**](https://arxiv.org/abs/1706.03762), el artículo que introdujo la arquitectura Transformer sobre la que hoy se construyen GPT-4, Gemini y prácticamente todos los modelos de lenguaje de vanguardia. En el corazón de esa arquitectura (y de toda red neuronal, sistema de recomendación y modelo de visión por computadora—) vive un objeto engañosamente simple: el **vector**.

Cuando un modelo de lenguaje lee la palabra *"banco"*, no ve una cadena de texto. Ve un vector en un espacio de 4096 dimensiones donde *"banco (financiero)"* y *"banco (asiento)"* ocupan regiones mensurablemente distintas. Cuando un motor de búsqueda decide que tu consulta coincide con un documento, está calculando el ángulo entre dos vectores. Cuando una red neuronal aprende, está desplazando vectores por el espacio en respuesta a un gradiente, que es, él mismo, otro vector.

Este artículo construye tu fundamento operativo para todo eso. Al terminar de leerlo, podrás:

- Definir formalmente vectores, escalares y espacios vectoriales, y explicar *por qué* importan los axiomas.
- Calcular normas, productos punto y ángulos entre vectores a mano y con *NumPy*.
- Razonar geométricamente sobre datos de alta dimensión, una habilidad indispensable en investigación en Machine Learning.
- Leer un artículo de investigación con notación vectorial sin perder el hilo.

Sin mas preámbulos, empecemos.

## Prerrequisitos

Antes de leer este artículo, deberías manejar con comodidad:

- **Álgebra de secundaria**: variables, funciones, el plano cartesiano.
- **Python básico**: listas, bucles, funciones, importación de bibliotecas.
- **Intuición de cálculo** (útil, no obligatorio): la idea de que una derivada apunta en la dirección de mayor pendiente.

## Intuición primero

### La analogía del programador: vectores como arreglos tipados con alma geométrica

Como desarrollador, has usado arreglos toda tu carrera. Una lista de Python `[3.0, -1.5, 7.2]` almacena tres números. Un vector es superficialmente lo mismo, pero con una estructura adicional crucial: **posición en el espacio y la geometría que conecta posiciones**.

Considera este ejemplo. Supón que tienes dos diccionarios que representan usuarios de una plataforma de comercio electrónico:

```python
usuario_A = {"edad": 28, "freq_compras": 5, "gasto_promedio": 1200.0}
usuario_B = {"edad": 29, "freq_compras": 6, "gasto_promedio": 1150.0}
```

Como diccionarios, son solo contenedores de datos. Puedes leer valores, pero *"¿qué tan similares son estos usuarios?"* no es una pregunta que el diccionario pueda responder de forma nativa. Ahora conviértelos en vectores:

```python
A = [28, 5, 1200.0]
B = [29, 6, 1150.0]
```

De repente tienes geometría. Puedes medir la *distancia* entre ellos, el *ángulo* que forman respecto al origen, y si uno es una *versión escalada* del otro. Este es el salto que dan los vectores sobre los arreglos simples: **viven en un espacio equipado con reglas para medir, comparar y transformar**.

### El diagrama conceptual: vectores como flechas

Imagina un sistema de coordenadas 2D estándar. El vector \(\mathbf{v} = [3, 2]\) es una flecha que parte del origen \((0, 0)\) y termina en el punto \((3, 2)\). Dos cosas lo definen completamente: su **magnitud** (qué tan larga es la flecha) y su **dirección** (hacia dónde apunta).

{{< figure
    src="images/vector.png"
    alt="Representación visual de un vector en dos dimensiones"
    caption="Representación visual de un vector en dos dimensiones"
    >}}

Esta interpretación geométrica no es solo azúcar visual. En aprendizaje automático, un punto de datos (una fila en tu dataset) *es* un vector, es una flecha en el espacio de características. Dos puntos similares son flechas apuntando en direcciones parecidas. Un valor atípico es una flecha apuntando hacia algún lugar inesperado. La reducción de dimensionalidad (PCA, UMAP) es el arte de encontrar un espacio de menor dimensión donde esas flechas cuenten esencialmente la misma historia.

### Escalares: el caso más simple

Un **escalar** es simplemente un número solo, sin dirección, sin componentes. La temperatura, el valor de pérdida (*loss*), la tasa de aprendizaje: todos son escalares. Cuando multiplicas un vector por un escalar, estiras o encoges la flecha sin rotarla:

Misma dirección, el doble de largo.
$$
2 \cdot \begin{pmatrix} 3 \\ 2 \end{pmatrix} = \begin{pmatrix} 2 \cdot 3 \\ 2 \cdot 2 \end{pmatrix} = \begin{pmatrix} 6 \\ 4 \end{pmatrix}
$$
```python
2 x [3, 2] = [6, 4]
```

Misma dirección, invertida (180°).
$$
-1 \cdot \begin{pmatrix} 3 \\ 2 \end{pmatrix} = \begin{pmatrix} -1 \cdot 3 \\ -1 \cdot 2 \end{pmatrix} = \begin{pmatrix} -3 \\ -2 \end{pmatrix}
$$
```python
-1 x [3, 2] = [-3, -2]
```

Esta operación, **multiplicación escalar**, es una de las dos operaciones fundamentales que definen un espacio vectorial.

{{< callout type="important" >}}
Cuando depures una red neuronal cuya pérdida explota (*exploding gradients*), frecuentemente significa que vectores de activaciones o gradientes se están escalando por factores mayores a 1.0 en cada capa. Entender la multiplicación escalar geométricamente te ayuda a ver *por qué* el recorte de gradientes (*gradient clipping*) o la normalización por lotes (*batch normalization*) restauran la estabilidad: están renormalizando la longitud de esas flechas hacia valores manejables.
{{< /callout >}}

## Derivación matemática

### Definiciones formales

#### Escalar

Un elemento de un campo \(\mathbb{F}\), para nuestros propósitos, un número real \(\mathbb{R}\) o complejo \(\mathbb{C}\). Se denota con itálicas estándar y generalmente letras griegas: \(\alpha\), \(\beta\), \(\lambda\).

#### Vector

Una tupla ordenada de escalares de \(\mathbb{F}\). Un vector real de \(n\) dimensiones es un elemento del espacio \(\mathbb{R}^n\):

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n$$

{{< callout type="info" >}}
En términos simples: \(\mathbf{v}\) es una columna de \(n\) números reales. El subíndice indica en qué *"ranura"* estás.
{{< /callout >}}

#### Espacio vectorial

Un conjunto \(V\) de vectores sobre un campo \(\mathbb{F}\), equipado con dos operaciones para las cuales será ***cerrado***.

**Suma de vectores**: \(\mathbf{u} + \mathbf{v} \in V\) para todo \(\mathbf{u}, \mathbf{v} \in V\). Operación interna con las propiedades:
- *Propiedad Asociativa*: \(\mathbf{u} + (\mathbf{v} + \mathbf{w}) = (\mathbf{u} + \mathbf{v}) + \mathbf{w}\).
- *Propiedad Conmutativa*: \(\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}\).
- *Existencia del elemento neutro*: existe un elemento \(\mathbf{0} \in V\), denominado vector nulo, tal que \(\mathbf{v} + \mathbf{0} = \mathbf{v}\) para todo \(\mathbf{v} \in V\).
- *Existencia del inverso aditivo*: para todo \(\mathbf{v} \in V\), existe un elemento \(\mathbf{-v} \in V\) tal que \(\mathbf{v} + (\mathbf{-v}) = \mathbf{0}\).

**Multiplicación por escalar**: \(\alpha \mathbf{v} \in V\) para todo \(\alpha \in \mathbb{F}, \mathbf{v} \in V\). Operación externa con las propiedades:
- *Propiedad Asociativa*: \(\alpha (\beta \mathbf{v}) = (\alpha \beta) \mathbf{v}\).
- *Existencia del elemento neutro*: existe un escalar \(\alpha\), tal que \(\alpha \mathbf{v} = \mathbf{v} \alpha = \mathbf{v}\) para todo \(\mathbf{v} \in V\).
- *Propiedad distributiva respecto de la suma vectorial*: Para cualquier escalar \(\alpha\), se cumple que \(\alpha (\mathbf{u} + \mathbf{v}) = \alpha \mathbf{u} + \alpha \mathbf{v}\) para todo \(\mathbf{u}, \mathbf{v} \in V\).
- *Propiedad distributiva respecto de la suma de escalares*: Para cualquier par de escalares \(\alpha\) y \(\beta\), se cumple que \((\alpha + \beta) \mathbf{v} = \alpha \mathbf{v} + \beta \mathbf{v}\) para todo \(\mathbf{v} \in V\).

{{< callout >}}
¿Qué significa que el espacio vectorial es cerrado para esas dos operaciones?

Significa que producen resultados que pertenecen al mismo espacio vectorial.
- Si \(\mathbf{u} \in V\) y \(\mathbf{v} \in V\) entonces \((\mathbf{u} + \mathbf{v}) \in V\).
- Para cualquier escalar \(\alpha \in \mathbb{R}\) y vector \(\mathbf{v} \in V\), entonces \(\alpha \mathbf{v} \in V\).
{{< /callout >}}

### Suma de vectores

Dados \(\mathbf{u} = [u_1, u_2, \ldots, u_n]^T\) y \(\mathbf{v} = [v_1, v_2, \ldots, v_n]^T\):

$$
\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{bmatrix}
$$

{{< callout type="info" >}}
En términos simples: suma componente a componente. Geométricamente, coloca el inicio de \(\mathbf{v}\) en el extremo de \(\mathbf{u}\), el resultado es la flecha del punto de partida al punto de llegada según la [regla del paralelogramo](https://es.wikipedia.org/wiki/Regla_del_paralelogramo).
{{< /callout >}}

### Normas de vectores

La **norma** de un vector mide su longitud. La más común es la **norma euclidiana** (norma \(L^2\)):

$$
\|\mathbf{v}\|_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \sqrt{\sum_{i=1}^{n} v_i^2}
$$

{{< callout type="info" >}}
En términos simples: se eleva al cuadrado cada componente, se suman y luego se calcula la raíz. Es exactamente el teorema de Pitágoras generalizado a \(n\) dimensiones.
{{< /callout >}}

La familia general es la **norma \(L^p\)**:

$$\|\mathbf{v}\|_p = \left( \sum_{i=1}^{n} |v_i|^p \right)^{1/p}$$

Dos casos especiales aparecen constantemente en Machine Learning (ML):

- **Norma \(L^1\) (Manhattan)**:

    Usada en regularización LASSO porque induce dispersión (*sparsity*), penaliza cualquier componente no nulo por igual.
    $$
    \|\mathbf{v}\|_1 = \sum_{i=1}^{n} |v_i|
    $$

- **Norma \(L^\infty\) (norma máximo)**:

    Útil cuando te importa la activación más grande de un vector, ignorando las demás.
    $$
    \|\mathbf{v}\|_\infty = \max_i |v_i|
    $$

Para investigar más sobre normas de vectores, entrá a [este](https://en.wikipedia.org/wiki/Norm_(mathematics)) artículo de Wikipedia.

### El producto punto

El **producto punto** (o producto interno) de dos vectores es:

$$
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n
$$

{{< callout type="info" >}}
En términos sencillos: multiplica los componentes correspondientes de cada vector y suma los resultados. El resultado es un escalar, un número que codifica cuánto ***"se alinean"*** los dos vectores.
{{< /callout >}}

{{< callout >}}
El producto punto de un vector consigo mismo es igual a su magnitud al cuadrado.
$$
\mathbf{v} \cdot \mathbf{v} = \|\mathbf{v}\|^2
$$
{{< /callout >}}

### El ángulo entre vectores

Aquí es donde la geometría y el álgebra se fusionan elegantemente. De la geometría euclidiana, la [**Ley de Cosenos**](https://es.wikipedia.org/wiki/Teorema_del_coseno) establece que para un triángulo con lados \(a\), \(b\), \(c\) y ángulo \(\theta\) opuesto al lado \(c\):

$$
c^2 = a^2 + b^2 - 2 a b \cos(\theta)
$$

Apliquemos esto a vectores. Sean \(\mathbf{u}\) y \(\mathbf{v}\) dos vectores. El tercer lado del triángulo que forman es \(\mathbf{u} - \mathbf{v}\).

Sustituyendo \(a = \|\mathbf{u}\|\), \(b = \|\mathbf{v}\|\), \(c = \|\mathbf{u} - \mathbf{v}\|\):

$$
\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2 \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)
$$

Expandiendo el lado izquierdo mediante el producto punto:

$$
\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2 \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)
$$

$$
(\mathbf{u} - \mathbf{v}) \cdot (\mathbf{u} - \mathbf{v}) = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2 \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)
$$

$$
\|\mathbf{u}\|^2 - 2 (\mathbf{u} \cdot \mathbf{v}) + \|\mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2 \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)
$$

Cancelando \(\|\mathbf{u}\|^2\) y \(\|\mathbf{v}\|^2\) de ambos lados:
$$
-2 (\mathbf{u} \cdot \mathbf{v}) = - 2 \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)
$$

Dividiendo ambos lados entre \(-2 \|\mathbf{u}\| \|\mathbf{v}\|\) (asumiendo que ninguno es el vector cero):

$$
\frac{-2 (\mathbf{u} \cdot \mathbf{v})}{-2 \|\mathbf{u}\| \|\mathbf{v}\|} = \frac{-2 \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)}{-2 \|\mathbf{u}\| \|\mathbf{v}\|}
$$

$$
\boxed{\frac{(\mathbf{u} \cdot \mathbf{v})}{\|\mathbf{u}\| \|\mathbf{v}\|} = \cos(\theta)}
$$

{{< callout type="info" >}}
En términos sencillos: el coseno del ángulo entre dos vectores es igual a su producto punto dividido entre el producto de sus longitudes. Esta fórmula es ***fundamental***, nos da la **similitud coseno**, una de las métricas de distancia más utilizadas en Machine Learning.
{{< /callout >}}

Interpretaciones clave:
- \(\cos\theta = 1\) (\(\theta = 0°\)): los vectores apuntan en la **misma dirección** (temas idénticos en un embedding de documentos).
- \(\cos\theta = 0\) (\(\theta = 90°\)) los vectores son **ortogonales**, completamente no relacionados
- \(\cos\theta = -1\) (\(\theta = 180°\)): los vectores apuntan en **direcciones opuestas** (antónimos en un espacio de embeddings bien entrenado).

{{< callout >}}
En el paper original de [Word2Vec](https://arxiv.org/abs/1301.3781) (Mikolov et al., 2013), la famosa analogía:
$$
rey − hombre + mujer ≈ reina
$$
funciona precisamente gracias a esta geometría. Las relaciones semánticas se codifican como *direcciones* en el espacio vectorial, y encontrar `reina` significa hallar el vector con mayor similitud coseno al vector de consulta. Todo modelo de embedding moderno (BERT, GPT, sentence-transformers) hereda esta filosofía geométrica. La siguiente vez que leas sobre ***"espacios de representación"*** en un paper de IA, recuerda: están hablando literalmente de la geometría que acabas de derivar.
{{< /callout >}}

## Implementación en Python

Implementamos todo desde cero: primero en Python puro para ver la mecánica, luego verificamos con NumPy.

{{< tabs >}}
  {{< tab name="Python puro" >}}
    {{< details title="Código" closed="true" >}}
      {{< codeimporter
          url="https://raw.githubusercontent.com/learn-software-engineering/examples/refs/heads/main/ai/math/algebra/vectors/vector_pure_python.es.py"
          type="python"
          >}}
    {{< /details >}}
    {{< details title="Salida" closed="true" >}}
      ```bash
      >  python vector_pure_python.es.py
      =======================================
      Operaciones vectoriales con Python puro
      =======================================
      u = [1.0, 2.0, 3.0]
      v = [4.0, 0.0, -1.0]
      Suma (u + v)                   -> [5.0, 2.0, 2.0]
      Multiplicación escalar (2 * u) -> [2.0, 4.0, 6.0]
      Norma L2 de u (||u||₂)         -> 3.7417
      Norma L1 de u (||u||₁)         -> 6.0000
      Producto punto (u · v)         -> 1.0000
      Similitud coseno               -> 0.0648
      Ángulo entre u y v             -> 86.28°
      Vector unitario de u (û)       -> [0.2673, 0.5345, 0.8018]
      Verificar `||û||₂ = 1`         -> 1.000000
      ==========================
      Demo de analogía semántica
      ==========================
      En NLP real, estos serían embeddings aprendidos por un modelo.
      Aquí ilustramos el principio con vectores de características
      creados manualmente.
      Características: [realeza, masculinidad, edad_relativa, poder]
      rey    = [0.9, 0.9, 0.8, 0.9]
      reina  = [0.9, 0.1, 0.8, 0.9]
      hombre = [0.0, 0.9, 0.5, 0.4]
      mujer  = [0.0, 0.1, 0.5, 0.4]
      La famosa analogía: rey - hombre + mujer ≈ reina
      rey - hombre + mujer -> [0.9, 0.1, 0.8, 0.9]
      Similitud coseno con 'reina': 1.0000
      Similitud coseno con 'rey':   0.8902
      ==> El vector de analogía está más cerca de 'reina' que de 'rey'.
      ```
    {{< /details >}}
  {{< /tab >}}
  {{< tab name="Numpy" >}}
    {{< details title="Código" closed="true" >}}
      {{< codeimporter
          url="https://raw.githubusercontent.com/learn-software-engineering/examples/refs/heads/main/ai/math/algebra/vectors/vector_numpy.es.py"
          type="python"
          >}}
    {{< /details >}}
    {{< details title="Salida" closed="true" >}}
      ```bash
      >  python vector_numpy.es.py
      =================================
      Operaciones vectoriales con Numpy
      =================================
      u = [1. 2. 3.]
      v = [ 4.  0. -1.]
      Suma (u + v)                   -> [5. 2. 2.]
      Multiplicación escalar (2 * u) -> [2. 4. 6.]
      Norma L2 de u (||u||₂)         -> 3.7417
      Norma L1 de u (||u||₁)         -> 6.0000
      Producto punto (u · v)         -> 1.0000
      Similitud coseno               -> 0.0648
      Ángulo entre u y v             -> 86.28°
      Vector unitario de u (û)       -> [np.float64(0.2673), np.float64(0.5345), np.float64(0.8018)]
      Verificar `||û||₂ = 1`         -> 1.000000
      ==========================
      Demo de analogía semántica
      ==========================
      En NLP real, estos serían embeddings aprendidos por un modelo.
      Aquí ilustramos el principio con vectores de características
      creados manualmente.
      Características: [realeza, masculinidad, edad_relativa, poder]
      rey    = [0.9 0.9 0.8 0.9]
      reina  = [0.9 0.1 0.8 0.9]
      hombre = [0.  0.9 0.5 0.4]
      mujer  = [0.  0.1 0.5 0.4]
      La famosa analogía: rey - hombre + mujer ≈ reina
      rey - hombre + mujer -> [np.float64(0.9), np.float64(0.1), np.float64(0.8), np.float64(0.9)]
      Similitud coseno con 'reina': 1.0000
      Similitud coseno con 'rey':   0.8902
      ==> El vector de analogía está más cerca de 'reina' que de 'rey'.
      ```
    {{< /details >}}
  {{< /tab >}}
{{< /tabs >}}

## Perspectiva de Machine Learning e IA

Los vectores no son un concepto preliminar del que te graduarás, son la *lingua franca* de la investigación moderna en IA. Aquí tres formas en que aparecen en en contexto de Inteligencia Artificial y Machine Learning.

**Espacios de embedding y aprendizaje de representaciones**. Todo modelo de aprendizaje profundo moderno aprende a representar entradas como vectores. Los embeddings de tokens del Transformer que se describen en el artículo [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), son vectores en \(\mathbb{R}^{d_{model}}\) (típicamente entre 512 y 4096 dimensiones). El proceso de entrenamiento completo puede verse como una optimización de la geometría de esos vectores para que entradas semánticamente similares queden agrupadas. La investigación en [*aprendizaje contrastivo*](https://arxiv.org/abs/2002.05709) enmarca explícitamente el objetivo del aprendizaje como: `acercar vectores de muestras similares y alejar vectores de muestras distintas en el espacio de embedding`.

**Generación Aumentada con Recuperación (RAG) y bases de datos vectoriales**. Con la explosión de los LLMs, una dirección clave de investigación aplicada es la búsqueda eficiente del vecino más cercano sobre miles de millones de vectores. En el artículo [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html), Lewis et al. demostraron que aumentar la generación con documentos recuperados mejora dramáticamente la precisión factual. El paso de recuperación completo es una búsqueda por similitud coseno, exactamente la fórmula que vimos antes.

**Dimensionalidad**. La geometría de los espacios de alta dimensión es profundamente contraria a la intuición, un fenómeno conocido como la *maldición de la dimensionalidad*. En dimensiones muy altas, casi todos los pares de vectores se vuelven casi ortogonales (\(\cos\theta \approx 0\)), lo que puede degradar la similitud coseno como métrica significativa. Entender *cuándo* falla la similitud coseno y qué alternativas geométricas existen ([espacios hiperbólicos](https://es.wikipedia.org/wiki/Espacio_hiperb%C3%B3lico), [variedades de Riemann](https://es.wikipedia.org/wiki/Variedad_de_Riemann)) es un área de investigación activa. Si esto te interesa, [*Poincaré Embeddings*](https://arxiv.org/abs/1705.08039) es un excelente punto de entrada.

## Errores comunes y depuración

1. **Olvidar normalizar antes de computar similitud coseno, y cuándo NO hacerlo**. La similitud coseno mide dirección solamente, descartando la magnitud. Si dos documentos son similares pero uno es diez veces más largo, la similitud coseno ignora esa diferencia. A veces eso es un error (cuando la magnitud importa), a veces es una ventaja (clasificación de sentimiento donde importa el tema, no la extensión). Siempre pregunta: ¿debería importar la magnitud aquí? Si la respuesta es sí, usa distancia euclidiana en su lugar.

2. **Errores de dimensión que producen resultados silenciosamente incorrectos**. En NumPy, `np.dot(u, v)` lanzará un error si las dimensiones no coinciden para arreglos 1D, pero con arreglos 2D (matrices), NumPy puede hacer *broadcasting* de formas que devuelven un resultado con la forma equivocada. Siempre verifica `assert u.shape == v.shape` antes de productos punto en código de investigación, o usa `np.einsum` con notación de índices explícita para mayor claridad.

3. **El caso límite del vector cero**. `similitud_coseno([0,0,0], [1,2,3])` es matemáticamente indefinido (estás dividiendo entre cero). En sistemas de producción que calculan embeddings, un vector cero normalmente señala un error en etapas anteriores: entrada vacía, tokenización incorrecta, o una capa de red que colapsó. Si ves pérdidas `NaN`, verifica las normas de tus embeddings primero.

4. **Errores de precisión de punto flotante en `arccos`**. Debido a la aritmética de punto flotante, los productos punto pueden ocasionalmente producir valores de coseno ligeramente fuera de \([-1, 1]\) (por ejemplo, `1.0000000002`). Pasar este valor directamente a `math.acos()` lanza un `ValueError`. Aplica siempre *clamp*: `cos_theta = max(-1.0, min(1.0, cos_theta))` antes de llamar a `arccos`.

5. **Confundir los efectos de regularización \(L^1\) y \(L^2\)**. La regularización \(L^2\) (*weight decay*) encoge todos los pesos proporcionalmente, nunca los lleva exactamente a cero. La regularización \(L^1\) sí lleva pesos a cero, creando modelos dispersos. Esto es consecuencia directa de la geometría de las bolas de norma \(L^1\) vs \(L^2\). Elegir el regularizador equivocado es fuente común de modelos demasiado densos (cómputo desperdiciado) o no suficientemente dispersos (baja interpretabilidad).

## Resumen y lo que sigue

**Puntos clave de este artículo:**

- Un **vector** es una tupla ordenada de escalares que vive en un espacio geométrico, tiene magnitud y dirección.
- Un **espacio vectorial** se define por cerradura bajo adición y multiplicación escalar, por eso las operaciones lineales en redes neuronales se comportan tan bien.
- La **norma \(L^2\)** mide longitud euclidiana. La **norma \(L^1\)** mide distancia Manhattan y promueve dispersión.
- El **producto punto** (\(\mathbf{u} \cdot \mathbf{v}\)) mide alineación. Dividir entre ambas normas da la **similitud coseno**, la métrica angular central en recuperación, embeddings y mecanismos de atención.
- La fórmula \(\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}\) se *deriva* de la Ley de Cosenos, no es arbitraria, es geometría.
- Los vectores de alta dimensión son el lenguaje en el que está escrito todo el Machine Learning moderno. La fluidez aquí se multiplica en cada tema futuro.

**En el próximo artículo (Matrices y Transformaciones Lineales)**: Generalizamos de vectores individuales a colecciones de vectores, introducimos las matrices como funciones que transforman espacios vectoriales, y derivamos las reglas de multiplicación de matrices desde primeros principios. Verás exactamente por qué una capa totalmente conectada en una red neuronal no es más que una multiplicación de matrices seguida de una no linealidad.

---

{{< callout icon="sparkles" >}}
¡Gracias por llegar hasta acá! Espero que este recorrido por el universo de la IA haya sido tan apasionante para vos como lo fue para mí escribirlo.

Nos encantaría escuchar lo que pensás, así que no te quedes callado/a, dejá tus comentarios, sugerencias y todas esas ideas copadas que seguro se te ocurrieron.

Vas a encontrar todo el código y los proyectos en nuestro repositorio de GitHub [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

¡Gracias por ser parte de esta comunidad de aprendizaje. Seguí programando y explorando nuevos territorios en este fascinante mundo de la computación!
{{< /callout >}}
