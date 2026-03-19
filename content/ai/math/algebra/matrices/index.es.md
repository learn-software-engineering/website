---
weight: 2
title: "Matrices: operaciones y propiedades"
authors:
  - jnonino
description: >
  Domina la aritmética matricial, transformaciones lineales, determinantes, rango e inversas desde cero. Derivaciones matemáticas paso a paso, código Python ejecutable e insights de nivel investigación para aspirantes a Científicos de ML.
date: 2026-03-13
tags: ["IA", "Matemáticas", "Álgebra", "Álgebra Lineal", "Matrices"]
---

GPT-3 almacena su *"inteligencia"* en aproximadamente 1800 matrices de pesos. Cada *token* que escribís dispara una cascada de multiplicaciones a través de esas matrices (proyecciones, puntuaciones de atención, expansiones feed-forward) antes de que se produzca una única probabilidad de salida. El modelo completo de 175.000 millones de parámetros es, en su núcleo computacional, una composición intrincada de operaciones matriciales aplicadas en secuencia.

Las matrices hacen que la idea abstracta de un espacio vectorial se convierta en un motor que podés ejecutar de verdad. Un vector te dice dónde está algo; una matriz te dice cómo moverlo. Las matrices de pesos en las redes neuronales codifican transformaciones aprendidas del espacio de características. Las matrices de covarianza en estadística codifican la forma de una distribución de datos. La matriz Jacobiana de una función codifica cómo cambian sus salidas respecto a sus entradas, es el objeto que hace posible la *retropropagación*.

El problema es que la mayoría de las introducciones a las matrices las tratan como planillas de cálculo glorificadas: grillas de números con reglas aritméticas pegadas encima. Esa perspectiva hace que las operaciones parezcan arbitrarias. Este artículo toma el enfoque opuesto: arrancamos desde la geometría, derivamos cada operación desde primeros principios, y al final vas a ver las matrices no como tablas sino como **funciones**, con toda la riqueza que eso implica: composición, invertibilidad, imagen, núcleo y rango.

Al finalizar este artículo, serás capaz de:

- Definir formalmente una matriz y una transformación lineal, y explicar por qué importan los axiomas.
- Ejecutar y derivar todas las operaciones matriciales fundamentales: suma, multiplicación, transpuesta, traza e inversión.
- Calcular el determinante de manera geométrica y algebraica, y entender qué significa un determinante nulo.
- Razonar sobre el espacio columna, espacio fila, espacio nulo y rango. Enunciar y aplicar el Teorema Rango-Nulidad.
- Implementar estas operaciones desde cero en Python puro y NumPy.
- Conectar estas operaciones con la investigación contemporánea en aprendizaje profundo (*deep learning*).

Comencemos.

## Prerrequisitos

Antes de leer este artículo, deberías dominar:

- **Vectores y espacios vectoriales**: definiciones formales, productos punto, normas, espacio generado y combinaciones lineales.
- **Python y NumPy básico**: creación de arreglos, indexación, manipulación de formas (*shapes*).
- **Notación de funciones**: entender notación del estilo \(f: \mathbb{R}^m \rightarrow \mathbb{R}^n\).

Si puedes definir qué significa que un conjunto sea *cerrado* bajo una operación, tienes la base suficiente.

## Intuición primero

### La analogía del programador: matrices como funciones

Como desarrollador, has usado funciones toda tu carrera. Una función `transformar(x)` recibe una entrada y la mapea a una salida. Una matriz \(\mathbf{A}\) es exactamente eso, una **función** que toma un vector como entrada y produce un vector como salida. La restricción clave es que esta función debe ser *lineal*, lo que impone una estructura geométrica específica sobre qué transformaciones están permitidas.

Pensalo de esta manera. Imaginate un pipeline de datos donde vectores de características de usuarios pasan por etapas de procesamiento:

```python
# Etapa 1: expandir 3 características crudas a 5 características derivadas
etapa1 = transformar_3_a_5(caracteristicas_usuario)   # shape: (5,)

# Etapa 2: comprimir 5 características derivadas a 2 dimensiones latentes
etapa2 = transformar_5_a_2(etapa1)                    # shape: (2,)

# Combinado: ¿podemos hacer ambas en un solo paso?
combinado = transformar_3_a_2(caracteristicas_usuario)  # Sí, multiplicación de matrices
```

Esto es exactamente lo que calcula la multiplicación de matrices: la **composición** de dos transformaciones lineales en una. Una matriz \(5 \times 3\) (etapa 1) compuesta con una matriz \(2 \times 5\) (etapa 2) produce una única matriz \(2 \times 3\) que hace ambos pasos a la vez. Cada capa de una red neuronal es una etapa de este pipeline.

### La imagen geométrica: matrices como transformaciones del espacio

Imagina un sistema de coordenadas 2D. Los vectores de la [base canónica](https://es.wikipedia.org/wiki/Base_can%C3%B3nica) son:

$$
\hat{e}_1 = [1,0] \qquad \text{(apunta a la derecha a lo largo del eje x)}
$$

$$
\hat{e}_2 = [0,1] \qquad \text{(apunta hacia arriba a lo largo del eje y)}
$$

Aplica la matriz \(\mathbf{A} = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}\). Tras la transformación: \(\hat{e}_1\) se mapea a \([2, 0]\) (estirado a la derecha por factor 2) y \(\hat{e}_2\) se mapea a \([0, 3]\) (estirado hacia arriba por factor 3). El cuadrado unitario (con área 1), se convierte en un rectángulo de \(2 \times 3\) con área 6. El **determinante** de \(\mathbf{A}\), que derivaremos en breve, es exactamente \(6\). Esto no es una coincidencia: el determinante *es* el factor de escalado del volumen.

Compara esto con una matriz de rotación:

$$
\mathbf{R}(\theta) = \begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{bmatrix}
$$

Ésta rota cada vector por el ángulo \(\theta\) sin estirarlo, por lo que preserva todas las áreas, y su determinante siempre es 1.

Las columnas de cualquier matriz te dicen exactamente adónde van los vectores de la base. Dado que todo vector es una combinación lineal de los vectores base, saber adónde van los vectores base te dice adónde va *cada* vector del espacio. Este razonamiento explica por qué la multiplicación de matrices no es conmutativa (aplicar la transformación A y luego B es geométricamente diferente de B y luego A), y por qué las columnas de las matrices de pesos en las redes neuronales tienen un significado semántico que los investigadores analizan activamente.

{{< callout type="important" >}}
Las columnas de una matriz no son sólo números, son las *imágenes de los vectores de la base* bajo la transformación. Cuando examinas la matriz de pesos \(\mathbf{W}\) de una capa de red neuronal entrenada, cada columna te dice cómo responde esa capa a una dirección de entrada estándar. Éste es el fundamento de la investigación de visualización de características, donde los practicantes interpretan qué *"detecta"* cada neurona examinando las direcciones en las matrices de pesos.
{{< /callout >}}

## Derivación matemática

### Definición formal

Una **matriz \(m \times n\)** sobre \(\mathbb{R}\) es un arreglo rectangular de números reales con \(m\) filas y \(n\) columnas:

$$
\mathbf{A} = \begin{bmatrix} A_{11} & A_{12} & \cdots & A_{1n} \\ A_{21} & A_{22} & \cdots & A_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ A_{m1} & A_{m2} & \cdots & A_{mn} \end{bmatrix} \in \mathbb{R}^{m \times n}
$$

La entrada \(A_{ij}\) denota el elemento en la **fila \(i\), columna \(j\)**. Escribimos el conjunto de todas las matrices reales \(m \times n\) como \(\mathbb{R}^{m \times n}\).

{{< callout type="info" >}}
En palabras sencillas: una matriz es una tabla 2D de números. La notación \(\mathbb{R}^{m \times n}\) es el *"tipo"* de la matriz (cuántas filas y columnas tiene). Es exactamente la firma de tipo que adjuntarías a un arreglo 2D en un lenguaje de tipado estático: `Array[Float, m, n]`.
{{< /callout >}}

**Casos especiales** que encontrarás constantemente:

- **Vector columna**: una matriz de forma \(n \times 1\), es simplemente un vector \(\mathbf{v} \in \mathbb{R}^n\).
- **Vector fila**: una matriz de forma \(1 \times n\).
- **Matriz cuadrada**: una matriz donde \(m = n\).
- **Matriz identidad** \(\mathbf{I}_n\): la matriz \(n \times n\) con \(I_{ij} = 1\) si \(i = j\), de lo contrario \(0\).
- **Matriz cero** \(\mathbf{0}\): todas las entradas son cero.
- **Matriz diagonal**: una matriz cuadrada donde \(A_{ij} = 0\) para todo \(i \neq j\).

### Las matrices como transformaciones lineales

La propiedad más importante de una matriz es que define una **transformación lineal** \(T: \mathbb{R}^n \rightarrow \mathbb{R}^m\) mediante \(T(\mathbf{x}) = \mathbf{A}\mathbf{x}\).

Una función \(T\) se llama **lineal** si y solo si satisface dos axiomas para todos los vectores \(\mathbf{u}, \mathbf{v} \in \mathbb{R}^n\) y todos los escalares \(\alpha \in \mathbb{R}\):

- *Aditividad*: \(T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})\)
- *Homogeneidad*: \(T(\alpha \mathbf{u}) = \alpha T(\mathbf{u})\)

Estos dos axiomas son equivalentes a la condición única:

$$
T(\alpha \mathbf{u} + \beta \mathbf{v}) = \alpha T(\mathbf{u}) + \beta T(\mathbf{v})
$$

{{< callout type="info" >}}
En términos sencillos: una transformación lineal preserva la estructura del espacio vectorial, no importa si sumas dos vectores primero y luego transformas, o si transformas cada uno por separado y luego sumas. Por eso apilar múltiples capas lineales (sin activación) en una red neuronal se colapsa en un único producto matricial: \(\mathbf{W}_2(\mathbf{W}_1\mathbf{x}) = (\mathbf{W}_2\mathbf{W}_1)\mathbf{x}\).
{{< /callout >}}

Un teorema fundamental del álgebra lineal establece que **toda** transformación lineal entre espacios de dimensión finita puede representarse como una matriz, y a la inversa, toda matriz define una transformación lineal. La matriz y la transformación lineal son, a efectos prácticos, el mismo objeto.

### Suma de matrices y multiplicación por escalar

El conjunto \(\mathbb{R}^{m \times n}\) de todas las matrices reales \(m \times n\) forma un **espacio vectorial** bajo las siguientes operaciones.

**Suma de matrices**: Dadas \(\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}\):

$$
(\mathbf{A} + \mathbf{B})_{ij} = A_{ij} + B_{ij}
$$

Esta operación satisface:
- *Conmutatividad*: \(\mathbf{A} + \mathbf{B} = \mathbf{B} + \mathbf{A}\)
- *Asociatividad*: \((\mathbf{A} + \mathbf{B}) + \mathbf{C} = \mathbf{A} + (\mathbf{B} + \mathbf{C})\)
- *Elemento identidad*: existe una matriz cero \(\mathbf{0}\) tal que \(\mathbf{A} + \mathbf{0} = \mathbf{A}\)
- *Elemento inverso*: para cada \(\mathbf{A}\), existe \(-\mathbf{A}\) tal que \(\mathbf{A} + (-\mathbf{A}) = \mathbf{0}\)

**Multiplicación por escalar**: Dados \(\alpha \in \mathbb{R}\) y \(\mathbf{A} \in \mathbb{R}^{m \times n}\):

$$
(\alpha \mathbf{A})_{ij} = \alpha \cdot A_{ij}
$$

Esta operación satisface:
- *Asociatividad*: \(\alpha(\beta \mathbf{A}) = (\alpha\beta)\mathbf{A}\)
- *Elemento identidad*: \(1 \cdot \mathbf{A} = \mathbf{A}\)
- *Distributividad respecto a la suma de matrices*: \(\alpha(\mathbf{A} + \mathbf{B}) = \alpha\mathbf{A} + \alpha\mathbf{B}\)
- *Distributividad respecto a la suma de escalares*: \((\alpha + \beta)\mathbf{A} = \alpha\mathbf{A} + \beta\mathbf{A}\)

{{< callout >}}
Si estos axiomas te resultan familiares, tiene todo el sentido. Son exactamente los axiomas de espacio vectorial que vimos en el artículo sobre vectores. Las matrices *"son"* vectores en un espacio de mayor dimensión. Una matriz \(3 \times 4\) puede ser vista como un vector de \(12\) componentes dispuestos en una cuadrícula. Esto significa que todo teorema que demostremos sobre espacios vectoriales también aplica a matrices.
{{< /callout >}}

### Multiplicación de matrices

Dados \(\mathbf{A} \in \mathbb{R}^{m \times k}\) y \(\mathbf{B} \in \mathbb{R}^{k \times n}\), su producto \(\mathbf{C} = \mathbf{A}\mathbf{B} \in \mathbb{R}^{m \times n}\) se define como:

$$
\boxed{C_{ij} = \sum_{l=1}^{k} A_{il} \cdot B_{lj}}
$$

Para calcular la entrada \(C_{ij}\), el elemento en la fila \(i\), columna \(j\) del resultado, toma el **producto punto de la fila \(i\) de \(\mathbf{A}\) con la columna \(j\) de \(\mathbf{B}\)**. Las dimensiones internas deben coincidir:

$$\underbrace{\mathbf{A}}_{m \times k} \cdot \underbrace{\mathbf{B}}_{k \times n} = \underbrace{\mathbf{C}}_{m \times n}$$

{{< callout type="info" >}}
En otros términos: la multiplicación de matrices es composición de funciones. Aplicar la transformación \(\mathbf{B}\) primero, y luego \(\mathbf{A}\), produce el mismo resultado que la única transformación compuesta \(\mathbf{AB}\). Las dimensiones internas deben coincidir porque la dimensión de salida de la primera transformación debe igualar la dimensión de entrada de la segunda, exactamente como componer funciones.
{{< /callout >}}

La multiplicación de matrices satisface:
- *Asociatividad*: \((\mathbf{AB})\mathbf{C} = \mathbf{A}(\mathbf{BC})\)
- *Distributividad izquierda*: \(\mathbf{A}(\mathbf{B} + \mathbf{C}) = \mathbf{AB} + \mathbf{AC}\)
- *Distributividad derecha*: \((\mathbf{A} + \mathbf{B})\mathbf{C} = \mathbf{AC} + \mathbf{BC}\)
- *Identidad*: \(\mathbf{A}\mathbf{I} = \mathbf{I}\mathbf{A} = \mathbf{A}\) (para \(\mathbf{I}\) compatible)
- **No conmutatividad**: \(\mathbf{AB} \neq \mathbf{BA}\) en general

{{< callout >}}
La no conmutatividad no es un defecto sino una característica fundamental de las transformaciones. Rotar una figura y luego reflejarla es geométricamente diferente a reflejarla y luego rotarla.
{{< /callout >}}

### La transpuesta

La **transpuesta** de \(\mathbf{A} \in \mathbb{R}^{m \times n}\) es la matriz \(\mathbf{A}^\top \in \mathbb{R}^{n \times m}\) definida por:

$$
(\mathbf{A}^\top)_{ij} = A_{ji}
$$

{{< callout type="info" >}}
En palabras simples: voltea la matriz a lo largo de su diagonal principal, las filas se convierten en columnas y las columnas en filas. Una matriz \(3 \times 5\) se convierte en una \(5 \times 3\). Geométricamente, la transpuesta corresponde al *adjunto* de la transformación, la transformación que *"deshace la parte de rotación"* mientras conserva el escalado.
{{< /callout >}}

La transpuesta satisface las siguientes propiedades:

- *Doble transpuesta*: \((\mathbf{A}^\top)^\top = \mathbf{A}\)
- *Linealidad*: \((\mathbf{A} + \mathbf{B})^\top = \mathbf{A}^\top + \mathbf{B}^\top\) y \((\alpha\mathbf{A})^\top = \alpha\mathbf{A}^\top\)
- *Inversión del orden del producto*: \((\mathbf{AB})^\top = \mathbf{B}^\top \mathbf{A}^\top\)

La propiedad de inversión del orden es **crítica** y aparece en cada derivación de retropropagación. Vamos a probarla completamente. Queremos demostrar que \([(\mathbf{AB})^\top]_{ij} = [\mathbf{B}^\top \mathbf{A}^\top]_{ij}\).

Partimos de la definición de transpuesta:

$$
[(\mathbf{AB})^\top]_{ij} = [\mathbf{AB}]_{ji}
$$

Aplicando la definición de multiplicación de matrices a \([\mathbf{AB}]_{ji}\):

$$
[\mathbf{AB}]_{ji} = \sum_{l} A_{jl} \cdot B_{li}
$$

Reconociendo cada factor usando la definición de transpuesta (\(A_{jl} = [\mathbf{A}^\top]_{lj}\) y \(B_{li} = [\mathbf{B}^\top]_{il}\)):

$$
\sum_{l} A_{jl} B_{li} = \sum_{l} [\mathbf{B}^\top]_{il} \cdot [\mathbf{A}^\top]_{lj}
$$

Reconociendo la parte derecha como la definición de multiplicación de matrices \([\mathbf{B}^\top \mathbf{A}^\top]_{ij}\):

$$
\boxed{(\mathbf{AB})^\top = \mathbf{B}^\top \mathbf{A}^\top}
$$

**Matrices simétricas y antisimétricas** son casos especiales importantes:

- \(\mathbf{A}\) es **simétrica** si \(\mathbf{A}^\top = \mathbf{A}\), equivalentemente \(A_{ij} = A_{ji}\) para todo \(i, j\). Las matrices de covarianza y las matrices kernel en Machine Learning son siempre simétricas.
- \(\mathbf{A}\) es **antisimétrica** si \(\mathbf{A}^\top = -\mathbf{A}\), equivalentemente \(A_{ij} = -A_{ji}\) y todas las entradas diagonales son cero.

{{< callout >}}
Cualquier matriz cuadrada \(\mathbf{A}\) puede descomponerse de forma única en una parte simétrica y una parte antisimétrica:

$$
\mathbf{A} = \underbrace{\frac{\mathbf{A} + \mathbf{A}^\top}{2}}_{\text{simétrica}} + \underbrace{\frac{\mathbf{A} - \mathbf{A}^\top}{2}}_{\text{antisimétrica}}
$$

Esta descomposición tiene aplicaciones en física (análisis de deformación vs. rotación) y en investigación reciente sobre la estructura de los mecanismos de atención.
{{< /callout >}}

### La traza

Para una matriz cuadrada \(\mathbf{A} \in \mathbb{R}^{n \times n}\), la **traza** es la suma de las entradas diagonales:

$$
\text{tr}(\mathbf{A}) = \sum_{i=1}^{n} A_{ii}
$$

{{< callout type="info" >}}
En palabras simples: suma la diagonal principal. La traza captura cuánto la transformación *"expande"* el espacio en promedio.
{{< /callout >}}

Propiedades clave:
- *Linealidad*: \(\text{tr}(\mathbf{A} + \mathbf{B}) = \text{tr}(\mathbf{A}) + \text{tr}(\mathbf{B})\) y \(\text{tr}(\alpha\mathbf{A}) = \alpha\,\text{tr}(\mathbf{A})\)
- *Invarianza bajo transpuesta*: \(\text{tr}(\mathbf{A}^\top) = \text{tr}(\mathbf{A})\)
- **Propiedad cíclica**: \(\text{tr}(\mathbf{ABC}) = \text{tr}(\mathbf{BCA}) = \text{tr}(\mathbf{CAB})\)

La propiedad cíclica es omnipresente en las derivaciones de gradientes. Demostremos el caso base \(\text{tr}(\mathbf{AB}) = \text{tr}(\mathbf{BA})\).

Partiendo de la definición de traza:

$$
\text{tr}(\mathbf{AB}) = \sum_{i} [\mathbf{AB}]_{ii} = \sum_{i} \sum_{j} A_{ij} B_{ji}
$$

Intercambiando el orden de la suma (válido ya que ambas sumas son finitas):

$$
\sum_{i} \sum_{j} A_{ij} B_{ji} = \sum_{j} \sum_{i} B_{ji} A_{ij}
$$

Reconociendo la suma de la derecha como la entrada diagonal \([\mathbf{BA}]_{jj}\):

$$
\sum_{j} \sum_{i} B_{ji} A_{ij} = \sum_{j} [\mathbf{BA}]_{jj} = \text{tr}(\mathbf{BA})
$$

$$
\boxed{\text{tr}(\mathbf{AB}) = \text{tr}(\mathbf{BA})}
$$

{{< callout >}}
Notá que esto se cumple incluso cuando \(\mathbf{AB}\) y \(\mathbf{BA}\) tienen distintas formas (por ejemplo, \(\mathbf{A} \in \mathbb{R}^{m \times n}\), \(\mathbf{B} \in \mathbb{R}^{n \times m}\)), ambas trazas son escalares e iguales.
{{< /callout >}}

### El determinante

Antes de calcular ninguna fórmula, vale la pena hacerse las preguntas de fondo: ¿por qué existe el determinante? ¿Qué problema resuelve?

Cuando una matriz codifica una transformación lineal, la pregunta más fundamental que podés hacerle es: **¿destruye información?** Una transformación que aplasta un plano 2D en una recta 1D perdió toda la información de la dimensión colapsada, no puede deshacerse. Una transformación que rota o estira el plano preserva toda la información y puede revertirse. El determinante es el único número que responde a esta pregunta, e indica por cuánto.

Más precisamente, el determinante responde tres preguntas interconectadas de manera simultánea.

{{< details title="¿Cómo escala la transformación los volúmenes?" closed="true" >}}
En dos dimensiones, las dos columnas de una matriz son simplemente dos vectores dibujados desde el origen. Dos vectores desde el mismo origen siempre encierran un paralelogramo, no es una elección, es geometría. El determinante es el *área con signo* de ese paralelogramo. Si \(|\det(\mathbf{A})| = 6\), toda región del plano tiene su área multiplicada por 6 al aplicar la transformación. En 3D, se convierte en el volumen con signo del paralelepípedo formado por los tres vectores columna.

{{< figure
    src="images/volume-scale.es.svg"
    alt="Escala de volúmenes por el determinante"
    caption="Las dos columnas de una matriz \(2\times2\) son vectores que siempre forman un paralelogramo. El determinante es su área con signo."
    >}}
{{< /details >}}

{{< details title="¿Preserva o invierte la orientación?" closed="true" >}}
La palabra *con signo* importa. Dos vectores siempre forman un paralelogramo, pero el *orden* en que los recorrés determina el signo del área. Si la rotación de \(\mathbf{a}_1\) hacia \(\mathbf{a}_2\) es **antihoraria**, el determinante es positivo. Si es **horaria**, el determinante es negativo. El contenido geométrico es idéntico (misma figura, misma área) pero el signo codifica si la transformación preserva o invierte la lateralidad del espacio.

{{< figure
    src="images/orientation.es.svg"
    alt="Preservación o inversión de la orientación por el determinante"
    caption="El orden de los vectores columna determina el signo del área, codificando si la transformación preserva o invierte la orientación."
    >}}
{{< /details >}}

{{< details title="¿Es la transformación invertible?" closed="true" >}}
\(\det(\mathbf{A}) = 0\) significa que los dos vectores columna son paralelos, yacen sobre la misma recta, encerrando área cero. La transformación colapsa un plano 2D en una recta 1D. La información se pierde de manera irreversible: infinitas entradas distintas producen la misma salida, por lo que no podés determinar de cuál viniste. La matriz es **singular** y no tiene inversa.

{{< figure
    src="images/invertible.es.svg"
    alt="Transformación invertible"
    caption="Una transformación es invertible si y solo si su determinante es distinto de cero."
    >}}
{{< /details >}}

{{< callout type="important" >}}
En ML, verificar el determinante antes de invertir una matriz no es un formalismo matemático, es un paso de depuración práctico. Las matrices de covarianza casi singulares causan inestabilidad numérica en [procesos gaussianos](https://es.wikipedia.org/wiki/Proceso_de_Gauss), en la ecuación normal de la regresión lineal y en la reducción de dimensionalidad. Cuando un modelo produce salidas `NaN` o predicciones desproporcionadamente grandes después de una inversión matricial, un determinante cercano a cero suele ser el culpable.
{{< /callout >}}

#### Propiedades clave

- **Multiplicatividad**: \(\det(\mathbf{AB}) = \det(\mathbf{A})\det(\mathbf{B})\), aplicar una transformación que duplica el volumen seguida de una que lo triplica produce una expansión global de seis veces.
- **Invarianza bajo transpuesta**: \(\det(\mathbf{A}^\top) = \det(\mathbf{A})\), filas y columnas contribuyen simétricamente al volumen.
- **Escalado de fila**: escalar una fila por \(\alpha\) también escala el determinante por \(\alpha\), estás escalando una dimensión del paralelepípedo.
- **Intercambio de filas**: intercambiar dos filas niega el determinante, estás invirtiendo la orientación.
- **Adición de filas**: sumar un múltiplo de una fila a otra no cambia el determinante. Por esto la eliminación gaussiana funciona: nunca cambia el determinante.
- **Matrices triangulares**: \(\det(\mathbf{A}) = \prod_{i} A_{ii}\). Solo importan las entradas diagonales.
- **Invertibilidad**: \(\mathbf{A}\) es invertible si y solo si \(\det(\mathbf{A}) \neq 0\).

#### El caso 2x2, derivación desde la geometría

Sea \(\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}\). Los dos vectores columna son \(\mathbf{a}_1 = \begin{bmatrix}a \\ c\end{bmatrix}\) y \(\mathbf{a}_2 = \begin{bmatrix}b \\ d\end{bmatrix}\).

El determinante es el área con signo del paralelogramo que forman los cuatro vértices: \(O = (0,0)\), \(A = (a,c)\), \(B = (b,d)\) y \(C = (a+b,\, c+d)\).

$$
\boxed{\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc}
$$

Derivamos esto de dos maneras complementarias: primero geométricamente mediante sustracción del rectángulo delimitador, y luego algebraicamente mediante la fórmula de *Shoelace*.

{{< details title="Sustracción del rectángulo delimitador" closed="true" >}}
Encerrá el paralelogramo en el rectángulo alineado con los ejes más pequeño que lo contiene: un rectángulo de ancho \(a+b\) y alto \(c+d\). El área de este rectángulo es \((a+b)(c+d)\).

La región entre el rectángulo y el paralelogramo se compone de exactamente seis piezas:

- **Dos rectángulos** con lados \(b\) y \(c\) (esquinas superior izquierda e inferior derecha).
- **Dos triángulos** con catetos \(a\) y \(c\) (triángulos superior e inferior).
- **Dos triángulos** con catetos \(b\) y \(d\) (triángulos izquierdo y derecho).

{{< figure
    src="images/parallelogram-area.es.svg"
    alt="Área del paralelogramo"
    caption="El área del paralelogramo es la diferencia entre el rectángulo delimitador y el área de los triángulos."
    >}}

Expandiendo el área del rectángulo y restando las áreas de las seis piezas tendremos el area del paralelogramo (\(S\)):

$$
\begin{aligned}
  S &= (a+b)(c+d) - 2(bc) - 2\left(\frac{1}{2}ac\right) - 2\left(\frac{1}{2}bd\right) \\
  S &= ac + ad + bc + bd - 2(bc) - 2\left(\frac{1}{2}ac\right) - 2\left(\frac{1}{2}bd\right) \\
  S &= ac + ad + bc + bd - 2(bc) - ac - bd \\
  S &= ad - bc
\end{aligned}
$$
{{< /details >}}

{{< details title="Fórmula de Shoelace" closed="true" >}}
La **fórmula de Shoelace** (o fórmula del cordón de zapato) da el área con signo de cualquier polígono cuyos vértices se listen en orden. Para un polígono con \(n\) vértices \((x_1, y_1), \ldots, (x_n, y_n)\) recorridos en sentido antihorario:

$$
\text{área con signo} = \frac{1}{2}\sum_{i=1}^{n}\bigl(x_i y_{i+1} - x_{i+1} y_{i}\bigr)
$$

donde los índices son cíclicos, es decir, \(x_{n+1} = x_1\), \(y_{n+1} = y_1\), etcétera y \(n\) es el número de vértices.

La intuición detrás de cada término \(x_{i} y_{i+1} - x_{i+1} y_{i}\) es que mide el área con signo del triángulo formado por el origen, el vértice \(i\) y el vértice \(i+1\). Sumando estos triángulos alrededor del polígono se obtiene el área total con signo, positiva para recorrido antihorario, negativa para horario.

Aplicamos esto al paralelogramo con vértices en orden antihorario:

$$
\mathbf{O} = (0,0), \quad A = (a,c), \quad C = (a+b,\, c+d), \quad B = (b,d)
$$

Expandiendo la suma término a término:

**Término 1** (\(\mathbf{O} \to A\))
$$
\begin{aligned}
  x_{\mathbf{O}}\, y_A - x_A\, y_{\mathbf{O}} &= 0 \cdot c - a \cdot 0 \\
  x_{\mathbf{O}}\, y_A - x_A\, y_{\mathbf{O}} &= 0
\end{aligned}
$$

**Término 2** (\(A \to C\))
$$
\begin{aligned}
  x_A\, y_C - x_C\, y_A &= a(c+d) - (a+b)c \\
  x_A\, y_C - x_C\, y_A &= ac + ad - ac - bc \\
  x_A\, y_C - x_C\, y_A &= ad - bc
\end{aligned}
$$

**Término 3** (\(C \to B\))
$$
\begin{aligned}
  x_C\, y_B - x_B\, y_C &= (a+b)d - b(c+d) \\
  x_C\, y_B - x_B\, y_C &= ad + bd - bc - bd \\
  x_C\, y_B - x_B\, y_C &= ad - bc
\end{aligned}
$$

**Término 4** (\(B \to O\))
$$
\begin{aligned}
  x_B\, y_O - x_O\, y_B &= b \cdot 0 - 0 \cdot d \\
  x_B\, y_O - x_O\, y_B &= 0
\end{aligned}
$$

Sumando y aplicando el factor \(\tfrac{1}{2}\):
$$
\begin{aligned}
  \text{área con signo} &= \frac{1}{2}\bigl[0 + (ad - bc) + (ad - bc) + 0\bigr] \\
  \text{área con signo} &= \frac{1}{2} \cdot 2(ad-bc) \\
  \text{área con signo} &= ad - bc
\end{aligned}
$$

Esta derivación no asume ningún signo particular para \(a, b, c, d\) — vale para cualquier entrada real. El resultado es positivo cuando el recorrido \(\mathbf{O} \to A \to C \to B\) es antihorario (es decir, cuando \(ad > bc\)), y negativo cuando es horario. Ése es exactamente el signo que codifica la orientación.
{{< /details >}}

#### El caso 3x3, expansión por cofactores

En 3D, el determinante mide el volumen con signo del paralelepípedo formado por los tres vectores columna. El cálculo se expande a lo largo de la primera fila, donde cada término contribuye con un área proyectada ponderada por la entrada correspondiente. Dada la matriz \(\mathbf{A} \in \mathbb{R}^{3 \times 3}\):

$$
\mathbf{A} = \begin{bmatrix} A_{11} & A_{12} & A_{13} \\ A_{21} & A_{22} & A_{23} \\ A_{31} & A_{32} & A_{33} \end{bmatrix}
$$

$$
\det(\mathbf{A}) = A_{11}\det\begin{bmatrix}A_{22}&A_{23}\\A_{32}&A_{33}\end{bmatrix} - A_{12}\det\begin{bmatrix}A_{21}&A_{23}\\A_{31}&A_{33}\end{bmatrix} + A_{13}\det\begin{bmatrix}A_{21}&A_{22}\\A_{31}&A_{32}\end{bmatrix}
$$

{{< details title="Los menores de la matriz" closed="true" >}}
Cada matriz \(2 \times 2\) es un **menor** \(M_{1j}\): la submatriz obtenida borrando la fila 1 y la columna \(j\). El menor con signo \(C_{ij} = (-1)^{i+j} M_{ij}\) se llama **cofactor**. Los signos alternantes siguen el patrón de tablero de ajedrez que se muestra a continuación, lo que asegura que las contribuciones se combinen correctamente para dar el volumen con signo.

{{< figure
    src="images/cofactor-sign-pattern.svg"
    alt="Patrón de signos del cofactor"
    caption="El patrón de signos del cofactor sigue un patrón de tablero de ajedrez."
    >}}

{{< callout type="info" >}}
Elegí cualquier fila de la matriz. Para cada entrada, tapá su fila y su columna para revelar un bloque \(2 \times 2\), calculá su determinante, multiplicá por la entrada y por el signo del tablero de arriba, y sumá los tres resultados. Esto es recursivo: un determinante \(4 \times 4\) se expande en cuatro determinantes \(3 \times 3\), etc.
{{< /callout >}}
{{< /details >}}

#### La fórmula general de Leibniz

Para una matriz \(n \times n\), el determinante suma sobre todas las formas de elegir una entrada de cada fila y cada columna simultáneamente, es decir, sobre todas las permutaciones:

$$
\det(\mathbf{A}) = \sum_{\sigma \in S_n} \text{sgn}(\sigma) \prod_{i=1}^{n} A_{i,\sigma(i)}
$$

donde \(S_n\) es el conjunto de todas las permutaciones de \(\{1, \ldots, n\}\) y \(\text{sgn}(\sigma) = \pm 1\) es la paridad de la permutación (\(+1\) si la permutación es par y \(-1\) si es impar).

{{< callout type="important" >}}
Esta fórmula tiene \(n!\) términos, por lo que nunca se calcula directamente para matrices grandes. La descomposición LU calcula el determinante en \(\mathcal{O}(n^3)\) como subproducto de la factorización. Pero la fórmula de Leibniz es la herramienta correcta para demostrar identidades.
{{< /callout >}}

### Independencia lineal y espacios columna, fila y nulo

Entender una matriz significa entender los *espacios* que crea y destruye.

Un conjunto de vectores \(\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}\) es **linealmente independiente** si la única solución a:

$$
\alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \cdots + \alpha_k \mathbf{v}_k = \mathbf{0}
$$

es \(\alpha_1 = \alpha_2 = \cdots = \alpha_k = 0\). En otras palabras, ningún vector del conjunto puede escribirse como combinación de los demás. Las columnas linealmente dependientes son la firma de una matriz de rango deficiente.

Para \(\mathbf{A} \in \mathbb{R}^{m \times n}\), se definen tres subespacios fundamentales:

- **Espacio columna** (imagen), \(\text{col}(\mathbf{A}) \subseteq \mathbb{R}^m\): es el espacio generado por las columnas de \(\mathbf{A}\). Es el conjunto de todas las salidas posibles de la transformación \(\mathbf{A}\). El rango de \(\mathbf{A}\) es la dimensión de este espacio.

- **Espacio fila**, \(\text{row}(\mathbf{A}) \subseteq \mathbb{R}^n\): el espacio generado por las filas de \(\mathbf{A}\). Equivalentemente, \(\text{row}(\mathbf{A}) = \text{col}(\mathbf{A}^\top)\).

- **Espacio nulo** (núcleo), \(\text{null}(\mathbf{A}) \subseteq \mathbb{R}^n\): todos los vectores de entrada que \(\mathbf{A}\) mapea a cero:
    $$
    \text{null}(\mathbf{A}) = \{\mathbf{x} \in \mathbb{R}^n : \mathbf{A}\mathbf{x} = \mathbf{0}\}
    $$

{{< callout type="info" >}}
En otras palabras: el espacio nulo es el "punto ciego" de la transformación. Cualquier vector en el espacio nulo es invisible para la matriz, produce cero señal de salida sin importar cuán grande sea. En el contexto de redes neuronales, las direcciones en el espacio nulo de una matriz de pesos no reciben ningún gradiente de esa capa y nunca serán actualizadas por los gradientes de esa capa por sí solos.
{{< /callout >}}
