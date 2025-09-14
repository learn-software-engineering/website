---
draft: true
weight: 3
series: ["Matemática para Machine Learning"]
series_order: 3
title: "Cálculo Diferencial para Machine Learning: gradientes y optimización explicados para programadores"
description: "Descubre cómo el cálculo diferencial impulsa todos los algoritmos de machine learning. Aprende gradient descent, derivadas parciales y optimización desde cero con ejemplos prácticos en Python."
authors:
  - jnonino
date: 2025-09-09
tags: ["Inteligencia Artificial", "Aprendizaje Automático", "Machine Learning", "Matemática", "Cálculo Diferencia", "Gradiente", "Optimización"]
---
{{< katex >}}

# El Motor Invisible del Machine Learning: Cálculo Diferencial

Imagínate por un momento que tenés una pelota en la cima de una montaña irregular. Si la soltás, naturalmente va a rodar hacia abajo, buscando el punto más bajo posible. **Esa pelota está resolviendo un problema de optimización usando las leyes de la física.**

En el mundo del machine learning, nuestros algoritmos hacen exactamente lo mismo, pero en lugar de montañas físicas, navegan por paisajes matemáticos complejos llamados **funciones de costo**. Y la herramienta que les permite "saber hacia dónde rodar" es el **cálculo diferencial**.

{{< alert >}}
**¿Por qué necesitás entender cálculo para ML?** Porque literalmente **todos** los algoritmos de machine learning modernos (desde regresión lineal hasta transformers) usan gradient descent o variantes de optimización basadas en derivadas. Sin entender las derivadas, es como programar sin entender loops.
{{< /alert >}}

En las primeras dos semanas vimos los **qué** (fundamentos de IA) y los **dónde** (álgebra lineal para representar datos). Esta semana vamos a ver el **cómo**: **cómo los algoritmos aprenden automáticamente**.

## ¿Qué vas a aprender esta semana?

Al final de esta semana, vas a poder:

- **Entender intuitivamente** qué es una derivada y por qué es tan poderosa
- **Calcular gradientes** de funciones de múltiples variables
- **Implementar gradient descent desde cero** en Python
- **Visualizar** cómo los algoritmos "aprenden" navegando funciones de costo
- **Optimizar** modelos de machine learning usando estos conceptos

Y lo más importante: vas a **ver** el machine learning de una forma completamente nueva, entendiendo el motor matemático que lo impulsa.

---

## Parte 1: El Concepto Fundamental - ¿Qué es una Derivada?

### La Intuición Física: Velocidad y Cambio

Antes de cualquier fórmula, pensemos en términos físicos. Si estás manejando y querés saber qué tan rápido estás acelerando, necesitás medir **cómo cambia tu velocidad con respecto al tiempo**.

- Si tu velocidad pasa de 60 km/h a 80 km/h en 2 segundos, tu aceleración promedio es: $\frac{80-60}{2-0} = 10$ km/h por segundo.
- Pero eso es el cambio **promedio**. ¿Qué pasa si querés saber la aceleración **exacta** en un momento específico?

**Esa es exactamente la pregunta que responde la derivada.**

### Definición Matemática (Sin Dolor)

Para cualquier función $f(x)$, la derivada en un punto $x$ nos dice:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Traducido al español**: "¿Qué tan rápido cambia $f(x)$ cuando $x$ cambia una cantidad infinitesimalmente pequeña?"

Pero miremos esto visualmente:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

def visualizar_derivada_concepto():
    """
    Visualización interactiva del concepto de derivada
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Función ejemplo: f(x) = x^2
    x = np.linspace(-3, 3, 1000)
    y = x**2

    # Punto donde calculamos la derivada
    x_punto = 1
    y_punto = x_punto**2

    # Diferentes valores de h (incremento)
    h_values = [1, 0.5, 0.1, 0.01]
    colors = ['red', 'orange', 'blue', 'green']

    # Gráfico principal
    ax1.plot(x, y, 'k-', linewidth=2, label='f(x) = x²')
    ax1.plot(x_punto, y_punto, 'ro', markersize=8, label=f'Punto ({x_punto}, {y_punto})')

    # Mostrar diferentes aproximaciones a la derivada
    for i, (h, color) in enumerate(zip(h_values, colors)):
        x_h = x_punto + h
        y_h = x_h**2

        # Línea secante
        slope = (y_h - y_punto) / h
        x_line = np.array([x_punto - 0.5, x_punto + h + 0.5])
        y_line = y_punto + slope * (x_line - x_punto)

        ax1.plot(x_line, y_line, color=color, linestyle='--', alpha=0.7,
                label=f'h={h}, pendiente≈{slope:.2f}')
        ax1.plot(x_h, y_h, 'o', color=color, markersize=6)

    # Derivada exacta (pendiente verdadera)
    derivada_exacta = 2 * x_punto  # Para f(x) = x², f'(x) = 2x
    x_tangente = np.array([x_punto - 1, x_punto + 1])
    y_tangente = y_punto + derivada_exacta * (x_tangente - x_punto)
    ax1.plot(x_tangente, y_tangente, 'purple', linewidth=3,
            label=f'Tangente (derivada exacta = {derivada_exacta})')

    ax1.set_xlim(-2, 3)
    ax1.set_ylim(-1, 5)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Concepto Visual de la Derivada')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')

    # Gráfico de convergencia
    h_range = np.logspace(-5, 0, 100)
    aproximaciones = [(((x_punto + h)**2) - y_punto) / h for h in h_range]

    ax2.semilogx(h_range, aproximaciones, 'b-', linewidth=2)
    ax2.axhline(y=derivada_exacta, color='purple', linestyle='--', linewidth=2,
               label=f'Derivada exacta = {derivada_exacta}')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Convergencia hacia la Derivada Exacta')
    ax2.set_xlabel('h (incremento)')
    ax2.set_ylabel('Aproximación de la derivada')
    ax2.legend()

    plt.tight_layout()
    plt.show()

visualizar_derivada_concepto()
```

{{< alert "circle-info" >}}
**Insight clave**: Cuando $h$ se hace cada vez más pequeño, la pendiente de la línea secante se acerca cada vez más a la pendiente de la línea tangente. **Esa pendiente de la tangente ES la derivada.**
{{< /alert >}}

### ¿Por qué esto importa en Machine Learning?

En ML, siempre estamos tratando de **minimizar errores**. Imagínate que tenés una función que mide qué tan "mal" está tu modelo:

```python
def funcion_de_costo_simple(parametro):
    """
    Función de costo hipotética. En la vida real, esto podría ser
    el error cuadrático medio de un modelo de regresión.
    """
    return (parametro - 3)**2 + 2

# Si sabemos que la derivada es 2*(parametro - 3),
# podemos encontrar hacia dónde "mover" el parámetro para reducir el error
def derivada_costo(parametro):
    return 2 * (parametro - 3)

# Ejemplo: si nuestro parámetro actual es 5
parametro_actual = 5
print(f"Costo actual: {funcion_de_costo_simple(parametro_actual)}")
print(f"Derivada: {derivada_costo(parametro_actual)}")
print(f"Derivada > 0 significa: mover el parámetro hacia la IZQUIERDA reduce el costo")
```

**La derivada nos dice exactamente en qué dirección cambiar nuestros parámetros para mejorar el modelo.**

---

## Parte 2: Las Reglas del Juego - Derivación Básica

### Reglas Fundamentales

Como programador, ya sabés que hay patrones y reglas que se repiten. En cálculo diferencial también:

#### 1. Regla de la Potencia
Si $f(x) = x^n$, entonces $f'(x) = n \cdot x^{n-1}$

```python
def derivada_potencia(n, x):
    """Derivada de x^n"""
    if n == 0:
        return 0  # Derivada de constante
    return n * (x ** (n - 1))

# Ejemplos
print(f"Derivada de x² en x=3: {derivada_potencia(2, 3)}")  # 2*3 = 6
print(f"Derivada de x³ en x=2: {derivada_potencia(3, 2)}")  # 3*4 = 12
```

#### 2. Regla de la Suma
Si $f(x) = g(x) + h(x)$, entonces $f'(x) = g'(x) + h'(x)$

**Esto es súper importante**: la derivada es **lineal**. Podés derivar cada término por separado.

#### 3. Regla del Producto
Si $f(x) = g(x) \cdot h(x)$, entonces $f'(x) = g'(x) \cdot h(x) + g(x) \cdot h'(x)$

#### 4. Regla de la Cadena (La Más Importante para ML)
Si $f(x) = g(h(x))$, entonces $f'(x) = g'(h(x)) \cdot h'(x)$

**Esta es la regla que hace posible entrenar redes neuronales profundas** (backpropagation es básicamente aplicar la regla de la cadena repetidamente).

### Derivadas de Funciones Comunes

Estas aparecen constantemente en ML:

| Función | Derivada | Uso en ML |
|---------|----------|-----------|
| $x^n$ | $n \cdot x^{n-1}$ | Regresión polinomial |
| $e^x$ | $e^x$ | Función exponencial |
| $\ln(x)$ | $\frac{1}{x}$ | Función logarítmica |
| $\sin(x)$ | $\cos(x)$ | Análisis de series temporales |
| $\frac{1}{1+e^{-x}}$ | $\frac{e^{-x}}{(1+e^{-x})^2}$ | **Función sigmoid** |

```python
import numpy as np
import matplotlib.pyplot as plt

def mostrar_funciones_y_derivadas():
    """
    Visualización de funciones comunes y sus derivadas
    """
    x = np.linspace(-5, 5, 1000)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # 1. Función cuadrática
    y1 = x**2
    dy1 = 2*x
    axes[0].plot(x, y1, 'b-', label='f(x) = x²', linewidth=2)
    axes[0].plot(x, dy1, 'r--', label="f'(x) = 2x", linewidth=2)
    axes[0].set_title('Función Cuadrática')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Función exponencial
    y2 = np.exp(x)
    dy2 = np.exp(x)  # Su propia derivada!
    axes[1].plot(x, y2, 'b-', label='f(x) = eˣ', linewidth=2)
    axes[1].plot(x, dy2, 'r--', label="f'(x) = eˣ", linewidth=2)
    axes[1].set_title('Función Exponencial')
    axes[1].set_ylim(0, 10)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Función sigmoid (súper importante en ML)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivada(x):
        s = sigmoid(x)
        return s * (1 - s)

    y3 = sigmoid(x)
    dy3 = sigmoid_derivada(x)
    axes[2].plot(x, y3, 'b-', label='σ(x) = 1/(1+e⁻ˣ)', linewidth=2)
    axes[2].plot(x, dy3, 'r--', label="σ'(x) = σ(x)(1-σ(x))", linewidth=2)
    axes[2].set_title('Función Sigmoid (Activación Neural)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # 4. Función logarítmica
    x_pos = x[x > 0.1]  # Evitar log de números negativos
    y4 = np.log(x_pos)
    dy4 = 1/x_pos
    axes[3].plot(x_pos, y4, 'b-', label='f(x) = ln(x)', linewidth=2)
    axes[3].plot(x_pos, dy4, 'r--', label="f'(x) = 1/x", linewidth=2)
    axes[3].set_title('Función Logarítmica')
    axes[3].set_xlim(0.1, 5)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

mostrar_funciones_y_derivadas()
```

### Implementación Práctica: Calculadora de Derivadas

Vamos a implementar un calculador de derivadas numérico y compararlo con el analítico:

```python
def derivada_numerica(func, x, h=1e-7):
    """
    Calcula la derivada numérica usando la definición de límite
    """
    return (func(x + h) - func(x)) / h

def comparar_derivadas():
    """
    Compara derivadas numéricas vs analíticas
    """
    # Función test: f(x) = x³ + 2x² - 5x + 1
    def f(x):
        return x**3 + 2*x**2 - 5*x + 1

    def f_derivada_analitica(x):
        # f'(x) = 3x² + 4x - 5
        return 3*x**2 + 4*x - 5

    puntos = np.linspace(-3, 3, 10)

    print("Comparación Derivadas Numéricas vs Analíticas:")
    print("-" * 60)
    print(f"{'x':>8} {'Numérica':>12} {'Analítica':>12} {'Error':>12}")
    print("-" * 60)

    for x in puntos:
        numerica = derivada_numerica(f, x)
        analitica = f_derivada_analitica(x)
        error = abs(numerica - analitica)

        print(f"{x:>8.2f} {numerica:>12.6f} {analitica:>12.6f} {error:>12.2e}")

    # Visualización
    x_plot = np.linspace(-3, 3, 1000)
    y_original = [f(x) for x in x_plot]
    y_derivada_num = [derivada_numerica(f, x) for x in x_plot]
    y_derivada_ana = [f_derivada_analitica(x) for x in x_plot]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.plot(x_plot, y_original, 'b-', linewidth=2, label='f(x) = x³ + 2x² - 5x + 1')
    ax1.set_title('Función Original')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x_plot, y_derivada_ana, 'g-', linewidth=2, label="Analítica: f'(x) = 3x² + 4x - 5")
    ax2.plot(x_plot, y_derivada_num, 'r--', linewidth=2, alpha=0.7, label='Numérica')
    ax2.set_title('Comparación de Derivadas')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

comparar_derivadas()
```

{{< alert "lightbulb" >}}
**Insight práctico**: En machine learning, a menudo usamos derivadas numéricas para verificar que nuestras implementaciones de gradientes analíticos están correctas. Esto se llama "gradient checking".
{{< /alert >}}

---

## Parte 3: El Mundo Multidimensional - Derivadas Parciales

### El Problema: Funciones de Múltiples Variables

En machine learning, rara vez tenemos funciones de una sola variable. Un modelo típico podría tener millones de parámetros:

```python
# En regresión lineal simple: y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
# Necesitamos optimizar TODOS los parámetros: w₁, w₂, ..., wₙ, b
```

¿Cómo calculamos "la derivada" de una función que depende de múltiples variables?

### Derivadas Parciales: Una Variable a la Vez

**Idea clave**: Mantén todas las variables constantes excepto una, y deriva con respecto a esa variable.

Para $f(x, y) = x^2 + 3xy + y^2$:

- $\frac{\partial f}{\partial x} = 2x + 3y$ (tratando $y$ como constante)
- $\frac{\partial f}{\partial y} = 3x + 2y$ (tratando $x$ como constante)

```python
def visualizar_derivadas_parciales():
    """
    Visualización 3D de derivadas parciales
    """
    fig = plt.figure(figsize=(18, 6))

    # Crear grilla de puntos
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)

    # Función: f(x,y) = x² + 3xy + y²
    Z = X**2 + 3*X*Y + Y**2

    # Derivadas parciales
    dZ_dx = 2*X + 3*Y  # ∂f/∂x
    dZ_dy = 3*X + 2*Y  # ∂f/∂y

    # Gráfico 1: Función original
    ax1 = fig.add_subplot(131, projection='3d')
    surface = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_title('f(x,y) = x² + 3xy + y²')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')

    # Gráfico 2: Derivada parcial respecto a x
    ax2 = fig.add_subplot(132, projection='3d')
    surface2 = ax2.plot_surface(X, Y, dZ_dx, cmap='Reds', alpha=0.8)
    ax2.set_title('∂f/∂x = 2x + 3y')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('∂f/∂x')

    # Gráfico 3: Derivada parcial respecto a y
    ax3 = fig.add_subplot(133, projection='3d')
    surface3 = ax3.plot_surface(X, Y, dZ_dy, cmap='Blues', alpha=0.8)
    ax3.set_title('∂f/∂y = 3x + 2y')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('∂f/∂y')

    plt.tight_layout()
    plt.show()

visualizar_derivadas_parciales()
```

### El Vector Gradiente: La Clave de Todo

El **gradiente** es simplemente el vector que contiene todas las derivadas parciales:

$$\nabla f(x, y) = \begin{pmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{pmatrix}$$

**¿Por qué es tan importante?** Porque el gradiente **siempre apunta en la dirección de máximo crecimiento** de la función.

```python
def visualizar_gradiente_direccion():
    """
    Visualización del gradiente como dirección de máximo crecimiento
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Función simple para visualizar
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2  # Paraboloide simple

    # Gradiente: ∇f = [2x, 2y]
    dZ_dx = 2*X
    dZ_dy = 2*Y

    # Gráfico 1: Función con curvas de nivel
    contour = ax1.contour(X, Y, Z, levels=20, colors='gray', alpha=0.5)
    ax1.clabel(contour, inline=True, fontsize=8)

    # Mostrar vectores gradiente en algunos puntos
    puntos_x = [-2, -1, 0, 1, 2]
    puntos_y = [-2, -1, 0, 1, 2]

    for px in puntos_x[::2]:  # Solo algunos puntos para claridad
        for py in puntos_y[::2]:
            if px != 0 or py != 0:  # Evitar el origen donde el gradiente es cero
                grad_x = 2*px
                grad_y = 2*py
                # Normalizar para mejor visualización
                norm = np.sqrt(grad_x**2 + grad_y**2)
                grad_x, grad_y = grad_x/norm * 0.3, grad_y/norm * 0.3

                ax1.arrow(px, py, grad_x, grad_y,
                         head_width=0.1, head_length=0.1,
                         fc='red', ec='red', alpha=0.8)

    ax1.set_title('Gradiente apunta hacia máximo crecimiento')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Gráfico 2: Magnitud del gradiente
    grad_magnitude = np.sqrt(dZ_dx**2 + dZ_dy**2)
    im = ax2.contourf(X, Y, grad_magnitude, levels=20, cmap='hot')
    plt.colorbar(im, ax=ax2)
    ax2.set_title('Magnitud del Gradiente ||∇f||')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    plt.tight_layout()
    plt.show()

visualizar_gradiente_direccion()
```

{{< alert "circle-info" >}}
**Insight crucial**: Si querés **minimizar** una función, necesitás moverte en la dirección **opuesta** al gradiente. Eso es exactamente lo que hace gradient descent: **-∇f**.
{{< /alert >}}

### Implementación de Gradientes Numéricos

```python
def gradiente_numerico(func, punto, h=1e-7):
    """
    Calcula el gradiente numérico de una función en un punto

    Args:
        func: función que toma un vector y devuelve un escalar
        punto: punto donde calcular el gradiente (array de numpy)
        h: paso para la aproximación numérica

    Returns:
        gradiente: vector gradiente en el punto
    """
    punto = np.array(punto, dtype=float)
    grad = np.zeros_like(punto)

    for i in range(len(punto)):
        # Vector de perturbación
        delta = np.zeros_like(punto)
        delta[i] = h

        # Aproximación de derivada parcial
        grad[i] = (func(punto + delta) - func(punto - delta)) / (2 * h)

    return grad

# Ejemplo de uso
def funcion_ejemplo(params):
    """f(x,y) = x² + 3xy + y²"""
    x, y = params
    return x**2 + 3*x*y + y**2

def gradiente_analitico_ejemplo(params):
    """Gradiente analítico de f(x,y) = x² + 3xy + y²"""
    x, y = params
    return np.array([2*x + 3*y, 3*x + 2*y])

# Verificación
punto_test = [1.5, -0.8]
grad_num = gradiente_numerico(funcion_ejemplo, punto_test)
grad_ana = gradiente_analitico_ejemplo(punto_test)

print("Verificación de Gradientes:")
print(f"Punto: {punto_test}")
print(f"Gradiente numérico:  {grad_num}")
print(f"Gradiente analítico: {grad_ana}")
print(f"Error: {np.linalg.norm(grad_num - grad_ana)}")
```

---

## Parte 4: La Magia de la Optimización

### El Problema Fundamental del Machine Learning

Todo en machine learning se reduce a esto:

1. Definís una **función de costo** $J(\theta)$ que mide qué tan "malo" es tu modelo
2. Encontrás los parámetros $\theta$ que **minimizan** esa función
3. Esos parámetros óptimos son tu "modelo entrenado"

El gradient descent es el algoritmo que resuelve el paso 2.

### Gradient Descent: El Algoritmo Estrella

**Idea súper simple**:
1. Empezás con parámetros aleatorios
2. Calculás el gradiente (¿en qué dirección crece la función?)
3. Te movés en la dirección **opuesta** (para minimizar)
4. Repetís hasta converger

Matemáticamente:

$$\theta_{nuevo} = \theta_{actual} - \alpha \nabla J(\theta_{actual})$$

Donde $\alpha$ es el **learning rate** (qué tan grandes son los pasos).

```python
def gradient_descent_visual():
    """
    Visualización completa del algoritmo gradient descent
    """
    # Función objetivo: f(x,y) = (x-2)² + (y+1)² + 3
    # Mínimo en (2, -1)
    def funcion_objetivo(params):
        x, y = params
        return (x - 2)**2 + (y + 1)**2 + 3

    def gradiente_funcion(params):
        x, y = params
        return np.array([2*(x - 2), 2*(y + 1)])

    # Configuración
    learning_rates = [0.1, 0.3, 0.5, 0.9]
    punto_inicial = [-1, 2]
    max_iteraciones = 20

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Crear superficie para visualización
    x_range = np.linspace(-3, 4, 100)
    y_range = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (X - 2)**2 + (Y + 1)**2 + 3

    for idx, lr in enumerate(learning_rates):
        ax = axes[idx]

        # Contorno de la función
        contour = ax.contour(X, Y, Z, levels=20, colors='gray', alpha=0.5)

        # Gradient descent
        trayectoria_x = [punto_inicial[0]]
        trayectoria_y = [punto_inicial[1]]
        costos = [funcion_objetivo(punto_inicial)]

        punto_actual = np.array(punto_inicial, dtype=float)

        for i in range(max_iteraciones):
            grad = gradiente_funcion(punto_actual)
            punto_actual = punto_actual - lr * grad

            trayectoria_x.append(punto_actual[0])
            trayectoria_y.append(punto_actual[1])
            costos.append(funcion_objetivo(punto_actual))

            # Si converge, parar
            if np.linalg.norm(grad) < 1e-6:
                break

        # Plotear trayectoria
        ax.plot(trayectoria_x, trayectoria_y, 'ro-', markersize=4, linewidth=2,
                label=f'Learning Rate = {lr}')
        ax.plot(punto_inicial[0], punto_inicial[1], 'go', markersize=10,
                label='Inicio')
        ax.plot(2, -1, 'r*', markersize=15, label='Mínimo Global')

        ax.set_title(f'LR = {lr}, Iteraciones: {len(trayectoria_x)-1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-3, 4)
        ax.set_ylim(-3, 3)

    plt.tight_layout()
    plt.show()

    return learning_rates, costos

gradient_descent_visual()
```

### Análisis del Learning Rate

El **learning rate** es crítico. Muy pequeño → converges lento. Muy grande → no converges nunca.

```python
def analisis_learning_rate():
    """
    Análisis detallado del impacto del learning rate
    """
    def f(x):
        return x**2 + 2  # Función simple 1D

    def df_dx(x):
        return 2*x

    learning_rates = [0.1, 0.5, 1.0, 1.5]  # El último es demasiado grande
    x_inicial = 3.0
    iteraciones = 20

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Función objetivo
    x_plot = np.linspace(-4, 4, 1000)
    y_plot = f(x_plot)

    for lr in learning_rates:
        trayectoria = [x_inicial]
        costos = [f(x_inicial)]
        x_actual = x_inicial

        for i in range(iteraciones):
            grad = df_dx(x_actual)
            x_nuevo = x_actual - lr * grad

            # Verificar divergencia
            if abs(x_nuevo) > 100:
                print(f"¡Learning rate {lr} causa divergencia en iteración {i}!")
                break

            trayectoria.append(x_nuevo)
            costos.append(f(x_nuevo))
            x_actual = x_nuevo

        # Gráfico 1: Trayectoria en el espacio de parámetros
        ax1.plot(x_plot, y_plot, 'k-', alpha=0.3)
        if len(trayectoria) < 50:  # Solo plotear si no diverge
            ax1.plot(trayectoria, [f(x) for x in trayectoria], 'o-',
                    label=f'LR = {lr}', markersize=4)

    ax1.axhline(y=2, color='red', linestyle='--', label='Mínimo Global')
    ax1.set_xlabel('Parámetro x')
    ax1.set_ylabel('Costo f(x)')
    ax1.set_title('Trayectorias de Gradient Descent')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(0, 20)

    # Gráfico 2: Convergencia del costo
    for lr in learning_rates:
        costos = [f(x_inicial)]
        x_actual = x_inicial

        for i in range(iteraciones):
            grad = df_dx(x_actual)
            x_actual = x_actual - lr * grad
            if abs(x_actual) > 100:
                break
            costos.append(f(x_actual))

        if len(costos) < 50:
            ax2.semilogy(range(len(costos)), costos, 'o-',
                        label=f'LR = {lr}', markersize=4)

    ax2.axhline(y=2, color='red', linestyle='--', label='Mínimo Global')
    ax2.set_xlabel('Iteración')
    ax2.set_ylabel('Costo (escala log)')
    ax2.set_title('Convergencia del Costo')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

analisis_learning_rate()
```

{{< alert "warning" >}}
**Reglas prácticas para Learning Rate**:
- Empezá con 0.01 o 0.1
- Si el costo aumenta → LR muy alto, reducí por 10
- Si converge muy lento → LR muy bajo, aumentá por 2-5
- Usá técnicas como learning rate scheduling o algoritmos adaptativos
{{< /alert >}}

### Funciones de Costo en Machine Learning

Veamos las funciones que realmente optimizamos en ML:

```python
def funciones_costo_ml():
    """
    Visualización de funciones de costo comunes en ML
    """
    # Datos sintéticos para ejemplos
    np.random.seed(42)
    n_puntos = 100
    x_true = np.linspace(0, 1, n_puntos)
    y_true = 2 * x_true + 1 + 0.1 * np.random.randn(n_puntos)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Error Cuadrático Medio (MSE) - Regresión
    def mse(w, b):
        y_pred = w * x_true + b
        return np.mean((y_true - y_pred)**2)

    w_range = np.linspace(0, 4, 50)
    b_range = np.linspace(-1, 3, 50)
    W, B = np.meshgrid(w_range, b_range)
    MSE_surface = np.array([[mse(w, b) for w in w_range] for b in b_range])

    contour1 = axes[0,0].contour(W, B, MSE_surface, levels=20)
    axes[0,0].clabel(contour1, inline=True, fontsize=8)
    axes[0,0].plot(2, 1, 'r*', markersize=15, label='Mínimo verdadero')
    axes[0,0].set_title('MSE para Regresión Lineal')
    axes[0,0].set_xlabel('Peso (w)')
    axes[0,0].set_ylabel('Sesgo (b)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # 2. Función de pérdida logística - Clasificación
    # Simulamos datos de clasificación
    y_class = (y_true > np.mean(y_true)).astype(int)

    def logistic_loss(w, b):
        z = w * x_true + b
        # Evitar overflow numérico
        z = np.clip(z, -500, 500)
        return np.mean(np.log(1 + np.exp(-y_class * z + (1-y_class) * z)))

    # Esta función es más compleja de visualizar, usamos una aproximación
    w_range2 = np.linspace(-1, 5, 30)
    b_range2 = np.linspace(-2, 4, 30)
    W2, B2 = np.meshgrid(w_range2, b_range2)

    Logistic_surface = np.zeros_like(W2)
    for i in range(len(b_range2)):
        for j in range(len(w_range2)):
            try:
                Logistic_surface[i,j] = logistic_loss(W2[i,j], B2[i,j])
            except:
                Logistic_surface[i,j] = np.inf

    # Reemplazar infinitos con valores grandes
    Logistic_surface = np.where(np.isinf(Logistic_surface),
                               np.max(Logistic_surface[np.isfinite(Logistic_surface)]) * 2,
                               Logistic_surface)

    contour2 = axes[0,1].contour(W2, B2, Logistic_surface, levels=20)
    axes[0,1].set_title('Pérdida Logística para Clasificación')
    axes[0,1].set_xlabel('Peso (w)')
    axes[0,1].set_ylabel('Sesgo (b)')
    axes[0,1].grid(True, alpha=0.3)

    # 3. Función de activación y su derivada
    z = np.linspace(-6, 6, 1000)
    sigmoid = 1 / (1 + np.exp(-z))
    sigmoid_deriv = sigmoid * (1 - sigmoid)

    axes[1,0].plot(z, sigmoid, 'b-', linewidth=2, label='σ(z) = 1/(1+e^-z)')
    axes[1,0].plot(z, sigmoid_deriv, 'r--', linewidth=2, label="σ'(z) = σ(z)(1-σ(z))")
    axes[1,0].set_title('Función Sigmoid y su Derivada')
    axes[1,0].set_xlabel('z')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # 4. Comparación de optimizadores
    def optimizar_comparacion():
        # Función objetivo complicada con múltiples mínimos locales
        def f_compleja(x, y):
            return 0.5 * (x**2 + y**2) + 0.3 * np.sin(5*x) + 0.3 * np.cos(5*y)

        def grad_f_compleja(x, y):
            dx = x + 1.5 * np.cos(5*x)
            dy = y - 1.5 * np.sin(5*y)
            return np.array([dx, dy])

        # SGD clásico
        punto = np.array([2.0, 2.0])
        lr = 0.1
        trayectoria_sgd = [punto.copy()]

        for _ in range(100):
            grad = grad_f_compleja(punto[0], punto[1])
            punto = punto - lr * grad
            trayectoria_sgd.append(punto.copy())

        # SGD con momentum
        punto_mom = np.array([2.0, 2.0])
        velocidad = np.zeros(2)
        momentum = 0.9
        trayectoria_momentum = [punto_mom.copy()]

        for _ in range(100):
            grad = grad_f_compleja(punto_mom[0], punto_mom[1])
            velocidad = momentum * velocidad - lr * grad
            punto_mom = punto_mom + velocidad
            trayectoria_momentum.append(punto_mom.copy())

        # Visualizar
        x_range = np.linspace(-3, 3, 100)
        y_range = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = f_compleja(X, Y)

        axes[1,1].contour(X, Y, Z, levels=30, colors='gray', alpha=0.5)

        traj_sgd = np.array(trayectoria_sgd)
        traj_mom = np.array(trayectoria_momentum)

        axes[1,1].plot(traj_sgd[:, 0], traj_sgd[:, 1], 'b-',
                      label='SGD clásico', linewidth=2, alpha=0.7)
        axes[1,1].plot(traj_mom[:, 0], traj_mom[:, 1], 'r-',
                      label='SGD + Momentum', linewidth=2, alpha=0.7)
        axes[1,1].plot(2, 2, 'go', markersize=8, label='Inicio')

        axes[1,1].set_title('Comparación de Optimizadores')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_xlim(-3, 3)
        axes[1,1].set_ylim(-3, 3)

    optimizar_comparacion()
    plt.tight_layout()
    plt.show()

funciones_costo_ml()
```

---

## Parte 5: Implementación desde Cero

### Gradient Descent para Regresión Lineal

Vamos a implementar gradient descent completamente desde cero para resolver un problema real:

```python
class GradientDescentLinear:
    """
    Implementación completa de Gradient Descent para Regresión Lineal
    """

    def __init__(self, learning_rate=0.01, max_iter=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.history = {'cost': [], 'weights': [], 'bias': []}

    def _add_intercept(self, X):
        """Agregar columna de 1s para el término de sesgo"""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def _cost_function(self, h, y):
        """Calcular función de costo MSE"""
        return (1 / (2 * len(y))) * np.sum((h - y) ** 2)

    def _gradient(self, X, h, y):
        """Calcular gradiente de la función de costo"""
        return (1 / len(y)) * X.T.dot(h - y)

    def fit(self, X, y, verbose=True):
        """
        Entrenar el modelo usando gradient descent
        """
        # Agregar término de intercept
        X = self._add_intercept(X)

        # Inicializar parámetros aleatoriamente
        self.theta = np.random.normal(0, 0.01, X.shape[1])

        if verbose:
            print(f"Iniciando entrenamiento con {len(X)} muestras...")
            print(f"Learning rate: {self.learning_rate}")
            print("-" * 50)

        for i in range(self.max_iter):
            # Forward pass: calcular predicciones
            h = X.dot(self.theta)

            # Calcular costo
            cost = self._cost_function(h, y)

            # Calcular gradiente
            gradient = self._gradient(X, h, y)

            # Actualizar parámetros
            theta_anterior = self.theta.copy()
            self.theta -= self.learning_rate * gradient

            # Guardar historial
            self.history['cost'].append(cost)
            self.history['weights'].append(self.theta[1:].copy())
            self.history['bias'].append(self.theta[0])

            # Verificar convergencia
            if np.linalg.norm(self.theta - theta_anterior) < self.tolerance:
                if verbose:
                    print(f"Convergencia alcanzada en iteración {i}")
                break

            if verbose and i % 100 == 0:
                print(f"Iteración {i:4d}: Costo = {cost:.6f}")

        if verbose:
            print(f"Entrenamiento completado!")
            print(f"Parámetros finales: {self.theta}")
            print(f"Costo final: {self.history['cost'][-1]:.6f}")

        return self

    def predict(self, X):
        """Hacer predicciones"""
        X = self._add_intercept(X)
        return X.dot(self.theta)

    def plot_training_history(self):
        """Visualizar el proceso de entrenamiento"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        iterations = range(len(self.history['cost']))

        # 1. Convergencia del costo
        ax1.plot(iterations, self.history['cost'], 'b-', linewidth=2)
        ax1.set_title('Convergencia de la Función de Costo')
        ax1.set_xlabel('Iteración')
        ax1.set_ylabel('MSE')
        ax1.grid(True, alpha=0.3)
        ax1.semilogy()  # Escala logarítmica para mejor visualización

        # 2. Evolución de los pesos
        weights_history = np.array(self.history['weights'])
        if weights_history.shape[1] <= 5:  # Solo si no hay demasiados features
            for i in range(weights_history.shape[1]):
                ax2.plot(iterations, weights_history[:, i],
                        label=f'Peso {i+1}', linewidth=2)
        ax2.set_title('Evolución de los Pesos')
        ax2.set_xlabel('Iteración')
        ax2.set_ylabel('Valor del Peso')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Evolución del sesgo
        ax3.plot(iterations, self.history['bias'], 'g-', linewidth=2)
        ax3.set_title('Evolución del Sesgo (Intercept)')
        ax3.set_xlabel('Iteración')
        ax3.set_ylabel('Valor del Sesgo')
        ax3.grid(True, alpha=0.3)

        # 4. Gradiente de la función de costo (aproximado por cambio en parámetros)
        cambios_theta = [0] + [np.linalg.norm(np.array(self.history['weights'][i]) -
                                             np.array(self.history['weights'][i-1]))
                              for i in range(1, len(self.history['weights']))]
        ax4.semilogy(iterations, cambios_theta, 'r-', linewidth=2)
        ax4.set_title('Magnitud del Cambio en Parámetros')
        ax4.set_xlabel('Iteración')
        ax4.set_ylabel('||Δθ|| (escala log)')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# Ejemplo de uso completo
def demo_gradient_descent():
    """
    Demostración completa del algoritmo
    """
    print("🚀 DEMO: Gradient Descent para Regresión Lineal")
    print("=" * 60)

    # 1. Generar datos sintéticos
    np.random.seed(42)
    n_samples = 200
    n_features = 2

    X = np.random.randn(n_samples, n_features)
    true_weights = np.array([3, -2])
    true_bias = 1.5
    noise = 0.1 * np.random.randn(n_samples)

    y = X.dot(true_weights) + true_bias + noise

    print(f"Datos generados:")
    print(f"  - {n_samples} muestras, {n_features} características")
    print(f"  - Pesos verdaderos: {true_weights}")
    print(f"  - Sesgo verdadero: {true_bias}")
    print()

    # 2. Entrenar modelo
    modelo = GradientDescentLinear(learning_rate=0.1, max_iter=2000)
    modelo.fit(X, y, verbose=True)

    print()
    print("📊 RESULTADOS:")
    print("-" * 30)
    print(f"Pesos estimados: {modelo.theta[1:]}")
    print(f"Sesgo estimado: {modelo.theta[0]:.4f}")
    print(f"Error en pesos: {np.abs(modelo.theta[1:] - true_weights)}")
    print(f"Error en sesgo: {abs(modelo.theta[0] - true_bias):.4f}")

    # 3. Evaluar modelo
    y_pred = modelo.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

    print()
    print("📈 MÉTRICAS:")
    print("-" * 20)
    print(f"MSE: {mse:.4f}")
    print(f"R²:  {r2:.4f}")

    # 4. Visualizaciones
    modelo.plot_training_history()

    # 5. Comparación visual (solo para 1D)
    if n_features == 1:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(X, y, alpha=0.6, label='Datos reales')
        plt.plot(X, y_pred, 'r-', linewidth=2, label='Predicciones')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.title('Ajuste del Modelo')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        residuos = y - y_pred
        plt.scatter(y_pred, residuos, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicciones')
        plt.ylabel('Residuos')
        plt.title('Análisis de Residuos')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return modelo, X, y

# Ejecutar demo
modelo_entrenado, X_data, y_data = demo_gradient_descent()
```

### Comparación con Solución Analítica

Para verificar que nuestro gradient descent funciona, comparémoslo con la solución analítica de regresión lineal:

```python
def comparar_con_solucion_analitica():
    """
    Compara gradient descent con la solución de forma cerrada
    """
    print("🔍 COMPARACIÓN: Gradient Descent vs Solución Analítica")
    print("=" * 70)

    # Usar los mismos datos del ejemplo anterior
    X = X_data
    y = y_data

    # 1. Solución analítica (Normal Equation)
    # θ = (X^T X)^(-1) X^T y
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    theta_analitica = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

    print("Solución Analítica:")
    print(f"  Sesgo: {theta_analitica[0]:.6f}")
    print(f"  Pesos: {theta_analitica[1:]}")

    print()
    print("Gradient Descent:")
    print(f"  Sesgo: {modelo_entrenado.theta[0]:.6f}")
    print(f"  Pesos: {modelo_entrenado.theta[1:]}")

    print()
    print("Diferencias:")
    diferencias = np.abs(modelo_entrenado.theta - theta_analitica)
    print(f"  Sesgo: {diferencias[0]:.8f}")
    print(f"  Pesos: {diferencias[1:]}")
    print(f"  Error total: {np.linalg.norm(diferencias):.8f}")

    # Verificar predicciones
    pred_analitica = X_with_intercept @ theta_analitica
    pred_gd = modelo_entrenado.predict(X)

    mse_analitica = np.mean((y - pred_analitica) ** 2)
    mse_gd = np.mean((y - pred_gd) ** 2)

    print()
    print("Comparación de MSE:")
    print(f"  Solución Analítica: {mse_analitica:.8f}")
    print(f"  Gradient Descent:   {mse_gd:.8f}")
    print(f"  Diferencia:         {abs(mse_analitica - mse_gd):.10f}")

comparar_con_solucion_analitica()
```

### Visualización 3D del Landscape de Optimización

Para realmente entender cómo funciona gradient descent, veamos el "paisaje" que está navegando:

```python
def visualizar_landscape_3d():
    """
    Visualización 3D del proceso de optimización
    """
    # Usar un ejemplo 2D simple para poder visualizar el landscape completo
    # Datos sintéticos 1D
    np.random.seed(42)
    X_simple = np.linspace(0, 1, 20).reshape(-1, 1)
    y_simple = 2 * X_simple.flatten() + 1 + 0.1 * np.random.randn(20)

    # Función de costo en función de w (peso) y b (sesgo)
    def cost_surface(w, b):
        y_pred = w * X_simple.flatten() + b
        return np.mean((y_simple - y_pred) ** 2)

    # Crear grid para la superficie
    w_range = np.linspace(-1, 5, 50)
    b_range = np.linspace(-1, 4, 50)
    W, B = np.meshgrid(w_range, b_range)

    Cost = np.array([[cost_surface(w, b) for w in w_range] for b in b_range])

    # Ejecutar gradient descent y guardar trayectoria
    modelo_3d = GradientDescentLinear(learning_rate=0.1, max_iter=100)
    modelo_3d.fit(X_simple, y_simple, verbose=False)

    # Extraer trayectoria
    w_traj = [w[0] if len(w) > 0 else 0 for w in modelo_3d.history['weights']]
    b_traj = modelo_3d.history['bias']
    cost_traj = modelo_3d.history['cost']

    # Crear visualización 3D
    fig = plt.figure(figsize=(20, 6))

    # 1. Vista 3D del landscape
    ax1 = fig.add_subplot(131, projection='3d')
    surface = ax1.plot_surface(W, B, Cost, cmap='viridis', alpha=0.6)

    # Plotear trayectoria de gradient descent
    ax1.plot(w_traj, b_traj, cost_traj, 'ro-', linewidth=3, markersize=4,
             label='Trayectoria GD')
    ax1.plot(w_traj[0], b_traj[0], cost_traj[0], 'go', markersize=10,
             label='Inicio')
    ax1.plot(w_traj[-1], b_traj[-1], cost_traj[-1], 'r*', markersize=15,
             label='Final')

    ax1.set_xlabel('Peso (w)')
    ax1.set_ylabel('Sesgo (b)')
    ax1.set_zlabel('Costo MSE')
    ax1.set_title('Landscape 3D de Optimización')
    ax1.legend()

    # 2. Vista contorno desde arriba
    ax2 = fig.add_subplot(132)
    contour = ax2.contour(W, B, Cost, levels=30, colors='gray', alpha=0.6)
    ax2.clabel(contour, inline=True, fontsize=8)

    ax2.plot(w_traj, b_traj, 'ro-', linewidth=3, markersize=4)
    ax2.plot(w_traj[0], b_traj[0], 'go', markersize=10, label='Inicio')
    ax2.plot(w_traj[-1], b_traj[-1], 'r*', markersize=15, label='Final')

    # Agregar vectores de gradiente en algunos puntos
    for i in range(0, len(w_traj), 10):
        if i < len(w_traj) - 1:
            dw = w_traj[i+1] - w_traj[i]
            db = b_traj[i+1] - b_traj[i]
            ax2.arrow(w_traj[i], b_traj[i], dw*10, db*10,
                     head_width=0.05, head_length=0.05, fc='blue', ec='blue')

    ax2.set_xlabel('Peso (w)')
    ax2.set_ylabel('Sesgo (b)')
    ax2.set_title('Vista de Contorno + Gradientes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Convergencia del costo
    ax3 = fig.add_subplot(133)
    ax3.plot(range(len(cost_traj)), cost_traj, 'b-', linewidth=2)
    ax3.set_xlabel('Iteración')
    ax3.set_ylabel('Costo MSE')
    ax3.set_title('Convergencia del Costo')
    ax3.grid(True, alpha=0.3)
    ax3.semilogy()  # Escala logarítmica

    plt.tight_layout()
    plt.show()

    print(f"Punto inicial: w={w_traj[0]:.3f}, b={b_traj[0]:.3f}")
    print(f"Punto final:   w={w_traj[-1]:.3f}, b={b_traj[-1]:.3f}")
    print(f"Iteraciones:   {len(cost_traj)}")

visualizar_landscape_3d()
```

---

## Parte 6: Proyecto Semanal - Gradient Descent desde Cero

### El Desafío

Tu misión esta semana es implementar y validar un algoritmo completo de gradient descent para regresión lineal multivariable, comparando diferentes configuraciones y analizando su comportamiento.

```python
class GradientDescentProyecto:
    """
    Implementación completa para el proyecto semanal
    Incluye múltiples variantes y análisis exhaustivo
    """

    def __init__(self, learning_rate=0.01, max_iter=1000, tolerance=1e-6,
                 regularization=None, lambda_reg=0.01):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.regularization = regularization  # None, 'l1', 'l2'
        self.lambda_reg = lambda_reg
        self.reset_history()

    def reset_history(self):
        """Reiniciar historial de entrenamiento"""
        self.history = {
            'cost': [],
            'cost_regularized': [],
            'theta': [],
            'gradients': [],
            'gradient_norms': [],
            'learning_rates': []  # Para learning rate adaptativo
        }

    def _add_intercept(self, X):
        """Agregar columna de bias"""
        return np.column_stack([np.ones(X.shape[0]), X])

    def _cost_function(self, X, y, theta, return_regularized=False):
        """
        Función de costo con regularización opcional
        """
        m = len(y)
        h = X.dot(theta)
        cost = (1/(2*m)) * np.sum((h - y)**2)

        # Agregar regularización si está especificada
        reg_cost = 0
        if self.regularization == 'l2':
            # L2 regularization (Ridge)
            reg_cost = self.lambda_reg * np.sum(theta[1:]**2)  # No regularizar bias
        elif self.regularization == 'l1':
            # L1 regularization (Lasso)
            reg_cost = self.lambda_reg * np.sum(np.abs(theta[1:]))

        if return_regularized:
            return cost, cost + reg_cost
        return cost + reg_cost

    def _gradient(self, X, y, theta):
        """
        Calcular gradiente con regularización opcional
        """
        m = len(y)
        h = X.dot(theta)
        gradient = (1/m) * X.T.dot(h - y)

        # Agregar término de regularización al gradiente
        if self.regularization == 'l2':
            # Para L2: agregar λ*θ (excepto bias)
            reg_gradient = np.zeros_like(theta)
            reg_gradient[1:] = self.lambda_reg * theta[1:]
            gradient += reg_gradient
        elif self.regularization == 'l1':
            # Para L1: agregar λ*sign(θ) (excepto bias)
            reg_gradient = np.zeros_like(theta)
            reg_gradient[1:] = self.lambda_reg * np.sign(theta[1:])
            gradient += reg_gradient

        return gradient

    def fit(self, X, y, adaptive_lr=False, verbose=True):
        """
        Entrenar con opciones avanzadas
        """
        X = self._add_intercept(X)
        self.reset_history()

        # Inicialización inteligente de parámetros
        self.theta = np.random.normal(0, 0.01, X.shape[1])

        # Para learning rate adaptativo
        lr_current = self.learning_rate
        cost_anterior = float('inf')
        paciencia_lr = 0

        if verbose:
            print(f"🚀 Iniciando entrenamiento...")
            print(f"Datos: {X.shape[0]} muestras, {X.shape[1]-1} características")
            print(f"Regularización: {self.regularization}")
            print(f"Learning rate adaptativo: {adaptive_lr}")
            print("-" * 60)

        for iteration in range(self.max_iter):
            # Forward pass
            cost_base, cost_reg = self._cost_function(X, y, self.theta, True)

            # Backward pass
            gradient = self._gradient(X, y, self.theta)
            gradient_norm = np.linalg.norm(gradient)

            # Actualización de parámetros
            theta_anterior = self.theta.copy()
            self.theta -= lr_current * gradient

            # Learning rate adaptativo
            if adaptive_lr and iteration > 10:
                if cost_reg > cost_anterior:
                    # Si el costo aumentó, reducir learning rate
                    lr_current *= 0.9
                    paciencia_lr += 1
                    if paciencia_lr > 5:
                        if verbose:
                            print(f"Iteración {iteration}: Reduciendo LR a {lr_current:.6f}")
                        paciencia_lr = 0
                else:
                    # Si el costo disminuyó, aumentar ligeramente el learning rate
                    lr_current = min(lr_current * 1.01, self.learning_rate * 2)

            # Guardar historial
            self.history['cost'].append(cost_base)
            self.history['cost_regularized'].append(cost_reg)
            self.history['theta'].append(self.theta.copy())
            self.history['gradients'].append(gradient.copy())
            self.history['gradient_norms'].append(gradient_norm)
            self.history['learning_rates'].append(lr_current)

            # Verificar convergencia
            if gradient_norm < self.tolerance:
                if verbose:
                    print(f"✅ Convergencia por gradiente en iteración {iteration}")
                break

            if np.linalg.norm(self.theta - theta_anterior) < self.tolerance:
                if verbose:
                    print(f"✅ Convergencia por parámetros en iteración {iteration}")
                break

            cost_anterior = cost_reg

            if verbose and iteration % 200 == 0:
                print(f"Iter {iteration:4d}: Costo = {cost_reg:.6f}, "
                      f"||∇|| = {gradient_norm:.6f}, LR = {lr_current:.6f}")

        self.n_iterations = len(self.history['cost'])

        if verbose:
            print(f"🏁 Entrenamiento completado en {self.n_iterations} iteraciones")
            print(f"Costo final: {self.history['cost_regularized'][-1]:.6f}")
            print(f"Parámetros finales: {self.theta}")

        return self

    def predict(self, X):
        """Predicciones"""
        X = self._add_intercept(X)
        return X.dot(self.theta)

    def score(self, X, y):
        """R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    def plot_comprehensive_analysis(self):
        """
        Análisis visual completo del entrenamiento
        """
        fig = plt.figure(figsize=(20, 16))

        iterations = range(len(self.history['cost']))

        # 1. Convergencia de costos
        ax1 = plt.subplot(3, 3, 1)
        plt.plot(iterations, self.history['cost'], 'b-',
                label='Costo base (MSE)', linewidth=2)
        if self.regularization:
            plt.plot(iterations, self.history['cost_regularized'], 'r-',
                    label='Costo regularizado', linewidth=2)
        plt.yscale('log')
        plt.xlabel('Iteración')
        plt.ylabel('Costo (escala log)')
        plt.title('1. Convergencia del Costo')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Evolución de parámetros
        ax2 = plt.subplot(3, 3, 2)
        theta_history = np.array(self.history['theta'])
        for i in range(min(5, theta_history.shape[1])):  # Max 5 parámetros para claridad
            label = 'Bias' if i == 0 else f'θ_{i}'
            plt.plot(iterations, theta_history[:, i], label=label, linewidth=2)
        plt.xlabel('Iteración')
        plt.ylabel('Valor del Parámetro')
        plt.title('2. Evolución de Parámetros')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. Norma del gradiente
        ax3 = plt.subplot(3, 3, 3)
        plt.plot(iterations, self.history['gradient_norms'], 'g-', linewidth=2)
        plt.yscale('log')
        plt.xlabel('Iteración')
        plt.ylabel('||∇J|| (escala log)')
        plt.title('3. Magnitud del Gradiente')
        plt.grid(True, alpha=0.3)

        # 4. Learning rate adaptativo
        ax4 = plt.subplot(3, 3, 4)
        plt.plot(iterations, self.history['learning_rates'], 'orange', linewidth=2)
        plt.xlabel('Iteración')
        plt.ylabel('Learning Rate')
        plt.title('4. Learning Rate Adaptativo')
        plt.grid(True, alpha=0.3)

        # 5. Distribución de gradientes por componente
        ax5 = plt.subplot(3, 3, 5)
        gradients = np.array(self.history['gradients'])
        # Tomar solo las últimas iteraciones para ver la convergencia
        recent_grads = gradients[-min(100, len(gradients)):]
        for i in range(min(3, gradients.shape[1])):
            label = '∇(Bias)' if i == 0 else f'∇θ_{i}'
            plt.hist(recent_grads[:, i], bins=20, alpha=0.7, label=label)
        plt.xlabel('Valor del Gradiente')
        plt.ylabel('Frecuencia')
        plt.title('5. Distribución de Gradientes (últimas iter.)')
        plt.legend()

        # 6. Análisis de la convergencia (cambios en parámetros)
        ax6 = plt.subplot(3, 3, 6)
        cambios_theta = [0] + [np.linalg.norm(self.history['theta'][i] -
                                             self.history['theta'][i-1])
                               for i in range(1, len(self.history['theta']))]
        plt.plot(iterations, cambios_theta, 'purple', linewidth=2)
        plt.yscale('log')
        plt.xlabel('Iteración')
        plt.ylabel('||Δθ|| (escala log)')
        plt.title('6. Cambio en Parámetros')
        plt.grid(True, alpha=0.3)

        # 7. Trajectory en espacio 2D de parámetros (si hay exactamente 2 parámetros)
        ax7 = plt.subplot(3, 3, 7)
        if theta_history.shape[1] >= 2:
            plt.plot(theta_history[:, 0], theta_history[:, 1], 'b-o',
                    markersize=2, linewidth=1, alpha=0.7)
            plt.plot(theta_history[0, 0], theta_history[0, 1], 'go',
                    markersize=10, label='Inicio')
            plt.plot(theta_history[-1, 0], theta_history[-1, 1], 'ro',
                    markersize=10, label='Final')
            plt.xlabel('θ₀ (Bias)')
            plt.ylabel('θ₁')
            plt.title('7. Trayectoria en Espacio de Parámetros')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'Necesita ≥2 parámetros', ha='center', va='center')
            plt.title('7. Trayectoria (N/A)')
        plt.grid(True, alpha=0.3)

        # 8. Rate de convergencia (log del costo vs iteración)
        ax8 = plt.subplot(3, 3, 8)
        log_cost = np.log(self.history['cost_regularized'])
        if len(log_cost) > 10:
            # Ajustar línea recta a las últimas iteraciones para estimar rate
            x_fit = np.arange(len(log_cost)//2, len(log_cost))
            y_fit = log_cost[len(log_cost)//2:]
            if len(x_fit) > 1:
                slope, intercept = np.polyfit(x_fit, y_fit, 1)
                plt.plot(iterations, log_cost, 'b-', linewidth=2, label='log(Costo)')
                plt.plot(x_fit, slope*x_fit + intercept, 'r--', linewidth=2,
                        label=f'Pendiente ≈ {slope:.4f}')
                plt.legend()
        plt.xlabel('Iteración')
        plt.ylabel('log(Costo)')
        plt.title('8. Rate de Convergencia')
        plt.grid(True, alpha=0.3)

        # 9. Estadísticas finales
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        stats_text = f"""
        ESTADÍSTICAS FINALES

        Iteraciones: {self.n_iterations}
        Costo final: {self.history['cost_regularized'][-1]:.6f}
        ||∇|| final: {self.history['gradient_norms'][-1]:.2e}

        Parámetros finales:
        """

        for i, theta in enumerate(self.theta):
            param_name = 'Bias' if i == 0 else f'θ_{i}'
            stats_text += f"  {param_name}: {theta:.4f}\n"

        if self.regularization:
            stats_text += f"\nRegularización: {self.regularization}"
            stats_text += f"\nλ = {self.lambda_reg}"

        ax9.text(0.1, 0.9, stats_text, transform=ax9.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.show()

# Función para ejecutar experimentos comparativos
def experimentos_comparativos():
    """
    Ejecutar múltiples experimentos para comparar diferentes configuraciones
    """
    print("🧪 EXPERIMENTOS COMPARATIVOS")
    print("=" * 80)

    # Generar dataset más complejo
    np.random.seed(42)
    n_samples = 300
    n_features = 4

    # Features correlacionadas para hacer el problema más interesante
    X = np.random.randn(n_samples, n_features)
    X[:, 1] += 0.5 * X[:, 0]  # Correlación entre features

    true_theta = np.array([0.5, 2.0, -1.5, 0.8, 0.3])  # [bias, w1, w2, w3, w4]
    noise_level = 0.1

    X_with_bias = np.column_stack([np.ones(n_samples), X])
    y = X_with_bias @ true_theta + noise_level * np.random.randn(n_samples)

    print(f"Dataset: {n_samples} muestras, {n_features} características")
    print(f"Parámetros verdaderos: {true_theta}")
    print(f"Nivel de ruido: {noise_level}")
    print()

    # Dividir en train/test
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Experimentos con diferentes configuraciones
    configuraciones = [
        {'name': 'SGD Básico', 'lr': 0.01, 'reg': None},
        {'name': 'SGD LR Alto', 'lr': 0.1, 'reg': None},
        {'name': 'SGD + L2', 'lr': 0.01, 'reg': 'l2', 'lambda': 0.01},
        {'name': 'SGD + L1', 'lr': 0.01, 'reg': 'l1', 'lambda': 0.01},
        {'name': 'SGD Adaptativo', 'lr': 0.05, 'reg': None, 'adaptive': True},
    ]

    resultados = []

    for config in configuraciones:
        print(f"🔄 Ejecutando: {config['name']}")

        modelo = GradientDescentProyecto(
            learning_rate=config['lr'],
            max_iter=2000,
            regularization=config.get('reg'),
            lambda_reg=config.get('lambda', 0.01)
        )

        adaptive_lr = config.get('adaptive', False)
        modelo.fit(X_train, y_train, adaptive_lr=adaptive_lr, verbose=False)

        # Evaluación
        train_score = modelo.score(X_train, y_train)
        test_score = modelo.score(X_test, y_test)

        # Error en parámetros
        param_error = np.linalg.norm(modelo.theta - true_theta)

        resultado = {
            'config': config['name'],
            'train_r2': train_score,
            'test_r2': test_score,
            'param_error': param_error,
            'iterations': modelo.n_iterations,
            'final_cost': modelo.history['cost_regularized'][-1],
            'modelo': modelo
        }

        resultados.append(resultado)
        print(f"  ✅ R² train: {train_score:.4f}, R² test: {test_score:.4f}")

    print()
    print("📊 RESUMEN DE RESULTADOS:")
    print("-" * 80)
    print(f"{'Configuración':<15} {'R² Train':<10} {'R² Test':<10} {'Error Param':<12} {'Iter':<6}")
    print("-" * 80)

    for r in resultados:
        print(f"{r['config']:<15} {r['train_r2']:<10.4f} {r['test_r2']:<10.4f} "
              f"{r['param_error']:<12.4f} {r['iterations']:<6d}")

    # Visualización comparativa
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Comparación de convergencia
    for r in resultados:
        iterations = range(len(r['modelo'].history['cost_regularized']))
        ax1.plot(iterations, r['modelo'].history['cost_regularized'],
                label=r['config'], linewidth=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Iteración')
    ax1.set_ylabel('Costo (escala log)')
    ax1.set_title('Convergencia de Diferentes Configuraciones')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. R² comparación
    configs = [r['config'] for r in resultados]
    train_scores = [r['train_r2'] for r in resultados]
    test_scores = [r['test_r2'] for r in resultados]

    x_pos = np.arange(len(configs))
    ax2.bar(x_pos - 0.2, train_scores, 0.4, label='Train', alpha=0.8)
    ax2.bar(x_pos + 0.2, test_scores, 0.4, label='Test', alpha=0.8)
    ax2.set_xlabel('Configuración')
    ax2.set_ylabel('R²')
    ax2.set_title('Comparación de Performance')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(configs, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Error en parámetros
    param_errors = [r['param_error'] for r in resultados]
    ax3.bar(configs, param_errors, alpha=0.8, color='orange')
    ax3.set_xlabel('Configuración')
    ax3.set_ylabel('||θ_estimado - θ_verdadero||')
    ax3.set_title('Error en Estimación de Parámetros')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    # 4. Número de iteraciones
    iterations_list = [r['iterations'] for r in resultados]
    ax4.bar(configs, iterations_list, alpha=0.8, color='green')
    ax4.set_xlabel('Configuración')
    ax4.set_ylabel('Número de Iteraciones')
    ax4.set_title('Velocidad de Convergencia')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return resultados

# Ejecutar experimentos
resultados_exp = experimentos_comparativos()

# Análisis detallado del mejor modelo
print("\n🏆 ANÁLISIS DETALLADO DEL MEJOR MODELO")
print("=" * 50)

mejor_modelo_idx = max(range(len(resultados_exp)),
                      key=lambda i: resultados_exp[i]['test_r2'])
mejor_modelo = resultados_exp[mejor_modelo_idx]['modelo']
mejor_config = resultados_exp[mejor_modelo_idx]['config']

print(f"Mejor configuración: {mejor_config}")
mejor_modelo.plot_comprehensive_analysis()
```

### Desafíos Adicionales

Para llevar tu comprensión al siguiente nivel, intenta estos desafíos:

```python
def desafios_avanzados():
    """
    Desafíos adicionales para profundizar el entendimiento
    """
    print("🎯 DESAFÍOS AVANZADOS")
    print("=" * 50)

    print("""
    1. 🧮 IMPLEMENTA DIFERENTES OPTIMIZADORES:
       - SGD con Momentum
       - RMSprop (simplificado)
       - Adam (simplificado)

    2. 🔍 GRADIENT CHECKING:
       - Implementa verificación numérica de gradientes
       - Compara con gradientes analíticos
       - Encuentra bugs en implementaciones

    3. 📊 ANÁLISIS DE SENSIBILIDAD:
       - ¿Cómo afecta el ruido en los datos?
       - ¿Qué pasa con datos no lineales?
       - ¿Cómo se comporta con outliers?

    4. 🎛️ HYPERPARAMETER TUNING:
       - Grid search para learning rate óptimo
       - Regularización automática
       - Early stopping

    5. 🔢 BATCH PROCESSING:
       - Mini-batch gradient descent
       - Stochastic gradient descent
       - Comparación de varianza vs velocidad
    """)

# Ejemplo de implementación de SGD con Momentum
class SGDMomentum:
    """
    Implementación de SGD con Momentum como ejemplo
    """

    def __init__(self, learning_rate=0.01, momentum=0.9, max_iter=1000):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.column_stack([np.ones(X.shape[0]), X])
        self.theta = np.random.normal(0, 0.01, X.shape[1])
        self.velocity = np.zeros_like(self.theta)

        self.history = {'cost': [], 'theta': []}

        for i in range(self.max_iter):
            # Forward pass
            h = X.dot(self.theta)
            cost = np.mean((h - y)**2)

            # Backward pass
            gradient = (1/len(y)) * X.T.dot(h - y)

            # Update con momentum
            self.velocity = self.momentum * self.velocity - self.learning_rate * gradient
            self.theta += self.velocity

            self.history['cost'].append(cost)
            self.history['theta'].append(self.theta.copy())

        return self

    def predict(self, X):
        X = np.column_stack([np.ones(X.shape[0]), X])
        return X.dot(self.theta)

# Demostración rápida
def demo_momentum():
    """Demo rápida de SGD con momentum"""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = 2*X[:,0] - X[:,1] + 1 + 0.1*np.random.randn(100)

    # SGD clásico vs SGD con momentum
    sgd_clasico = GradientDescentProyecto(learning_rate=0.1, max_iter=500)
    sgd_momentum = SGDMomentum(learning_rate=0.1, momentum=0.9, max_iter=500)

    sgd_clasico.fit(X, y, verbose=False)
    sgd_momentum.fit(X, y)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(sgd_clasico.history['cost_regularized'], 'b-',
             label='SGD Clásico', linewidth=2)
    plt.plot(sgd_momentum.history['cost'], 'r-',
             label='SGD + Momentum', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Iteración')
    plt.ylabel('Costo (escala log)')
    plt.title('SGD vs SGD + Momentum')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    theta_sgd = np.array(sgd_clasico.history['theta'])
    theta_mom = np.array(sgd_momentum.history['theta'])

    plt.plot(theta_sgd[:, 0], theta_sgd[:, 1], 'b-', label='SGD Clásico', alpha=0.7)
    plt.plot(theta_mom[:, 0], theta_mom[:, 1], 'r-', label='SGD + Momentum', alpha=0.7)
    plt.xlabel('θ₀ (bias)')
    plt.ylabel('θ₁')
    plt.title('Trayectorias en Espacio de Parámetros')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

desafios_avanzados()
demo_momentum()
```

---

## Parte 7: Conexión con el Futuro - ¿Qué Viene Después?

### El Puente hacia Deep Learning

Todo lo que aprendiste esta semana es la **base fundamental** de deep learning:

```python
def preview_deep_learning():
    """
    Vista previa de cómo se conecta con deep learning
    """
    print("🔮 CONEXIÓN CON DEEP LEARNING")
    print("=" * 50)

    print("""
    🧠 REDES NEURONALES:
    - Cada neurona aplica: output = σ(w·x + b)
    - σ es una función de activación (sigmoid, ReLU, etc.)
    - La derivada de σ es crucial para backpropagation

    🔗 BACKPROPAGATION:
    - Es simplemente la regla de la cadena aplicada repetidamente
    - ∂Loss/∂w = ∂Loss/∂output × ∂output/∂w
    - Cada capa propaga el gradiente hacia atrás

    ⚡ OPTIMIZADORES AVANZADOS:
    - Adam: combina momentum + RMSprop
    - AdaGrad: learning rates adaptativos por parámetro
    - Todos usan los mismos principios de gradient descent

    📊 FUNCIONES DE PÉRDIDA:
    - Cross-entropy para clasificación
    - Huber loss para robustez a outliers
    - Todas se optimizan con gradientes
    """)

    # Ejemplo simple de "neurona" artificial
    def neurona_sigmoid(x, w, b):
        z = np.dot(w, x) + b
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Sigmoid

    def derivada_sigmoid(a):
        return a * (1 - a)

    # Visualización de una "mini red neuronal"
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Función de activación y su derivada
    z = np.linspace(-6, 6, 1000)
    sigmoid_vals = 1 / (1 + np.exp(-z))
    sigmoid_deriv_vals = sigmoid_vals * (1 - sigmoid_vals)

    axes[0].plot(z, sigmoid_vals, 'b-', linewidth=2, label='σ(z)')
    axes[0].plot(z, sigmoid_deriv_vals, 'r--', linewidth=2, label="σ'(z)")
    axes[0].set_xlabel('z = wx + b')
    axes[0].set_ylabel('Activación')
    axes[0].set_title('Neurona: Activación y Derivada')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Superficie de pérdida para una neurona simple
    w_range = np.linspace(-3, 3, 50)
    b_range = np.linspace(-3, 3, 50)
    W, B = np.meshgrid(w_range, b_range)

    # Datos de ejemplo para clasificación binaria
    np.random.seed(42)
    X_class = np.random.randn(50, 1)
    y_class = (X_class > 0).astype(float).flatten()

    # Cross-entropy loss para diferentes w, b
    def cross_entropy_loss(w, b):
        predictions = neurona_sigmoid(X_class.flatten(), w, b)
        predictions = np.clip(predictions, 1e-7, 1-1e-7)  # Evitar log(0)
        return -np.mean(y_class * np.log(predictions) + (1-y_class) * np.log(1-predictions))

    Loss = np.array([[cross_entropy_loss(w, b) for w in w_range] for b in b_range])

    contour = axes[1].contour(W, B, Loss, levels=20)
    axes[1].clabel(contour, inline=True, fontsize=8)
    axes[1].set_xlabel('Peso (w)')
    axes[1].set_ylabel('Sesgo (b)')
    axes[1].set_title('Landscape de Cross-Entropy Loss')
    axes[1].grid(True, alpha=0.3)

    # 3. Comparación de funciones de pérdida
    y_true = 1  # Clase verdadera
    predictions = np.linspace(0.01, 0.99, 100)

    mse_loss = (predictions - y_true)**2
    cross_entropy = -np.log(predictions)  # Para y_true = 1

    axes[2].plot(predictions, mse_loss, 'b-', linewidth=2, label='MSE Loss')
    axes[2].plot(predictions, cross_entropy, 'r-', linewidth=2, label='Cross-Entropy')
    axes[2].set_xlabel('Predicción')
    axes[2].set_ylabel('Pérdida')
    axes[2].set_title('Comparación de Funciones de Pérdida')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

preview_deep_learning()
```

### Preparándote para la Semana 4

La próxima semana nos sumergiremos en **estadística y probabilidad**, que son el otro pilar fundamental del machine learning:

```python
def preview_semana_4():
    """
    Vista previa de lo que viene en Estadística y Probabilidad
    """
    print("🔜 PRÓXIMA SEMANA: ESTADÍSTICA Y PROBABILIDAD")
    print("=" * 60)

    print("""
    📈 DISTRIBUCIONES DE PROBABILIDAD:
    - ¿Por qué los algoritmos de ML hacen "suposiciones" sobre los datos?
    - Distribución normal, Bernoulli, Poisson
    - Máxima verosimilitud: la conexión con funciones de costo

    🎯 INFERENCIA ESTADÍSTICA:
    - Intervalos de confianza para predicciones
    - Tests de hipótesis para validar modelos
    - Bootstrap y validación cruzada

    🎲 PROBABILIDAD BAYESIANA:
    - Naive Bayes: probabilidad aplicada a clasificación
    - Prior, likelihood, posterior: el trío mágico
    - Incertidumbre en machine learning

    📊 CONEXIONES CON HOY:
    - MLE (Maximum Likelihood) → minimizar pérdida → gradient descent
    - Regularización → distribuciones prior bayesianas
    - Cross-validation → distribuciones muestrales
    """)

    # Ejemplo simple: conexión MLE con gradient descent
    print("\n🔗 CONEXIÓN DIRECTA CON GRADIENT DESCENT:")
    print("-" * 40)

    print("Para regresión lineal con ruido gaussiano:")
    print("  1. Suponemos: y = Xw + ε, donde ε ~ N(0, σ²)")
    print("  2. Maximum Likelihood Estimation:")
    print("     L(w) = ∏ P(yᵢ | xᵢ, w)")
    print("  3. Log-likelihood: log L(w) = -½∑(yᵢ - xᵢw)²/σ²")
    print("  4. Maximizar log L(w) ≡ Minimizar ∑(yᵢ - xᵢw)²")
    print("  5. ¡Eso es exactamente MSE que optimizamos con gradient descent!")

    # Visualización rápida
    plt.figure(figsize=(15, 5))

    # 1. Distribución normal del error
    plt.subplot(1, 3, 1)
    x_error = np.linspace(-3, 3, 1000)
    y_normal = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x_error**2)
    plt.plot(x_error, y_normal, 'b-', linewidth=2)
    plt.fill_between(x_error, 0, y_normal, alpha=0.3)
    plt.axvline(0, color='red', linestyle='--', label='Error = 0')
    plt.xlabel('Error (ε)')
    plt.ylabel('Densidad de Probabilidad')
    plt.title('Suposición: Errores ~ N(0,σ²)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Función de verosimilitud vs MSE
    plt.subplot(1, 3, 2)
    w_vals = np.linspace(-2, 4, 100)

    # Para datos sintéticos simples
    np.random.seed(42)
    x_simple = np.array([1, 2, 3, 4, 5])
    y_simple = 2 * x_simple + 1 + 0.2 * np.random.randn(5)

    mse_vals = [(np.sum((y_simple - w * x_simple)**2)) for w in w_vals]
    log_likelihood = [-0.5 * mse for mse in mse_vals]  # Simplificado

    plt.plot(w_vals, mse_vals, 'r-', linewidth=2, label='MSE')
    plt.xlabel('Parámetro w')
    plt.ylabel('MSE')
    plt.title('MSE a Minimizar')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Log-likelihood (a maximizar)
    plt.subplot(1, 3, 3)
    plt.plot(w_vals, log_likelihood, 'g-', linewidth=2, label='Log-Likelihood')
    plt.axvline(w_vals[np.argmax(log_likelihood)], color='red', linestyle='--',
                label='MLE Óptimo')
    plt.xlabel('Parámetro w')
    plt.ylabel('Log-Likelihood')
    plt.title('Log-Likelihood a Maximizar')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n💡 LA GRAN REVELACIÓN:")
    print("Gradient descent no es solo un 'truco de optimización'.")
    print("Es la implementación computacional de principios estadísticos profundos!")

preview_semana_4()
```

---

## Resumen y Próximos Pasos

### Lo que Dominaste Esta Semana

```python
def resumen_semanal():
    """
    Resumen completo de los conceptos aprendidos
    """
    print("🎉 RESUMEN DE LA SEMANA 3")
    print("=" * 50)

    print("✅ CONCEPTOS DOMINADOS:")
    print("""
    🔢 DERIVADAS:
    - Intuición geométrica como pendiente de la tangente
    - Reglas básicas: potencia, suma, producto, cadena
    - Implementación numérica vs analítica

    🌐 DERIVADAS PARCIALES:
    - Funciones de múltiples variables
    - Vector gradiente como dirección de máximo crecimiento
    - Interpretación geométrica en 3D

    ⚡ GRADIENT DESCENT:
    - Algoritmo de optimización fundamental
    - Importancia del learning rate
    - Convergencia y problemas comunes

    🛠️ IMPLEMENTACIÓN PRÁCTICA:
    - Gradient descent desde cero
    - Diferentes variantes y optimizadores
    - Análisis de convergencia y debugging

    🤖 CONEXIÓN CON ML:
    - Funciones de costo comunes (MSE, cross-entropy)
    - Regularización L1 y L2
    - Fundamentos para redes neuronales
    """)

    print("\n🎯 HABILIDADES PRÁCTICAS ADQUIRIDAS:")
    skills = [
        "Implementar gradient descent desde cero",
        "Calcular gradientes numéricos para verificación",
        "Visualizar paisajes de optimización en 2D/3D",
        "Diagnosticar problemas de convergencia",
        "Comparar diferentes optimizadores",
        "Aplicar regularización para evitar overfitting",
        "Conectar teoría matemática con implementación práctica"
    ]

    for i, skill in enumerate(skills, 1):
        print(f"  {i}. {skill}")

    print("\n🚀 PREPARADO PARA:")
    print("  - Estadística y Probabilidad (Semana 4)")
    print("  - Algoritmos de Machine Learning supervisado")
    print("  - Redes neuronales y deep learning")
    print("  - Optimización avanzada en problemas reales")

resumen_semanal()
```

### Desafío Final de la Semana

```python
def desafio_final():
    """
    Desafío integrador para consolidar todo el aprendizaje
    """
    print("🏆 DESAFÍO FINAL: OPTIMIZADOR INTELIGENTE")
    print("=" * 60)

    print("""
    🎯 TU MISIÓN:
    Implementa un optimizador que combine TODAS las técnicas aprendidas:

    1. 📊 MÚLTIPLES ALGORITMOS:
       - SGD básico
       - SGD con momentum
       - Learning rate adaptativo

    2. 🛡️ ROBUSTEZ:
       - Gradient clipping para gradientes explosivos
       - Early stopping para evitar overfitting
       - Regularización automática

    3. 📈 MONITOREO:
       - Métricas de convergencia en tiempo real
       - Detección automática de problemas
       - Visualizaciones interactivas

    4. 🧪 VALIDACIÓN:
       - Gradient checking automático
       - Comparación con soluciones analíticas
       - Tests unitarios para cada componente

    📝 ENTREGABLES:
    - Código comentado y documentado
    - Análisis comparativo de performance
    - Visualizaciones comprehensivas
    - Reporte técnico con conclusiones
    """)

    print("\n💡 CRITERIOS DE EVALUACIÓN:")
    criterios = [
        "Corrección matemática de las implementaciones",
        "Calidad del código (legibilidad, documentación)",
        "Profundidad del análisis experimental",
        "Creatividad en las visualizaciones",
        "Conexión con conceptos teóricos",
        "Preparación para temas avanzados"
    ]

    for i, criterio in enumerate(criterios, 1):
        print(f"  {i}. {criterio}")

    print(f"\n⏰ TIEMPO ESTIMADO: 4-6 horas")
    print(f"🎁 RECOMPENSA: Comprensión profunda del motor de todo ML moderno")

desafio_final()
```

---

## Palabras Finales

Felicitaciones por completar la semana más matemáticamente intensa del programa. Lo que acabás de aprender no es solo teoría abstracta: **es el corazón pulsante de toda la inteligencia artificial moderna**.

Cada vez que una red neuronal aprende a reconocer imágenes, cada vez que un modelo de lenguaje genera texto coherente, cada vez que un algoritmo de recomendación sugiere contenido personalizado, **está usando los principios de cálculo diferencial que dominaste hoy**.

El gradient descent que implementaste desde cero es el mismo algoritmo (con variaciones) que entrena:
- GPT y otros modelos de lenguaje
- Redes convolucionales para visión computacional
- Sistemas de recomendación de Netflix y Spotify
- Algoritmos de trading automatizado
- Modelos de predicción médica

**Has construido los cimientos. Ahora viene lo divertido: construir el edificio.**

La próxima semana, cuando exploremos estadística y probabilidad, vas a ver cómo estos conceptos matemáticos se conectan con la **incertidumbre** y la **toma de decisiones** - los otros pilares fundamentales de la inteligencia artificial.

Pero por ahora, tomate un momento para apreciar lo que lograste. Pasaste de no entender qué era una derivada a implementar algoritmos de optimización desde cero. **Eso no es poca cosa.**

**¡Nos vemos la próxima semana para conquistar el mundo de la probabilidad!** 🚀

---

{{< alert "circle-info" >}}
**Recursos adicionales para profundizar:**
- Khan Academy: Calculus
- 3Blue1Brown: Essence of Calculus (YouTube)
- MIT 18.01: Single Variable Calculus
- Coursera: Mathematics for Machine Learning Specialization
{{< /alert >}}
