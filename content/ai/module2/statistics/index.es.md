---
title: "Estadística Práctica para Machine Learning: De las Distribuciones a los Modelos Predictivos"
date: 2024-01-15
draft: false
description: "Domina los conceptos estadísticos esenciales para machine learning. Aprende distribuciones, probabilidad y análisis de incertidumbre con Python desde cero."
tags: ["estadistica", "machine learning", "probabilidad", "python", "data science", "AI engineering"]
categories: ["Machine Learning", "Estadística", "Python"]
series: "AI Engineer Path"
weight: 3
math: true
author: "AI Engineering Team"
keywords: ["estadistica machine learning", "probabilidad IA", "analisis datos python", "distribuciones probabilidad", "teorema bayes ML"]
---

{{< alert "circle-info" >}}
**Serie: De Software Engineer a AI Engineer - Semana 3**

Esta es la tercera entrega de nuestro programa de 16 semanas. Si llegaste aquí directamente, te recomendamos revisar las semanas anteriores:
- [Semana 1: Fundamentos de IA](../fundamentos-ia)
- [Semana 2: Álgebra Lineal para ML](../algebra-lineal-ml)
{{< /alert >}}

## Introducción: La Incertidumbre Como Motor de la IA

En el mundo real, los datos nunca son perfectos. Los sensores fallan, las mediciones contienen ruido, y los patrones están mezclados con variabilidad aleatoria. Como AI engineers, nuestro trabajo no es eliminar esta incertidumbre, sino **modelarla, cuantificarla y aprovecharla** para tomar mejores decisiones.

Imaginate que estás desarrollando un sistema de recomendaciones para una plataforma de streaming. ¿Cómo podes estar seguro de que tu modelo realmente entiende las preferencias del usuario y no está simplemente memorizando patrones aleatorios? ¿Cómo medís la confianza en cada predicción? ¿Cómo diferenciás entre correlación y causación?

La respuesta está en la **estadística y probabilidad aplicada a machine learning**.

### ¿Por Qué la Estadística es Fundamental en IA?

Mientras que el álgebra lineal nos da las herramientas computacionales y el cálculo nos permite optimizar, la estadística nos proporciona el **framework conceptual** para:

- **Cuantificar la incertidumbre** en nuestras predicciones
- **Validar** que nuestros modelos generalizan correctamente
- **Interpretar** los resultados de manera rigurosa
- **Detectar** cuando algo está funcionando mal
- **Comparar** diferentes enfoques de manera objetiva

{{< mermaid >}}
graph TD
    A[Datos del Mundo Real] --> B[Ruido + Incertidumbre]
    B --> C[Modelos Estadísticos]
    C --> D[Distribuciones de Probabilidad]
    D --> E[Inferencia y Predicción]
    E --> F[Decisiones Informadas]

    style A fill:#ff6b6b
    style F fill:#51cf66
{{< /mermaid >}}

### Lo Que Vas a Aprender Hoy

En esta semana vas a desarrollar una comprensión sólida de:

1. **Estadística descriptiva** que realmente importa en ML
2. **Distribuciones de probabilidad** y cuándo usarlas
3. **Correlación vs causación** (el error más costoso en data science)
4. **Teorema de Bayes** aplicado a clasificación y actualización de creencias
5. **Intervalos de confianza** para cuantificar incertidumbre
6. **Implementación práctica** con Python desde cero

## Parte I: Fundamentos Estadísticos para Machine Learning

### 1.1 Medidas de Tendencia Central: Más Allá del Promedio

Como software engineers, probablemente ya conoces el concepto de promedio. Pero en machine learning, necesitamos ser más precisos sobre **cuándo y por qué** usar cada medida.

#### Media Aritmética: El Punto de Equilibrio

La media es el valor que minimiza la suma de errores cuadráticos:

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# Implementación desde cero de estadísticas básicas
class EstadisticasML:
    """Clase para cálculos estadísticos fundamentales en ML"""

    @staticmethod
    def media(datos):
        """
        Calcula la media aritmética

        Args:
            datos: lista o array de números

        Returns:
            float: media aritmética
        """
        return sum(datos) / len(datos)

    @staticmethod
    def mediana(datos):
        """
        Calcula la mediana (valor central)
        Más robusta que la media ante outliers
        """
        datos_ordenados = sorted(datos)
        n = len(datos_ordenados)

        if n % 2 == 0:
            # Si hay número par de elementos, promedio de los dos centrales
            return (datos_ordenados[n//2 - 1] + datos_ordenados[n//2]) / 2
        else:
            # Si hay número impar, el elemento central
            return datos_ordenados[n//2]

    @staticmethod
    def moda(datos):
        """
        Encuentra el valor más frecuente
        Útil para variables categóricas
        """
        from collections import Counter
        contador = Counter(datos)
        return contador.most_common(1)[0][0]

# Ejemplo práctico: Análisis de latencia de respuesta de una API
np.random.seed(42)

# Simulamos latencias de API (la mayoría normales, algunos outliers)
latencias_normales = np.random.normal(100, 15, 950)  # 95% de requests normales
latencias_outliers = np.random.exponential(500, 50)  # 5% con problemas de red
latencias = np.concatenate([latencias_normales, latencias_outliers])

calc = EstadisticasML()

print("=== Análisis de Latencias de API ===")
print(f"Media: {calc.media(latencias):.2f} ms")
print(f"Mediana: {calc.mediana(latencias):.2f} ms")

# ¿Por qué la diferencia es importante en ML?
print(f"\nDiferencia: {calc.media(latencias) - calc.mediana(latencias):.2f} ms")
print("La media está inflada por outliers - en ML esto puede sesgar tu modelo!")
```

{{< alert "lightbulb" >}}
**Insight para ML**: En sistemas de producción, la mediana es a menudo más útil que la media para métricas de performance, ya que no se ve afectada por outliers extremos.
{{< /alert >}}

#### Visualización Inteligente de Tendencias Centrales

```python
# Creamos una visualización que muestra por qué importa la elección
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Distribución con outliers
ax1.hist(latencias, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
ax1.axvline(calc.media(latencias), color='red', linestyle='--',
           label=f'Media: {calc.media(latencias):.1f}ms', linewidth=2)
ax1.axvline(calc.mediana(latencias), color='green', linestyle='-',
           label=f'Mediana: {calc.mediana(latencias):.1f}ms', linewidth=2)
ax1.set_xlabel('Latencia (ms)')
ax1.set_ylabel('Frecuencia')
ax1.set_title('Distribución con Outliers\n(Datos Reales de API)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Solo datos normales
ax2.hist(latencias_normales, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
ax2.axvline(calc.media(latencias_normales), color='red', linestyle='--',
           label=f'Media: {calc.media(latencias_normales):.1f}ms', linewidth=2)
ax2.axvline(calc.mediana(latencias_normales), color='green', linestyle='-',
           label=f'Mediana: {calc.mediana(latencias_normales):.1f}ms', linewidth=2)
ax2.set_xlabel('Latencia (ms)')
ax2.set_ylabel('Frecuencia')
ax2.set_title('Distribución Sin Outliers\n(Condiciones Ideales)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 1.2 Medidas de Dispersión: Cuantificando la Incertidumbre

La dispersión nos dice **qué tan confiables** son nuestras medidas centrales. En ML, esto es crucial para entender la **variabilidad** de nuestros datos y modelos.

#### Varianza y Desviación Estándar: Los Fundamentos

$$\sigma^2 = \frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})^2$$

$$\sigma = \sqrt{\sigma^2}$$

```python
class DispersionML:
    """Cálculos de dispersión optimizados para ML"""

    @staticmethod
    def varianza(datos, poblacional=True):
        """
        Calcula la varianza

        Args:
            datos: array de datos
            poblacional: True para división por n, False para n-1 (muestra)
        """
        media = sum(datos) / len(datos)
        suma_cuadrados = sum((x - media)**2 for x in datos)

        # Corrección de Bessel para muestras
        divisor = len(datos) if poblacional else len(datos) - 1
        return suma_cuadrados / divisor

    @staticmethod
    def desviacion_estandar(datos, poblacional=True):
        """Desviación estándar - mismas unidades que los datos originales"""
        return DispersionML.varianza(datos, poblacional) ** 0.5

    @staticmethod
    def rango_intercuartil(datos):
        """
        IQR - medida de dispersión robusta ante outliers
        Útil para detección de anomalías
        """
        datos_ordenados = sorted(datos)
        n = len(datos_ordenados)

        # Cuartiles
        q1_idx = n // 4
        q3_idx = 3 * n // 4

        q1 = datos_ordenados[q1_idx]
        q3 = datos_ordenados[q3_idx]

        return q3 - q1, q1, q3

# Ejemplo: Análisis de precisión de un modelo
# Simulamos predicciones de un modelo de regresión
np.random.seed(123)

modelo_A_errors = np.random.normal(0, 2, 1000)    # Modelo consistente
modelo_B_errors = np.random.normal(0, 5, 1000)    # Modelo inconsistente
modelo_C_errors = np.random.laplace(0, 1.5, 1000) # Modelo con distribución diferente

disp = DispersionML()

modelos = {
    'Modelo A (Consistente)': modelo_A_errors,
    'Modelo B (Inconsistente)': modelo_B_errors,
    'Modelo C (Heavy-tailed)': modelo_C_errors
}

print("=== Comparación de Modelos por Dispersión ===")
for nombre, errores in modelos.items():
    media = sum(errores) / len(errores)
    std = disp.desviacion_estandar(errores)
    iqr, q1, q3 = disp.rango_intercuartil(errores)

    print(f"\n{nombre}:")
    print(f"  Media del error: {media:.3f}")
    print(f"  Desv. estándar: {std:.3f}")
    print(f"  IQR: {iqr:.3f}")
    print(f"  Rango 68%: [{media-std:.3f}, {media+std:.3f}]")
```

#### Visualización de Dispersión Comparativa

```python
# Boxplot comparativo para entender dispersión
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Boxplot
errores_lista = [modelo_A_errors, modelo_B_errors, modelo_C_errors]
labels = ['Modelo A\n(σ=2)', 'Modelo B\n(σ=5)', 'Modelo C\n(Heavy-tail)']

ax1.boxplot(errores_lista, labels=labels)
ax1.set_ylabel('Error de Predicción')
ax1.set_title('Comparación de Dispersión\n(Boxplots)')
ax1.grid(True, alpha=0.3)

# Histogramas superpuestos
colors = ['blue', 'red', 'green']
alphas = [0.7, 0.5, 0.6]

for i, (errores, label, color, alpha) in enumerate(zip(errores_lista, labels, colors, alphas)):
    ax2.hist(errores, bins=50, alpha=alpha, color=color,
            label=label.replace('\n', ' '), density=True)

ax2.set_xlabel('Error de Predicción')
ax2.set_ylabel('Densidad')
ax2.set_title('Distribución de Errores\n(Histogramas Superpuestos)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== Interpretación para ML ===")
print("• Modelo A: Baja varianza → Predicciones consistentes")
print("• Modelo B: Alta varianza → Predicciones menos confiables")
print("• Modelo C: Heavy tails → Mayor probabilidad de errores grandes")
```

{{< alert "exclamation-triangle" >}}
**Error Común**: Muchos desarrolladores se enfocan solo en la media del error y ignoran la varianza. En producción, la **consistencia** (baja varianza) puede ser más valiosa que la precisión promedio.
{{< /alert >}}

### 1.3 Correlación: El Arte de Detectar Relaciones

La correlación mide la **fuerza de la relación lineal** entre dos variables. Pero atención: correlación ≠ causación.

#### Coeficiente de Correlación de Pearson

$$r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}$$

```python
class CorrelacionML:
    """Análisis de correlación para feature engineering"""

    @staticmethod
    def correlacion_pearson(x, y):
        """
        Calcula correlación de Pearson desde cero

        Returns:
            float: correlación entre -1 y 1
        """
        n = len(x)

        # Medias
        media_x = sum(x) / n
        media_y = sum(y) / n

        # Numerador: covarianza
        numerador = sum((x[i] - media_x) * (y[i] - media_y) for i in range(n))

        # Denominador: producto de desviaciones estándar
        suma_sq_x = sum((x[i] - media_x)**2 for i in range(n))
        suma_sq_y = sum((y[i] - media_y)**2 for i in range(n))

        denominador = (suma_sq_x * suma_sq_y) ** 0.5

        return numerador / denominador if denominador != 0 else 0

    @staticmethod
    def correlacion_spearman(x, y):
        """
        Correlación de Spearman (basada en rankings)
        Más robusta para relaciones no lineales
        """
        def ranking(datos):
            # Crear rankings (1 = menor, n = mayor)
            sorted_data = sorted(enumerate(datos), key=lambda item: item[1])
            ranks = [0] * len(datos)
            for rank, (original_index, _) in enumerate(sorted_data):
                ranks[original_index] = rank + 1
            return ranks

        rank_x = ranking(x)
        rank_y = ranking(y)

        return CorrelacionML.correlacion_pearson(rank_x, rank_y)

# Ejemplo: Feature Engineering para un modelo predictivo
np.random.seed(456)

# Simulamos datos de un e-commerce
n_usuarios = 1000
edad = np.random.normal(35, 12, n_usuarios)
ingresos = np.random.normal(50000, 15000, n_usuarios)

# Relaciones realistas
tiempo_web = 5 + 0.1 * edad + np.random.normal(0, 2, n_usuarios)  # Relación débil con edad
compras = 0.3 + 0.002 * ingresos + 0.5 * tiempo_web + np.random.normal(0, 1, n_usuarios)  # Relación fuerte con ingresos y tiempo

# También creamos una variable no-lineal
satisfaccion = 5 / (1 + np.exp(-(compras - 2))) + np.random.normal(0, 0.3, n_usuarios)  # Relación sigmoidal

corr = CorrelacionML()

print("=== Matriz de Correlaciones ===")
variables = {
    'Edad': edad,
    'Ingresos': ingresos,
    'Tiempo Web': tiempo_web,
    'Compras': compras,
    'Satisfacción': satisfaccion
}

# Matriz de correlación manual
nombres = list(variables.keys())
n_vars = len(nombres)

print("Correlaciones de Pearson:")
print("Variable 1\t\tVariable 2\t\tPearson\t\tSpearman")
print("-" * 70)

for i in range(n_vars):
    for j in range(i+1, n_vars):
        var1_name, var1_data = nombres[i], variables[nombres[i]]
        var2_name, var2_data = nombres[j], variables[nombres[j]]

        pearson = corr.correlacion_pearson(var1_data, var2_data)
        spearman = corr.correlacion_spearman(var1_data, var2_data)

        print(f"{var1_name:<15}\t{var2_name:<15}\t{pearson:.3f}\t\t{spearman:.3f}")
```

#### Visualización de Correlaciones: Scatter Matrix

```python
# Creamos un scatter plot matrix profesional
fig, axes = plt.subplots(n_vars, n_vars, figsize=(16, 16))

for i in range(n_vars):
    for j in range(n_vars):
        ax = axes[i, j]

        if i == j:
            # Diagonal: histogramas
            ax.hist(variables[nombres[i]], bins=30, alpha=0.7, color='skyblue')
            ax.set_title(f'{nombres[i]}')
        else:
            # Off-diagonal: scatter plots
            x_data = variables[nombres[j]]
            y_data = variables[nombres[i]]

            ax.scatter(x_data, y_data, alpha=0.5, s=10)

            # Añadir línea de tendencia
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            ax.plot(sorted(x_data), p(sorted(x_data)), "r--", alpha=0.8)

            # Mostrar correlación en el plot
            pearson_corr = corr.correlacion_pearson(x_data, y_data)
            ax.text(0.05, 0.95, f'r={pearson_corr:.2f}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Limpiar labels excepto en bordes
        if i < n_vars - 1:
            ax.set_xlabel('')
        else:
            ax.set_xlabel(nombres[j])

        if j > 0:
            ax.set_ylabel('')
        else:
            ax.set_ylabel(nombres[i])

plt.tight_layout()
plt.show()
```

### 1.4 Correlación vs Causación: El Error de los $1000M

Esta distinción es **crítica** en machine learning. Una correlación fuerte no implica que una variable cause la otra.

```python
# Ejemplo clásico: El peligro de confundir correlación con causación
np.random.seed(789)

# Variable oculta: temperatura
temperatura = np.random.normal(25, 8, 365)  # Temperatura diaria

# Variables correlacionadas por la temperatura (no entre sí directamente)
ventas_helados = 50 + 3 * temperatura + np.random.normal(0, 10, 365)
ahogamientos = 1 + 0.2 * temperatura + np.random.normal(0, 2, 365)

# Correlación espuria
correlacion_espuria = corr.correlacion_pearson(ventas_helados, ahogamientos)

print("=== El Peligro de la Correlación Espuria ===")
print(f"Correlación Helados ↔ Ahogamientos: {correlacion_espuria:.3f}")
print(f"Correlación Temperatura ↔ Helados: {corr.correlacion_pearson(temperatura, ventas_helados):.3f}")
print(f"Correlación Temperatura ↔ Ahogamientos: {corr.correlacion_pearson(temperatura, ahogamientos):.3f}")

print("\n¿Conclusión equivocada?")
print("→ Los helados causan ahogamientos")
print("\nConclusión correcta:")
print("→ La temperatura (variable oculta) influye en ambas variables")

# Visualización del problema
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Correlación espuria
ax1.scatter(ventas_helados, ahogamientos, alpha=0.6, color='red')
ax1.set_xlabel('Ventas de Helados')
ax1.set_ylabel('Ahogamientos')
ax1.set_title(f'Correlación Espuria\nr = {correlacion_espuria:.2f}')
ax1.grid(True, alpha=0.3)

# Plot 2: Variable real (temperatura → helados)
scatter = ax2.scatter(temperatura, ventas_helados, c=temperatura,
                     cmap='coolwarm', alpha=0.6)
ax2.set_xlabel('Temperatura (°C)')
ax2.set_ylabel('Ventas de Helados')
ax2.set_title('Relación Real: Temperatura → Helados')
ax2.grid(True, alpha=0.3)

# Plot 3: Variable real (temperatura → ahogamientos)
scatter = ax3.scatter(temperatura, ahogamientos, c=temperatura,
                     cmap='coolwarm', alpha=0.6)
ax3.set_xlabel('Temperatura (°C)')
ax3.set_ylabel('Ahogamientos')
ax3.set_title('Relación Real: Temperatura → Ahogamientos')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

{{< alert "exclamation-triangle" >}}
**Error Crítico en ML**: Usar correlaciones espurias como features puede llevar a modelos que funcionan en entrenamiento pero fallan en producción cuando las condiciones cambian.
{{< /alert >}}

## Parte II: Distribuciones de Probabilidad en Machine Learning

Las distribuciones de probabilidad son los **bloques fundamentales** para modelar la incertidumbre en ML. Cada distribución tiene sus casos de uso específicos.

### 2.1 Distribución Normal: La Reina de las Distribuciones

La distribución normal aparece constantemente en ML debido al **Teorema del Límite Central**.

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

```python
class DistribucionesML:
    """Implementaciones y análisis de distribuciones clave para ML"""

    @staticmethod
    def densidad_normal(x, mu, sigma):
        """
        Calcula la densidad de probabilidad normal

        Args:
            x: valor donde evaluar
            mu: media
            sigma: desviación estándar
        """
        return (1 / (sigma * (2 * np.pi)**0.5)) * np.exp(-0.5 * ((x - mu) / sigma)**2)

    @staticmethod
    def generar_normal(mu, sigma, n):
        """
        Genera muestras de distribución normal usando Box-Muller
        (implementación educativa)
        """
        # Método Box-Muller
        muestras = []
        for _ in range(n // 2 + 1):
            u1 = np.random.random()
            u2 = np.random.random()

            z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
            z1 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)

            muestras.extend([mu + sigma * z0, mu + sigma * z1])

        return np.array(muestras[:n])

    @staticmethod
    def regla_68_95_99(mu, sigma):
        """
        Intervalos de confianza para distribución normal
        """
        return {
            '68%': (mu - sigma, mu + sigma),
            '95%': (mu - 2*sigma, mu + 2*sigma),
            '99.7%': (mu - 3*sigma, mu + 3*sigma)
        }

# Ejemplo: Análisis de errores de modelo
dist = DistribucionesML()

# Simulamos errores de diferentes modelos
np.random.seed(101)

# Modelo 1: Errores bien calibrados (normal)
errores_modelo1 = np.random.normal(0, 1, 1000)

# Modelo 2: Errores sesgados
errores_modelo2 = np.random.normal(0.5, 1.2, 1000)

# Modelo 3: Errores con heavy tails (no normal)
errores_modelo3 = np.random.laplace(0, 0.8, 1000)

print("=== Análisis de Normalidad de Errores ===")

modelos_errores = [
    ("Modelo Bien Calibrado", errores_modelo1),
    ("Modelo Sesgado", errores_modelo2),
    ("Modelo Heavy-Tail", errores_modelo3)
]

for nombre, errores in modelos_errores:
    media = np.mean(errores)
    std = np.std(errores)
    intervalos = dist.regla_68_95_99(media, std)

    print(f"\n{nombre}:")
    print(f"  Media: {media:.3f}, Std: {std:.3f}")
    print(f"  68% de errores en: [{intervalos['68%'][0]:.2f}, {intervalos['68%'][1]:.2f}]")

    # Verificar cuántos errores caen realmente en el intervalo 68%
    en_intervalo = np.sum((errores >= intervalos['68%'][0]) &
                         (errores <= intervalos['68%'][1]))
    porcentaje_real = en_intervalo / len(errores) * 100
    print(f"  Porcentaje real en 68%: {porcentaje_real:.1f}%")

    if abs(porcentaje_real - 68) > 5:
        print(f"  ⚠️  Los errores NO siguen distribución normal")
    else:
        print(f"  ✓ Los errores aproximan distribución normal")
```

#### Test de Normalidad: Shapiro-Wilk

```python
# Implementación simplificada del test de normalidad
def test_normalidad_visual(datos, nombre=""):
    """Test visual de normalidad usando Q-Q plot"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histograma vs normal teórica
    ax1.hist(datos, bins=30, density=True, alpha=0.7, color='skyblue')

    # Superponer normal teórica
    mu, sigma = np.mean(datos), np.std(datos)
    x_teorico = np.linspace(min(datos), max(datos), 100)
    y_teorico = dist.densidad_normal(x_teorico, mu, sigma)
    ax1.plot(x_teorico, y_teorico, 'r-', linewidth=2, label='Normal Teórica')

    ax1.set_xlabel('Valor')
    ax1.set_ylabel('Densidad')
    ax1.set_title(f'Histograma vs Normal\n{nombre}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Q-Q Plot (quantile-quantile)
    datos_ordenados = np.sort(datos)
    n = len(datos_ordenados)
    quantiles_teoricos = np.random.normal(mu, sigma, n)
    quantiles_teoricos = np.sort(quantiles_teoricos)

    ax2.scatter(quantiles_teoricos, datos_ordenados, alpha=0.6)

    # Línea de referencia (y = x)
    min_val = min(min(quantiles_teoricos), min(datos_ordenados))
    max_val = max(max(quantiles_teoricos), max(datos_ordenados))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

    ax2.set_xlabel('Quantiles Teóricos (Normal)')
    ax2.set_ylabel('Quantiles Observados')
    ax2.set_title(f'Q-Q Plot\n{nombre}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Aplicamos el test a nuestros modelos
for nombre, errores in modelos_errores:
    test_normalidad_visual(errores, nombre)
```

### 2.2 Distribución Binomial: Modelando el Éxito/Fracaso

Perfecta para clasificación binaria y A/B testing.

$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$

```python
# Implementación de distribución binomial
class DistribucionBinomial:
    """Distribución binomial para clasificación y testing"""

    @staticmethod
    def factorial(n):
        """Factorial usando iteración (más eficiente)"""
        if n <= 1:
            return 1
        resultado = 1
        for i in range(2, n + 1):
            resultado *= i
        return resultado

    @staticmethod
    def combinatoria(n, k):
        """Coeficiente binomial C(n,k)"""
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1

        # Optimización: C(n,k) = C(n,n-k)
        k = min(k, n - k)

        resultado = 1
        for i in range(k):
            resultado = resultado * (n - i) // (i + 1)
        return resultado

    @staticmethod
    def probabilidad(n, k, p):
        """P(X = k) para binomial(n, p)"""
        comb = DistribucionBinomial.combinatoria(n, k)
        return comb * (p ** k) * ((1 - p) ** (n - k))

    @staticmethod
    def media_varianza(n, p):
        """Media y varianza de binomial"""
        media = n * p
        varianza = n * p * (1 - p)
        return media, varianza

# Ejemplo: Análisis de conversión web
binomial = DistribucionBinomial()

# Escenario: A/B test de un botón de compra
n_visitantes = 1000
tasa_conversion_A = 0.05  # 5% versión original
tasa_conversion_B = 0.07  # 7% versión nueva

print("=== A/B Test: Análisis de Conversión ===")
print(f"Visitantes por variante: {n_visitantes}")
print(f"Tasa de conversión A: {tasa_conversion_A*100:.1f}%")
print(f"Tasa de conversión B: {tasa_conversion_B*100:.1f}%")

# Calculamos distribuciones esperadas
media_A, var_A = binomial.media_varianza(n_visitantes, tasa_conversion_A)
media_B, var_B = binomial.media_varianza(n_visitantes, tasa_conversion_B)

print(f"\nVariante A - Conversiones esperadas: {media_A:.1f} ± {np.sqrt(var_A):.1f}")
print(f"Variante B - Conversiones esperadas: {media_B:.1f} ± {np.sqrt(var_B):.1f}")

# Probabilidad de diferentes números de conversiones
conversiones_rango = range(20, 101, 5)
prob_A = [binomial.probabilidad(n_visitantes, k, tasa_conversion_A) for k in conversiones_rango]
prob_B = [binomial.probabilidad(n_visitantes, k, tasa_conversion_B) for k in conversiones_rango]

# Visualización
plt.figure(figsize=(12, 6))
plt.plot(conversiones_rango, prob_A, 'b-', marker='o', label=f'Variante A (p={tasa_conversion_A})', linewidth=2)
plt.plot(conversiones_rango, prob_B, 'r-', marker='s', label=f'Variante B (p={tasa_conversion_B})', linewidth=2)

plt.axvline(media_A, color='blue', linestyle='--', alpha=0.7, label=f'Media A: {media_A:.1f}')
plt.axvline(media_B, color='red', linestyle='--', alpha=0.7, label=f'Media B: {media_B:.1f}')

plt.xlabel('Número de Conversiones')
plt.ylabel('Probabilidad')
plt.title('Distribución Binomial: A/B Test de Conversiones')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Análisis de significancia estadística
diferencia_esperada = media_B - media_A
error_estandar = np.sqrt(var_A + var_B)
z_score = diferencia_esperada / error_estandar

print(f"\n=== Análisis Estadístico ===")
print(f"Diferencia esperada: {diferencia_esperada:.1f} conversiones")
print(f"Error estándar: {error_estandar:.2f}")
print(f"Z-score: {z_score:.2f}")

if z_score > 1.96:
    print("✓ Diferencia estadísticamente significativa (95% confianza)")
else:
    print("⚠️ Diferencia NO significativa - necesitas más datos")
```

### 2.3 Distribución Uniforme: Modelando la Aleatoriedad Pura

```python
class DistribucionUniforme:
    """Distribución uniforme para inicialización y sampling"""

    @staticmethod
    def densidad(x, a, b):
        """Densidad uniforme en [a, b]"""
        if a <= x <= b:
            return 1 / (b - a)
        return 0

    @staticmethod
    def generar_muestra(a, b, n):
        """Genera n muestras uniformes en [a, b]"""
        return [a + (b - a) * np.random.random() for _ in range(n)]

    @staticmethod
    def media_varianza(a, b):
        """Media y varianza de uniforme[a, b]"""
        media = (a + b) / 2
        varianza = ((b - a) ** 2) / 12
        return media, varianza

# Ejemplo: Inicialización de pesos en redes neuronales
uniforme = DistribucionUniforme()

# Diferentes estrategias de inicialización
n_pesos = 1000

# Xavier/Glorot initialization
fan_in, fan_out = 128, 64  # Neuronas entrada y salida
limite_xavier = np.sqrt(6 / (fan_in + fan_out))
pesos_xavier = uniforme.generar_muestra(-limite_xavier, limite_xavier, n_pesos)

# He initialization (para ReLU)
limite_he = np.sqrt(6 / fan_in)
pesos_he = uniforme.generar_muestra(-limite_he, limite_he, n_pesos)

# Inicialización simple
pesos_simple = uniforme.generar_muestra(-0.1, 0.1, n_pesos)

print("=== Estrategias de Inicialización ===")

estrategias = {
    'Xavier/Glorot': pesos_xavier,
    'He (ReLU)': pesos_he,
    'Simple [-0.1, 0.1]': pesos_simple
}

for nombre, pesos in estrategias.items():
    media = np.mean(pesos)
    std = np.std(pesos)
    print(f"{nombre}: μ={media:.4f}, σ={std:.4f}")

# Visualización comparativa
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (nombre, pesos) in enumerate(estrategias.items()):
    axes[i].hist(pesos, bins=50, alpha=0.7, density=True)
    axes[i].set_title(f'{nombre}\nσ={np.std(pesos):.3f}')
    axes[i].set_xlabel('Valor del Peso')
    axes[i].set_ylabel('Densidad')
    axes[i].grid(True, alpha=0.3)

    # Línea teórica
    a, b = min(pesos), max(pesos)
    x_teorico = np.linspace(a, b, 100)
    y_teorico = [uniforme.densidad(x, a, b) for x in x_teorico]
    axes[i].plot(x_teorico, y_teorico, 'r-', linewidth=2, label='Teórica')
    axes[i].legend()

plt.tight_layout()
plt.show()
```

## Parte III: Teorema de Bayes en Machine Learning

El teorema de Bayes es **fundamental** para la inferencia probabilística y el aprendizaje adaptativo.

$P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}$

Donde:
- **P(H|E)**: Probabilidad posterior (lo que queremos)
- **P(E|H)**: Verosimilitud (likelihood)
- **P(H)**: Probabilidad a priori
- **P(E)**: Evidencia (normalización)

### 3.1 Implementación del Teorema de Bayes

```python
class BayesML:
    """Implementación del teorema de Bayes para ML"""

    @staticmethod
    def bayes_simple(prior, likelihood, evidencia):
        """
        Calcula probabilidad posterior usando Bayes

        Args:
            prior: P(H) - probabilidad a priori
            likelihood: P(E|H) - verosimilitud
            evidencia: P(E) - evidencia total
        """
        posterior = (likelihood * prior) / evidencia
        return posterior

    @staticmethod
    def bayes_clasificacion(prior_clases, likelihood_features):
        """
        Clasificación bayesiana multi-clase

        Args:
            prior_clases: dict con probabilidades a priori de cada clase
            likelihood_features: dict con P(features|clase) para cada clase
        """
        # Calcular evidencia total
        evidencia = sum(prior_clases[clase] * likelihood_features[clase]
                       for clase in prior_clases.keys())

        # Calcular posterior para cada clase
        posteriores = {}
        for clase in prior_clases.keys():
            posteriores[clase] = BayesML.bayes_simple(
                prior_clases[clase],
                likelihood_features[clase],
                evidencia
            )

        return posteriores

    @staticmethod
    def naive_bayes_gaussian(X_train, y_train, X_test):
        """
        Implementación simple de Naive Bayes Gaussiano
        """
        from collections import defaultdict

        # Estadísticas por clase
        clases = np.unique(y_train)
        n_total = len(y_train)

        # Calcular priors
        priors = {}
        estadisticas = {}

        for clase in clases:
            mask = y_train == clase
            priors[clase] = np.sum(mask) / n_total

            # Estadísticas de features para esta clase
            X_clase = X_train[mask]
            estadisticas[clase] = {
                'media': np.mean(X_clase, axis=0),
                'std': np.std(X_clase, axis=0)
            }

        # Predicciones
        predicciones = []
        probabilidades = []

        for x_test in X_test:
            # Calcular likelihood para cada clase
            likelihoods = {}

            for clase in clases:
                # Asumimos independencia (naive) y normalidad
                media = estadisticas[clase]['media']
                std = estadisticas[clase]['std']

                # Producto de probabilidades (log para estabilidad numérica)
                log_likelihood = 0
                for i, feature in enumerate(x_test):
                    # Densidad normal
                    prob = dist.densidad_normal(feature, media[i], std[i])
                    log_likelihood += np.log(prob + 1e-10)  # Evitar log(0)

                likelihoods[clase] = np.exp(log_likelihood)

            # Aplicar Bayes
            posteriores = BayesML.bayes_clasificacion(priors, likelihoods)

            # Predecir clase con mayor probabilidad
            clase_predicha = max(posteriores.keys(), key=lambda k: posteriores[k])
            predicciones.append(clase_predicha)
            probabilidades.append(posteriores)

        return predicciones, probabilidades

# Ejemplo: Sistema de detección de spam
np.random.seed(555)

# Simulamos un dataset de emails
n_emails = 1000

# Features: [longitud, num_enlaces, num_mayusculas, num_signos_exclamacion]
# Emails normales
emails_normales = np.random.multivariate_normal(
    [150, 2, 10, 1],  # medias
    [[400, 5, 20, 1],   # matriz de covarianza
     [5, 4, 5, 0.5],
     [20, 5, 100, 2],
     [1, 0.5, 2, 1]],
    600
)

# Emails spam
emails_spam = np.random.multivariate_normal(
    [80, 8, 35, 5],   # medias diferentes
    [[200, 10, 30, 3],
     [10, 16, 15, 2],
     [30, 15, 200, 8],
     [3, 2, 8, 4]],
    400
)

# Combinar datasets
X = np.vstack([emails_normales, emails_spam])
y = np.array([0] * 600 + [1] * 400)  # 0=normal, 1=spam

# División train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Aplicar nuestro Naive Bayes
bayes = BayesML()
predicciones, probabilidades = bayes.naive_bayes_gaussian(X_train, y_train, X_test)

# Evaluación
precisiones = sum(pred == real for pred, real in zip(predicciones, y_test))
accuracy = precisiones / len(y_test)

print("=== Sistema de Detección de Spam ===")
print(f"Accuracy: {accuracy:.3f}")
print(f"Emails de prueba: {len(y_test)}")
print(f"Predicciones correctas: {precisiones}")

# Análisis detallado de algunos casos
print("\n=== Análisis de Casos ===")
feature_names = ['Longitud', 'Enlaces', 'Mayúsculas', 'Exclamaciones']

for i in range(5):  # Primeros 5 casos
    print(f"\nEmail {i+1}:")
    print(f"  Features: {[f'{x:.1f}' for x in X_test[i]]}")
    print(f"  Clase real: {'Spam' if y_test[i] else 'Normal'}")
    print(f"  Predicción: {'Spam' if predicciones[i] else 'Normal'}")
    print(f"  P(Normal): {probabilidades[i][0]:.3f}")
    print(f"  P(Spam): {probabilidades[i][1]:.3f}")

    if predicciones[i] != y_test[i]:
        print(f"  ❌ Error de clasificación!")
```

### 3.2 Actualización Bayesiana: Aprendizaje Online

```python
class AprendizajeBayesiano:
    """Aprendizaje adaptativo usando actualización bayesiana"""

    def __init__(self, prior_alpha=1, prior_beta=1):
        """
        Inicializa con distribución Beta como prior
        Beta(α, β) es conjugada de la binomial
        """
        self.alpha = prior_alpha  # "éxitos" previos
        self.beta = prior_beta    # "fracasos" previos
        self.total_observaciones = 0
        self.historial = []

    def actualizar(self, exito):
        """
        Actualiza creencias con nueva observación

        Args:
            exito: True si fue éxito, False si fracaso
        """
        if exito:
            self.alpha += 1
        else:
            self.beta += 1

        self.total_observaciones += 1

        # Guardar estado para visualización
        self.historial.append({
            'observacion': self.total_observaciones,
            'alpha': self.alpha,
            'beta': self.beta,
            'media_estimada': self.alpha / (self.alpha + self.beta),
            'exito': exito
        })

    def probabilidad_estimada(self):
        """Estimación actual de la probabilidad de éxito"""
        return self.alpha / (self.alpha + self.beta)

    def intervalo_credible(self, confianza=0.95):
        """
        Intervalo de credibilidad bayesiano
        (equivalente bayesiano del intervalo de confianza)
        """
        from scipy.stats import beta
        alpha_nivel = (1 - confianza) / 2
        lower = beta.ppf(alpha_nivel, self.alpha, self.beta)
        upper = beta.ppf(1 - alpha_nivel, self.alpha, self.beta)
        return lower, upper

# Ejemplo: Sistema de recomendaciones adaptativo
# Simulamos clics en anuncios con tasa real del 8%
np.random.seed(666)
tasa_real = 0.08
n_observaciones = 200

# Inicializar aprendizaje bayesiano
aprendiz = AprendizajeBayesiano(prior_alpha=1, prior_beta=1)  # Prior uniforme

print("=== Aprendizaje Bayesiano Online ===")
print(f"Tasa real (desconocida): {tasa_real:.1%}")
print(f"Prior inicial: Beta(1,1) - uniforme")

# Simular observaciones secuenciales
observaciones = np.random.binomial(1, tasa_real, n_observaciones)

estimaciones = []
intervalos = []

for i, obs in enumerate(observaciones):
    aprendiz.actualizar(obs == 1)

    if (i + 1) % 20 == 0:  # Reportar cada 20 observaciones
        est = aprendiz.probabilidad_estimada()
        lower, upper = aprendiz.intervalo_credible()

        print(f"Obs {i+1:3d}: Estimación {est:.3f} [{lower:.3f}, {upper:.3f}]")

        estimaciones.append(est)
        intervalos.append((lower, upper))

# Visualización del aprendizaje
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Evolución de la estimación
obs_puntos = list(range(20, n_observaciones + 1, 20))
estimaciones_array = np.array(estimaciones)
intervalos_array = np.array(intervalos)

ax1.plot(obs_puntos, estimaciones_array, 'b-', marker='o', linewidth=2,
         label='Estimación Bayesiana')
ax1.fill_between(obs_puntos, intervalos_array[:, 0], intervalos_array[:, 1],
                alpha=0.3, color='blue', label='Intervalo 95% Credible')
ax1.axhline(tasa_real, color='red', linestyle='--', linewidth=2,
           label=f'Tasa Real: {tasa_real:.1%}')

ax1.set_xlabel('Número de Observaciones')
ax1.set_ylabel('Probabilidad Estimada')
ax1.set_title('Aprendizaje Bayesiano: Convergencia a la Tasa Real')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Distribución posterior final
from scipy.stats import beta
x_beta = np.linspace(0, 0.2, 1000)
y_beta = beta.pdf(x_beta, aprendiz.alpha, aprendiz.beta)

ax2.plot(x_beta, y_beta, 'blue', linewidth=3, label=f'Posterior: Beta({aprendiz.alpha}, {aprendiz.beta})')
ax2.axvline(tasa_real, color='red', linestyle='--', linewidth=2, label='Tasa Real')
ax2.axvline(aprendiz.probabilidad_estimada(), color='blue', linestyle=':', linewidth=2,
           label=f'Estimación: {aprendiz.probabilidad_estimada():.3f}')

# Área del intervalo de credibilidad
lower, upper = aprendiz.intervalo_credible()
x_fill = x_beta[(x_beta >= lower) & (x_beta <= upper)]
y_fill = beta.pdf(x_fill, aprendiz.alpha, aprendiz.beta)
ax2.fill_between(x_fill, y_fill, alpha=0.3, color='green',
                label=f'95% Credible: [{lower:.3f}, {upper:.3f}]')

ax2.set_xlabel('Tasa de Conversión')
ax2.set_ylabel('Densidad de Probabilidad')
ax2.set_title('Distribución Posterior Final')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n=== Resultado Final ===")
print(f"Estimación final: {aprendiz.probabilidad_estimada():.4f}")
print(f"Error absoluto: {abs(aprendiz.probabilidad_estimada() - tasa_real):.4f}")
print(f"Intervalo 95%: [{lower:.4f}, {upper:.4f}]")
print(f"¿Contiene la tasa real? {'✓' if lower <= tasa_real <= upper else '❌'}")
```

## Parte IV: Intervalos de Confianza y Estimación

Los intervalos de confianza nos permiten **cuantificar la incertidumbre** en nuestras estimaciones.

### 4.1 Intervalos de Confianza para la Media

```python
class IntervalosConfianza:
    """Cálculo de intervalos de confianza para diferentes parámetros"""

    @staticmethod
    def t_student_critico(df, alpha=0.05):
        """
        Valor crítico de t-Student (aproximación)
        Para df > 30, aproxima a distribución normal
        """
        # Valores críticos comunes (tabla simplificada)
        valores_t = {
            # grados_libertad: {alpha: valor_critico}
            1: {0.05: 12.706, 0.01: 63.657},
            2: {0.05: 4.303, 0.01: 9.925},
            5: {0.05: 2.571, 0.01: 4.032},
            10: {0.05: 2.228, 0.01: 3.169},
            20: {0.05: 2.086, 0.01: 2.845},
            30: {0.05: 2.042, 0.01: 2.750},
        }

        # Para df > 30, usar aproximación normal
        if df > 30:
            if alpha == 0.05:
                return 1.96  # Z crítico para 95%
            elif alpha == 0.01:
                return 2.576  # Z crítico para 99%

        # Buscar el df más cercano en la tabla
        df_disponibles = sorted(valores_t.keys())
        df_usar = min(df_disponibles, key=lambda x: abs(x - df))

        return valores_t[df_usar].get(alpha, 1.96)

    @staticmethod
    def intervalo_media(muestra, confianza=0.95):
        """
        Intervalo de confianza para la media poblacional
        Usa t-Student para muestras pequeñas, normal para grandes
        """
        n = len(muestra)
        media_muestral = np.mean(muestra)
        std_muestral = np.std(muestra, ddof=1)  # Corrección de Bessel

        alpha = 1 - confianza

        if n >= 30:
            # Muestra grande: usar distribución normal
            z_critico = IntervalosConfianza.t_student_critico(31, alpha)
            error_estandar = std_muestral / np.sqrt(n)
            margen_error = z_critico * error_estandar
        else:
            # Muestra pequeña: usar t-Student
            df = n - 1
            t_critico = IntervalosConfianza.t_student_critico(df, alpha)
            error_estandar = std_muestral / np.sqrt(n)
            margen_error = t_critico * error_estandar

        return (media_muestral - margen_error, media_muestral + margen_error)

    @staticmethod
    def intervalo_proporcion(exitos, total, confianza=0.95):
        """
        Intervalo de confianza para una proporción
        Usa aproximación normal con corrección de continuidad
        """
        p_hat = exitos / total
        alpha = 1 - confianza
        z_critico = IntervalosConfianza.t_student_critico(31, alpha)

        # Error estándar de la proporción
        error_estandar = np.sqrt(p_hat * (1 - p_hat) / total)
        margen_error = z_critico * error_estandar

        # Corrección de continuidad para muestras pequeñas
        if total < 100:
            margen_error += 0.5 / total

        lower = max(0, p_hat - margen_error)
        upper = min(1, p_hat + margen_error)

        return lower, upper

# Ejemplo: Evaluación de rendimiento de modelo con intervalos
np.random.seed(777)

# Simulamos accuracy de un modelo en diferentes muestras de test
modelo_accuracy_real = 0.85  # Accuracy real (desconocida)

print("=== Intervalos de Confianza para Accuracy ===")

tamaños_muestra = [30, 100, 500, 1000]

for n in tamaños_muestra:
    # Simular resultados de test
    predicciones_correctas = np.random.binomial(n, modelo_accuracy_real)
    accuracy_observada = predicciones_correctas / n

    # Calcular intervalo de confianza
    ic = IntervalosConfianza()
    lower, upper = ic.intervalo_proporcion(predicciones_correctas, n, 0.95)

    # También calculamos con fórmula alternativa para comparar
    intervalo_media = ic.intervalo_media([1]*predicciones_correctas + [0]*(n-predicciones_correctas), 0.95)

    print(f"\nMuestra de {n:4d} casos:")
    print(f"  Accuracy observada: {accuracy_observada:.3f}")
    print(f"  IC 95% (proporción): [{lower:.3f}, {upper:.3f}]")
    print(f"  Ancho del intervalo: {upper - lower:.3f}")
    print(f"  ¿Contiene el real?   {'✓' if lower <= modelo_accuracy_real <= upper else '❌'}")
```

### 4.2 Bootstrap: Intervalos de Confianza Sin Asumir Distribuciones

```python
class Bootstrap:
    """Método de bootstrap para estimación de intervalos de confianza"""

    @staticmethod
    def bootstrap_muestra(datos, n_bootstrap=1000):
        """
        Genera muestras bootstrap

        Args:
            datos: muestra original
            n_bootstrap: número de remuestras
        """
        n = len(datos)
        muestras_boot = []

        for _ in range(n_bootstrap):
            # Muestreo con reemplazo
            muestra_boot = [datos[np.random.randint(0, n)] for _ in range(n)]
            muestras_boot.append(muestra_boot)

        return muestras_boot

    @staticmethod
    def intervalo_percentil(estadisticas_boot, confianza=0.95):
        """
        Intervalo de confianza usando método de percentiles
        """
        alpha = 1 - confianza
        lower_percentil = (alpha / 2) * 100
        upper_percentil = (1 - alpha / 2) * 100

        lower = np.percentile(estadisticas_boot, lower_percentil)
        upper = np.percentile(estadisticas_boot, upper_percentil)

        return lower, upper

    @staticmethod
    def bootstrap_estadistica(datos, estadistica_func, n_bootstrap=1000, confianza=0.95):
        """
        Bootstrap completo para cualquier estadística

        Args:
            datos: muestra original
            estadistica_func: función que calcula la estadística de interés
            n_bootstrap: número de remuestras
            confianza: nivel de confianza
        """
        # Generar muestras bootstrap
        muestras_boot = Bootstrap.bootstrap_muestra(datos, n_bootstrap)

        # Calcular estadística para cada muestra
        estadisticas_boot = [estadistica_func(muestra) for muestra in muestras_boot]

        # Calcular intervalo
        lower, upper = Bootstrap.intervalo_percentil(estadisticas_boot, confianza)

        return {
            'estadisticas_bootstrap': estadisticas_boot,
            'intervalo': (lower, upper),
            'estadistica_original': estadistica_func(datos)
        }

# Ejemplo: Bootstrap para métricas de ML complejas
np.random.seed(888)

# Simulamos predicciones y valores reales de un modelo de regresión
n_test = 200
y_true = np.random.normal(100, 15, n_test)
# Modelo con sesgo y ruido
y_pred = 0.9 * y_true + 5 + np.random.normal(0, 8, n_test)

# Combinamos en un dataset para bootstrap
datos_test = list(zip(y_true, y_pred))

print("=== Bootstrap para Métricas de ML ===")

# Definimos diferentes métricas
def mse(datos):
    """Mean Squared Error"""
    return np.mean([(true - pred)**2 for true, pred in datos])

def mae(datos):
    """Mean Absolute Error"""
    return np.mean([abs(true - pred) for true, pred in datos])

def r_squared(datos):
    """Coeficiente de determinación"""
    y_true_vals = [true for true, pred in datos]
    y_pred_vals = [pred for true, pred in datos]

    y_mean = np.mean(y_true_vals)
    ss_res = sum((true - pred)**2 for true, pred in datos)
    ss_tot = sum((true - y_mean)**2 for true in y_true_vals)

    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

def correlacion_personalizada(datos):
    """Correlación entre predicciones y valores reales"""
    y_true_vals = [true for true, pred in datos]
    y_pred_vals = [pred for true, pred in datos]
    return CorrelacionML.correlacion_pearson(y_true_vals, y_pred_vals)

# Aplicar bootstrap a cada métrica
bootstrap = Bootstrap()
metricas = {
    'MSE': mse,
    'MAE': mae,
    'R²': r_squared,
    'Correlación': correlacion_personalizada
}

resultados_bootstrap = {}

for nombre, func in metricas.items():
    resultado = bootstrap.bootstrap_estadistica(datos_test, func, n_bootstrap=2000)
    resultados_bootstrap[nombre] = resultado

    lower, upper = resultado['intervalo']
    original = resultado['estadistica_original']

    print(f"\n{nombre}:")
    print(f"  Valor observado: {original:.4f}")
    print(f"  IC 95%: [{lower:.4f}, {upper:.4f}]")
    print(f"  Ancho del IC: {upper - lower:.4f}")

# Visualización de las distribuciones bootstrap
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for i, (nombre, resultado) in enumerate(resultados_bootstrap.items()):
    ax = axes[i]
    estadisticas = resultado['estadisticas_bootstrap']
    original = resultado['estadistica_original']
    lower, upper = resultado['intervalo']

    # Histograma de la distribución bootstrap
    ax.hist(estadisticas, bins=50, alpha=0.7, density=True, color='skyblue')

    # Líneas importantes
    ax.axvline(original, color='red', linestyle='-', linewidth=2,
              label=f'Observado: {original:.3f}')
    ax.axvline(lower, color='green', linestyle='--', linewidth=2,
              label=f'IC 95%: [{lower:.3f}, {upper:.3f}]')
    ax.axvline(upper, color='green', linestyle='--', linewidth=2)

    # Área del intervalo de confianza
    x_fill = np.linspace(lower, upper, 100)
    # Aproximar densidad para el área
    hist_vals, bin_edges = np.histogram(estadisticas, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    from scipy.interpolate import interp1d
    f_interp = interp1d(bin_centers, hist_vals, kind='linear',
                       bounds_error=False, fill_value=0)
    y_fill = f_interp(x_fill)
    ax.fill_between(x_fill, y_fill, alpha=0.3, color='green')

    ax.set_xlabel(f'{nombre}')
    ax.set_ylabel('Densidad')
    ax.set_title(f'Distribución Bootstrap: {nombre}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n=== Interpretación de Intervalos ===")
print("• Intervalos estrechos → Estimaciones más precisas")
print("• Intervalos amplios → Mayor incertidumbre")
print("• Bootstrap NO asume distribución específica")
print("• Útil para métricas complejas sin fórmula teórica")
```

## Parte V: Proyecto Práctico - Análisis Estadístico Completo

Ahora vamos a aplicar todos los conceptos en un proyecto real de análisis estadístico para machine learning.

```python
# Proyecto: Análisis estadístico de dataset de precios de casas
# Vamos a generar un dataset sintético realista

class ProyectoEstadisticoML:
    """Proyecto completo de análisis estadístico para ML"""

    def __init__(self):
        self.datos = None
        self.resultados = {}

    def generar_dataset(self, n_casas=1000):
        """
        Genera dataset sintético de precios de casas
        Con relaciones realistas entre variables
        """
        np.random.seed(999)

        # Variables base
        tamaño = np.random.gamma(2, 50, n_casas)  # m², distribución gamma
        antiguedad = np.random.exponential(15, n_casas)  # años, exponencial
        habitaciones = np.random.poisson(3, n_casas) + 1  # count, poisson

        # Variables derivadas con ruido
        ubicacion_score = np.random.beta(2, 2, n_casas)  # 0-1, beta

        # Precio con relaciones realistas + ruido
        precio_base = (
            tamaño * 2000 +                    # $2000 por m²
            (10 - antiguedad) * 1000 +         # depreciación
            habitaciones * 5000 +              # valor por habitación
            ubicacion_score * 50000            # premium por ubicación
        )

        # Añadir ruido realista (heteroscedástico)
        ruido_relativo = 0.1 + 0.05 * ubicacion_score  # más ruido en ubicaciones premium
        ruido = np.random.normal(0, precio_base * ruido_relativo)
        precio = precio_base + ruido

        # Asegurar valores positivos y realistas
        precio = np.maximum(precio, 50000)
        antiguedad = np.minimum(antiguedad, 50)

        self.datos = pd.DataFrame({
            'precio': precio,
            'tamaño': tamaño,
            'antiguedad': antiguedad,
            'habitaciones': habitaciones,
            'ubicacion_score': ubicacion_score
        })

        print(f"Dataset generado: {n_casas} casas con 5 variables")
        return self.datos

    def analisis_descriptivo(self):
        """Análisis estadístico descriptivo completo"""
        print("\n" + "="*50)
        print("ANÁLISIS ESTADÍSTICO DESCRIPTIVO")
        print("="*50)

        calc = EstadisticasML()
        disp = DispersionML()

        resultados = {}

        for columna in self.datos.columns:
            datos = self.datos[columna].values

            # Estadísticas básicas
            media = calc.media(datos)
            mediana = calc.mediana(datos)
            std = disp.desviacion_estandar(datos, poblacional=False)
            iqr, q1, q3 = disp.rango_intercuartil(datos)

            # Coeficiente de variación
            cv = std / media if media != 0 else 0

            # Asimetría (skewness) aproximada
            skewness = (media - mediana) / std if std != 0 else 0

            resultados[columna] = {
                'media': media,
                'mediana': mediana,
                'std': std,
                'cv': cv,
                'iqr': iqr,
                'q1': q1,
                'q3': q3,
                'skewness': skewness,
                'min': min(datos),
                'max': max(datos)
            }

            print(f"\n{columna.upper()}:")
            print(f"  Media: {media:10.2f}  |  Mediana: {mediana:10.2f}")
            print(f"  Std: {std:12.2f}  |  CV: {cv:10.2%}")
            print(f"  IQR: {iqr:12.2f}  |  Asimetría: {skewness:8.2f}")
            print(f"  Rango: [{min(datos):8.1f}, {max(datos):8.1f}]")

        self.resultados['descriptivo'] = resultados
        return resultados

    def analisis_correlaciones(self):
        """Análisis de correlaciones y relaciones"""
        print("\n" + "="*50)
        print("ANÁLISIS DE CORRELACIONES")
        print("="*50)

        corr = CorrelacionML()
        variables = list(self.datos.columns)
        n_vars = len(variables)

        # Matriz de correlación
        matriz_corr = np.zeros((n_vars, n_vars))

        print("\nMatriz de Correlación de Pearson:")
        print(" " * 15, end="")
        for var in variables:
            print(f"{var[:8]:>8}", end="")
        print()

        for i in range(n_vars):
            print(f"{variables[i][:14]:<14}", end=" ")
            for j in range(n_vars):
                if i == j:
                    correlacion = 1.0
                else:
                    datos_i = self.datos[variables[i]].values
                    datos_j = self.datos[variables[j]].values
                    correlacion = corr.correlacion_pearson(datos_i, datos_j)

                matriz_corr[i, j] = correlacion
                print(f"{correlacion:8.3f}", end="")
            print()

        # Identificar correlaciones fuertes
        print("\n=== Correlaciones Significativas (|r| > 0.5) ===")
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                if abs(matriz_corr[i, j]) > 0.5:
                    print(f"{variables[i]} ↔ {variables[j]}: r = {matriz_corr[i, j]:.3f}")

        self.resultados['correlaciones'] = matriz_corr
        return matriz_corr

    def test_normalidad(self):
        """Test de normalidad para cada variable"""
        print("\n" + "="*50)
        print("ANÁLISIS DE NORMALIDAD")
        print("="*50)

        resultados_normalidad = {}

        for columna in self.datos.columns:
            datos = self.datos[columna].values

            # Test visual usando regla 68-95-99.7
            media = np.mean(datos)
            std = np.std(datos)

            # Contar datos en intervalos
            en_1std = np.sum((datos >= media - std) & (datos <= media + std))
            en_2std = np.sum((datos >= media - 2*std) & (datos <= media + 2*std))
            en_3std = np.sum((datos >= media - 3*std) & (datos <= media + 3*std))

            pct_1std = en_1std / len(datos) * 100
            pct_2std = en_2std / len(datos) * 100
            pct_3std = en_3std / len(datos) * 100

            # Determinar normalidad
            normal_68 = abs(pct_1std - 68) < 5
            normal_95 = abs(pct_2std - 95) < 5
            es_normal = normal_68 and normal_95

            print(f"\n{columna.upper()}:")
            print(f"  68% esperado vs real: 68.0% vs {pct_1std:.1f}% {'✓' if normal_68 else '❌'}")
            print(f"  95% esperado vs real: 95.0% vs {pct_2std:.1f}% {'✓' if normal_95 else '❌'}")
            print(f"  99.7% esperado vs real: 99.7% vs {pct_3std:.1f}%")
            print(f"  Conclusión: {'Normal' if es_normal else 'No Normal'}")

            resultados_normalidad[columna] = {
                'es_normal': es_normal,
                'pct_1std': pct_1std,
                'pct_2std': pct_2std,
                'pct_3std': pct_3std
            }

        self.resultados['normalidad'] = resultados_normalidad
        return resultados_normalidad

    def intervalos_confianza_bootstrap(self):
        """Intervalos de confianza usando bootstrap"""
        print("\n" + "="*50)
        print("INTERVALOS DE CONFIANZA (BOOTSTRAP)")
        print("="*50)

        bootstrap = Bootstrap()

        # Definir estadísticas de interés
        def precio_medio(datos_df):
            return datos_df['precio'].mean()

        def precio_por_m2(datos_df):
            return datos_df['precio'].mean() / datos_df['tamaño'].mean()

        def correlacion_precio_tamaño(datos_df):
            return CorrelacionML.correlacion_pearson(
                datos_df['precio'].values,
                datos_df['tamaño'].values
            )

        estadisticas = {
            'Precio Medio': precio_medio,
            'Precio por m²': precio_por_m2,
            'Corr(precio, tamaño)': correlacion_precio_tamaño
        }

        # Bootstrap para cada estadística
        print("Estadística" + " "*15 + "Valor" + " "*8 + "IC 95%")
        print("-" * 60)

        for nombre, func in estadisticas.items():
            # Adaptar bootstrap para DataFrames
            def bootstrap_df_func(datos_list):
                # datos_list es una lista de índices
                sample_df = self.datos.iloc[datos_list].reset_index(drop=True)
                return func(sample_df)

            # Generar índices bootstrap
            n = len(self.datos)
            muestras_indices = []
            for _ in range(1000):
                indices_boot = [np.random.randint(0, n) for _ in range(n)]
                muestras_indices.append(indices_boot)

            # Calcular estadística para cada muestra
            estadisticas_boot = [bootstrap_df_func(indices) for indices in muestras_indices]

            # Calcular intervalo
            lower, upper = Bootstrap.intervalo_percentil(estadisticas_boot)
            original = func(self.datos)

            print(f"{nombre:<25} {original:10.2f}   [{lower:8.2f}, {upper:8.2f}]")

    def visualizaciones_profesionales(self):
        """Crear visualizaciones profesionales del análisis"""
        print("\n" + "="*50)
        print("GENERANDO VISUALIZACIONES...")
        print("="*50)

        # Configuración de estilo
        plt.style.use('default')

        # 1. Histogramas y distribuciones
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        for i, columna in enumerate(self.datos.columns):
            if i < 5:  # Solo las primeras 5 variables
                ax = axes[i]
                datos = self.datos[columna].values

                # Histograma
                ax.hist(datos, bins=50, alpha=0.7, density=True, color='skyblue')

                # Estadísticas
                media = np.mean(datos)
                std = np.std(datos)
                ax.axvline(media, color='red', linestyle='-', linewidth=2, label=f'Media: {media:.1f}')
                ax.axvline(np.median(datos), color='green', linestyle='--', linewidth=2,
                          label=f'Mediana: {np.median(datos):.1f}')

                ax.set_xlabel(columna.replace('_', ' ').title())
                ax.set_ylabel('Densidad')
                ax.set_title(f'Distribución: {columna.replace("_", " ").title()}')
                ax.legend()
                ax.grid(True, alpha=0.3)

        # Ocultar el último subplot si no se usa
        if len(self.datos.columns) == 5:
            axes[5].axis('off')

        plt.suptitle('Distribuciones de Variables', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()

        # 2. Matriz de correlación como heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        matriz = self.resultados.get('correlaciones')
        variables = list(self.datos.columns)

        # Crear heatmap manual
        im = ax.imshow(matriz, cmap='RdBu_r', vmin=-1, vmax=1)

        # Configurar ticks
        ax.set_xticks(range(len(variables)))
        ax.set_yticks(range(len(variables)))
        ax.set_xticklabels([v.replace('_', ' ').title() for v in variables])
        ax.set_yticklabels([v.replace('_', ' ').title() for v in variables])

        # Rotar labels del eje x
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Añadir valores en las celdas
        for i in range(len(variables)):
            for j in range(len(variables)):
                text = ax.text(j, i, f'{matriz[i, j]:.2f}',
                             ha="center", va="center", color="black" if abs(matriz[i, j]) < 0.5 else "white")

        ax.set_title("Matriz de Correlación")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()

        # 3. Scatter plots con líneas de tendencia
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()

        # Relaciones importantes con precio
        relaciones = [
            ('tamaño', 'precio'),
            ('antiguedad', 'precio'),
            ('habitaciones', 'precio'),
            ('ubicacion_score', 'precio')
        ]

        for i, (var_x, var_y) in enumerate(relaciones):
            ax = axes[i]

            x_data = self.datos[var_x].values
            y_data = self.datos[var_y].values

            # Scatter plot
            ax.scatter(x_data, y_data, alpha=0.6, s=20)

            # Línea de tendencia
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            ax.plot(sorted(x_data), p(sorted(x_data)), "r--", alpha=0.8, linewidth=2)

            # Correlación
            corr_val = CorrelacionML.correlacion_pearson(x_data, y_data)
            ax.text(0.05, 0.95, f'r = {corr_val:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax.set_xlabel(var_x.replace('_', ' ').title())
            ax.set_ylabel(var_y.replace('_', ' ').title())
            ax.set_title(f'{var_x.replace("_", " ").title()} vs {var_y.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)

        plt.suptitle('Relaciones entre Variables', fontsize=16)
        plt.tight_layout()
        plt.show()

        print("✓ Visualizaciones completadas")

    def resumen_ejecutivo(self):
        """Resumen ejecutivo con insights para ML"""
        print("\n" + "="*60)
        print("RESUMEN EJECUTIVO PARA MACHINE LEARNING")
        print("="*60)

        print("\n🎯 INSIGHTS CLAVE:")

        # 1. Variables más importantes
        matriz_corr = self.resultados.get('correlaciones')
        if matriz_corr is not None:
            variables = list(self.datos.columns)
            precio_idx = variables.index('precio')

            correlaciones_precio = [(variables[i], abs(matriz_corr[precio_idx, i]))
                                  for i in range(len(variables)) if i != precio_idx]
            correlaciones_precio.sort(key=lambda x: x[1], reverse=True)

            print(f"\n1. VARIABLES MÁS PREDICTIVAS DEL PRECIO:")
            for i, (var, corr) in enumerate(correlaciones_precio[:3], 1):
                print(f"   {i}. {var.replace('_', ' ').title()}: |r| = {corr:.3f}")

        # 2. Calidad de los datos
        normalidad = self.resultados.get('normalidad', {})
        vars_normales = sum(1 for v in normalidad.values() if v['es_normal'])
        total_vars = len(normalidad)

        print(f"\n2. CALIDAD DE LOS DATOS:")
        print(f"   • Variables con distribución normal: {vars_normales}/{total_vars}")
        print(f"   • Outliers detectados: Revisar variables con CV > 50%")

        # 3. Recomendaciones para ML
        print(f"\n3. RECOMENDACIONES PARA ML:")

        descriptivo = self.resultados.get('descriptivo', {})
        for var, stats in descriptivo.items():
            cv = stats.get('cv', 0)
            if cv > 0.5:
                print(f"   ⚠️ {var}: Alta variabilidad (CV={cv:.1%}) - Considerar transformación")
            elif abs(stats.get('skewness', 0)) > 1:
                print(f"   ⚠️ {var}: Distribución asimétrica - Considerar log-transform")

        print(f"\n4. PRÓXIMOS PASOS:")
        print(f"   1. Feature engineering basado en correlaciones fuertes")
        print(f"   2. Transformaciones para variables asimétricas")
        print(f"   3. Detección y tratamiento de outliers")
        print(f"   4. Validación cruzada con estratificación")

        print(f"\n✅ Análisis estadístico completo - Listo para modelado ML")

# Ejecutar proyecto completo
print("INICIANDO PROYECTO DE ANÁLISIS ESTADÍSTICO")
print("="*60)

proyecto = ProyectoEstadisticoML()

# 1. Generar dataset
datos = proyecto.generar_dataset(1000)

# 2. Análisis descriptivo
proyecto.analisis_descriptivo()

# 3. Análisis de correlaciones
proyecto.analisis_correlaciones()

# 4. Test de normalidad
proyecto.test_normalidad()

# 5. Intervalos de confianza
proyecto.intervalos_confianza_bootstrap()

# 6. Visualizaciones
proyecto.visualizaciones_profesionales()

# 7. Resumen ejecutivo
proyecto.resumen_ejecutivo()
```

{{< alert "lightbulb" >}}
**Insight del Proyecto**: Este análisis estadístico completo te proporciona las bases para tomar decisiones informadas sobre preprocesamiento, selección de features y elección de algoritmos de ML.
{{< /alert >}}

## Parte VI: Tests de Hipótesis para Machine Learning

Los tests de hipótesis son fundamentales para validar si las diferencias que observamos en nuestros modelos son estadísticamente significativas.

### 6.1 Test t de Student para Comparación de Modelos

```python
class TestsHipotesisML:
    """Tests de hipótesis aplicados a machine learning"""

    @staticmethod
    def test_t_independiente(muestra1, muestra2, alpha=0.05):
        """
        Test t para muestras independientes
        H0: μ1 = μ2 (no hay diferencia entre medias)
        H1: μ1 ≠ μ2 (hay diferencia significativa)
        """
        n1, n2 = len(muestra1), len(muestra2)
        media1, media2 = np.mean(muestra1), np.mean(muestra2)
        var1, var2 = np.var(muestra1, ddof=1), np.var(muestra2, ddof=1)

        # Pooled standard error
        pooled_se = np.sqrt(var1/n1 + var2/n2)

        # t-statistic
        t_stat = (media1 - media2) / pooled_se

        # Grados de libertad (aproximación de Welch)
        df = ((var1/n1 + var2/n2)**2) / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))

        # Valor crítico (aproximado)
        t_critico = IntervalosConfianza.t_student_critico(int(df), alpha)

        # p-value aproximado (bilateral)
        p_value = 2 * (1 - 0.5 * (1 + np.sign(t_stat) * np.sqrt(t_stat**2 / (t_stat**2 + df))))

        return {
            'estadistico': t_stat,
            'p_value': p_value,
            'grados_libertad': df,
            'valor_critico': t_critico,
            'significativo': abs(t_stat) > t_critico,
            'diferencia_medias': media1 - media2,
            'intervalo_diferencia': ((media1 - media2) - t_critico * pooled_se,
                                   (media1 - media2) + t_critico * pooled_se)
        }

    @staticmethod
    def test_paired_t(antes, despues, alpha=0.05):
        """
        Test t pareado para comparar rendimiento antes/después
        H0: diferencia_media = 0
        H1: diferencia_media ≠ 0
        """
        diferencias = np.array(despues) - np.array(antes)
        n = len(diferencias)
        media_diff = np.mean(diferencias)
        std_diff = np.std(diferencias, ddof=1)
        se_diff = std_diff / np.sqrt(n)

        t_stat = media_diff / se_diff
        df = n - 1
        t_critico = IntervalosConfianza.t_student_critico(df, alpha)

        return {
            'estadistico': t_stat,
            'grados_libertad': df,
            'valor_critico': t_critico,
            'significativo': abs(t_stat) > t_critico,
            'mejora_media': media_diff,
            'intervalo_mejora': (media_diff - t_critico * se_diff,
                               media_diff + t_critico * se_diff)
        }

# Ejemplo: Comparación de dos algoritmos de ML
np.random.seed(123)

# Simulamos accuracy de dos modelos en diferentes folds de validación cruzada
modelo_A_accuracy = np.random.normal(0.82, 0.03, 20)  # Modelo base
modelo_B_accuracy = np.random.normal(0.85, 0.035, 20)  # Modelo mejorado

# Limitamos accuracies a [0, 1]
modelo_A_accuracy = np.clip(modelo_A_accuracy, 0, 1)
modelo_B_accuracy = np.clip(modelo_B_accuracy, 0, 1)

print("=== COMPARACIÓN DE MODELOS CON TEST T ===")
print(f"Modelo A - Media: {np.mean(modelo_A_accuracy):.4f}, Std: {np.std(modelo_A_accuracy, ddof=1):.4f}")
print(f"Modelo B - Media: {np.mean(modelo_B_accuracy):.4f}, Std: {np.std(modelo_B_accuracy, ddof=1):.4f}")

# Test t independiente
tests = TestsHipotesisML()
resultado = tests.test_t_independiente(modelo_A_accuracy, modelo_B_accuracy)

print(f"\n=== RESULTADOS DEL TEST T ===")
print(f"Estadístico t: {resultado['estadistico']:.3f}")
print(f"Grados de libertad: {resultado['grados_libertad']:.1f}")
print(f"Valor crítico (α=0.05): ±{resultado['valor_critico']:.3f}")
print(f"p-value: {resultado['p_value']:.4f}")
print(f"Diferencia de medias: {resultado['diferencia_medias']:.4f}")
print(f"IC 95% diferencia: [{resultado['intervalo_diferencia'][0]:.4f}, {resultado['intervalo_diferencia'][1]:.4f}]")

if resultado['significativo']:
    print(f"\n✅ CONCLUSIÓN: Diferencia estadísticamente significativa")
    print(f"   El Modelo B es significativamente mejor que el Modelo A")
else:
    print(f"\n❌ CONCLUSIÓN: No hay diferencia estadísticamente significativa")

# Visualización de la comparación
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Boxplot comparativo
ax1.boxplot([modelo_A_accuracy, modelo_B_accuracy], labels=['Modelo A', 'Modelo B'])
ax1.set_ylabel('Accuracy')
ax1.set_title('Comparación de Modelos\n(Boxplot)')
ax1.grid(True, alpha=0.3)

# Plot 2: Histogramas superpuestos
ax2.hist(modelo_A_accuracy, bins=15, alpha=0.7, label='Modelo A', color='blue', density=True)
ax2.hist(modelo_B_accuracy, bins=15, alpha=0.7, label='Modelo B', color='red', density=True)
ax2.axvline(np.mean(modelo_A_accuracy), color='blue', linestyle='--', linewidth=2)
ax2.axvline(np.mean(modelo_B_accuracy), color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Accuracy')
ax2.set_ylabel('Densidad')
ax2.set_title('Distribución de Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Diferencias pareadas (si fuera paired test)
diferencias = modelo_B_accuracy - modelo_A_accuracy
ax3.hist(diferencias, bins=15, alpha=0.7, color='green')
ax3.axvline(np.mean(diferencias), color='darkgreen', linestyle='-', linewidth=2,
           label=f'Diferencia media: {np.mean(diferencias):.4f}')
ax3.axvline(0, color='black', linestyle='--', linewidth=1, label='Sin diferencia')
ax3.set_xlabel('Diferencia de Accuracy (B - A)')
ax3.set_ylabel('Frecuencia')
ax3.set_title('Distribución de Diferencias')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Ejemplo de test pareado: antes/después de feature engineering
print("\n" + "="*60)
print("TEST PAREADO: ANTES/DESPUÉS DE FEATURE ENGINEERING")
print("="*60)

# Simulamos el mismo modelo evaluado antes y después de feature engineering
np.random.seed(456)
accuracy_antes = np.random.normal(0.78, 0.04, 15)
# Después del feature engineering: mejora correlacionada
mejora = np.random.normal(0.04, 0.015, 15)  # Mejora promedio de 4%
accuracy_despues = accuracy_antes + mejora

# Clip to valid range
accuracy_antes = np.clip(accuracy_antes, 0, 1)
accuracy_despues = np.clip(accuracy_despues, 0, 1)

print(f"Antes FE - Media: {np.mean(accuracy_antes):.4f}")
print(f"Después FE - Media: {np.mean(accuracy_despues):.4f}")
print(f"Mejora promedio: {np.mean(accuracy_despues - accuracy_antes):.4f}")

resultado_paired = tests.test_paired_t(accuracy_antes, accuracy_despues)

print(f"\n=== RESULTADOS DEL TEST T PAREADO ===")
print(f"Estadístico t: {resultado_paired['estadistico']:.3f}")
print(f"Valor crítico: ±{resultado_paired['valor_critico']:.3f}")
print(f"Mejora media: {resultado_paired['mejora_media']:.4f}")
print(f"IC 95% mejora: [{resultado_paired['intervalo_mejora'][0]:.4f}, {resultado_paired['intervalo_mejora'][1]:.4f}]")

if resultado_paired['significativo']:
    print(f"\n✅ El feature engineering produjo una mejora significativa")
else:
    print(f"\n❌ La mejora no es estadísticamente significativa")
```

### 6.2 ANOVA para Comparación Múltiple de Modelos

```python
class ANOVAMultiple:
    """ANOVA para comparar múltiples modelos simultáneamente"""

    @staticmethod
    def anova_one_way(grupos):
        """
        ANOVA de un factor para comparar múltiples grupos
        H0: μ1 = μ2 = ... = μk (todas las medias son iguales)
        H1: Al menos una media es diferente
        """
        # Calcular estadísticas por grupo
        k = len(grupos)  # número de grupos
        n_total = sum(len(grupo) for grupo in grupos)

        # Media global
        todos_valores = np.concatenate(grupos)
        media_global = np.mean(todos_valores)

        # Suma de cuadrados entre grupos (SSB)
        ssb = sum(len(grupo) * (np.mean(grupo) - media_global)**2 for grupo in grupos)

        # Suma de cuadrados dentro de grupos (SSW)
        ssw = sum(np.sum((grupo - np.mean(grupo))**2) for grupo in grupos)

        # Grados de libertad
        df_between = k - 1
        df_within = n_total - k
        df_total = n_total - 1

        # Cuadrados medios
        msb = ssb / df_between
        msw = ssw / df_within

        # Estadístico F
        f_stat = msb / msw if msw > 0 else float('inf')

        # Valor crítico F (aproximación)
        # Para α = 0.05, usamos aproximaciones comunes
        if df_between <= 3 and df_within >= 10:
            f_critico = 3.0  # Aproximación conservadora
        else:
            f_critico = 2.5  # Aproximación más liberal

        return {
            'f_estadistico': f_stat,
            'f_critico': f_critico,
            'df_between': df_between,
            'df_within': df_within,
            'ssb': ssb,
            'ssw': ssw,
            'msb': msb,
            'msw': msw,
            'significativo': f_stat > f_critico
        }

    @staticmethod
    def test_post_hoc_tukey(grupos, nombres_grupos):
        """
        Test post-hoc de Tukey para comparaciones múltiples
        (Versión simplificada)
        """
        k = len(grupos)
        comparaciones = []

        # MSW del ANOVA
        anova_result = ANOVAMultiple.anova_one_way(grupos)
        msw = anova_result['msw']

        # Comparaciones por pares
        for i in range(k):
            for j in range(i+1, k):
                n1, n2 = len(grupos[i]), len(grupos[j])
                media1, media2 = np.mean(grupos[i]), np.mean(grupos[j])

                # Error estándar para la diferencia
                se_diff = np.sqrt(msw * (1/n1 + 1/n2))

                # Estadístico Q (distribución estudentizada)
                q_stat = abs(media1 - media2) / se_diff

                # Valor crítico Q (aproximación)
                q_critico = 3.5  # Para k=3-5 grupos, α=0.05

                comparaciones.append({
                    'grupos': f"{nombres_grupos[i]} vs {nombres_grupos[j]}",
                    'diferencia': media1 - media2,
                    'q_estadistico': q_stat,
                    'q_critico': q_critico,
                    'significativo': q_stat > q_critico
                })

        return comparaciones

# Ejemplo: Comparación de múltiples algoritmos
np.random.seed(789)

# Simulamos performance de 4 algoritmos diferentes
algoritmos = {
    'Random Forest': np.random.normal(0.85, 0.03, 20),
    'XGBoost': np.random.normal(0.87, 0.025, 20),
    'SVM': np.random.normal(0.82, 0.04, 20),
    'Logistic Regression': np.random.normal(0.79, 0.035, 20)
}

# Clip to valid accuracy range
for nombre in algoritmos:
    algoritmos[nombre] = np.clip(algoritmos[nombre], 0, 1)

print("=== COMPARACIÓN MÚLTIPLE DE ALGORITMOS (ANOVA) ===")

# Mostrar estadísticas descriptivas
for nombre, scores in algoritmos.items():
    print(f"{nombre:20}: Media = {np.mean(scores):.4f}, Std = {np.std(scores, ddof=1):.4f}")

# ANOVA
grupos = list(algoritmos.values())
nombres = list(algoritmos.keys())

anova = ANOVAMultiple()
resultado_anova = anova.anova_one_way(grupos)

print(f"\n=== RESULTADOS ANOVA ===")
print(f"F-estadístico: {resultado_anova['f_estadistico']:.3f}")
print(f"F-crítico: {resultado_anova['f_critico']:.3f}")
print(f"Grados de libertad: between={resultado_anova['df_between']}, within={resultado_anova['df_within']}")

if resultado_anova['significativo']:
    print(f"✅ Hay diferencias significativas entre algoritmos")

    # Test post-hoc
    print(f"\n=== TEST POST-HOC (Tukey) ===")
    comparaciones = anova.test_post_hoc_tukey(grupos, nombres)

    for comp in comparaciones:
        significativo_symbol = "✅" if comp['significativo'] else "❌"
        print(f"{comp['grupos']:35}: Diff={comp['diferencia']:7.4f}, Q={comp['q_estadistico']:6.2f} {significativo_symbol}")

else:
    print(f"❌ No hay diferencias significativas entre algoritmos")

# Visualización ANOVA
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Boxplot de todos los algoritmos
ax1.boxplot(grupos, labels=[n[:8] for n in nombres])  # Truncar nombres largos
ax1.set_ylabel('Accuracy')
ax1.set_title('Comparación de Algoritmos\n(ANOVA)')
ax1.grid(True, alpha=0.3)
plt.setp(ax1.get_xticklabels(), rotation=45)

# Gráfico de medias con intervalos de confianza
medias = [np.mean(grupo) for grupo in grupos]
stds = [np.std(grupo, ddof=1) for grupo in grupos]
errors = [std/np.sqrt(len(grupo)) for std, grupo in zip(stds, grupos)]

x_pos = range(len(nombres))
ax2.bar(x_pos, medias, yerr=errors, capsize=5, alpha=0.7,
        color=['blue', 'red', 'green', 'orange'])
ax2.set_xticks(x_pos)
ax2.set_xticklabels([n[:8] for n in nombres], rotation=45)
ax2.set_ylabel('Accuracy Media')
ax2.set_title('Medias con Intervalos de Confianza')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Parte VII: Detección de Outliers Estadística

Los outliers pueden distorsionar nuestros modelos de ML. Aprendamos métodos estadísticos para detectarlos.

```python
class DeteccionOutliers:
    """Métodos estadísticos para detección de outliers en ML"""

    @staticmethod
    def metodo_iqr(datos, factor=1.5):
        """
        Método del rango intercuartil (IQR)
        Outliers: < Q1 - factor*IQR o > Q3 + factor*IQR
        """
        q1 = np.percentile(datos, 25)
        q3 = np.percentile(datos, 75)
        iqr = q3 - q1

        limite_inferior = q1 - factor * iqr
        limite_superior = q3 + factor * iqr

        outliers_mask = (datos < limite_inferior) | (datos > limite_superior)
        outliers = datos[outliers_mask]

        return {
            'outliers': outliers,
            'indices_outliers': np.where(outliers_mask)[0],
            'limite_inferior': limite_inferior,
            'limite_superior': limite_superior,
            'n_outliers': len(outliers),
            'porcentaje_outliers': len(outliers) / len(datos) * 100
        }

    @staticmethod
    def metodo_zscore(datos, umbral=3):
        """
        Método del Z-score
        Outliers: |z| > umbral (típicamente 3)
        """
        media = np.mean(datos)
        std = np.std(datos)

        z_scores = np.abs((datos - media) / std)
        outliers_mask = z_scores > umbral

        return {
            'z_scores': z_scores,
            'outliers': datos[outliers_mask],
            'indices_outliers': np.where(outliers_mask)[0],
            'umbral': umbral,
            'n_outliers': np.sum(outliers_mask),
            'porcentaje_outliers': np.sum(outliers_mask) / len(datos) * 100
        }

    @staticmethod
    def metodo_zscore_modificado(datos, umbral=3.5):
        """
        Z-score modificado usando mediana (más robusto)
        Usa MAD (Median Absolute Deviation) en lugar de std
        """
        mediana = np.median(datos)
        mad = np.median(np.abs(datos - mediana))

        # Factor de escala para aproximar desviación estándar
        mad_scaled = mad * 1.4826

        z_scores_mod = np.abs((datos - mediana) / mad_scaled)
        outliers_mask = z_scores_mod > umbral

        return {
            'z_scores_modificados': z_scores_mod,
            'outliers': datos[outliers_mask],
            'indices_outliers': np.where(outliers_mask)[0],
            'mediana': mediana,
            'mad': mad,
            'n_outliers': np.sum(outliers_mask),
            'porcentaje_outliers': np.sum(outliers_mask) / len(datos) * 100
        }

    @staticmethod
    def metodo_isolation_forest_simple(datos, contaminacion=0.1):
        """
        Versión simplificada de Isolation Forest
        Basada en profundidad promedio de aislamiento
        """
        n = len(datos)
        profundidades = []

        # Simulamos múltiples "árboles" de aislamiento
        n_trees = 100
        for _ in range(n_trees):
            # Para cada punto, simulamos profundidad de aislamiento
            profundidades_arbol = []
            datos_copia = datos.copy()

            for punto in datos:
                profundidad = 0
                datos_temp = datos_copia.copy()

                # Simular divisiones hasta aislar el punto
                while len(datos_temp) > 1 and profundidad < 20:
                    # División aleatoria
                    min_val, max_val = np.min(datos_temp), np.max(datos_temp)
                    if max_val == min_val:
                        break

                    split_val = np.random.uniform(min_val, max_val)

                    # Determinar en qué lado queda nuestro punto
                    if punto <= split_val:
                        datos_temp = datos_temp[datos_temp <= split_val]
                    else:
                        datos_temp = datos_temp[datos_temp > split_val]

                    profundidad += 1

                profundidades_arbol.append(profundidad)

            profundidades.append(profundidades_arbol)

        # Promedio de profundidades
        prof_promedio = np.mean(profundidades, axis=0)

        # Puntos con menor profundidad son más anómalos
        umbral_prof = np.percentile(prof_promedio, contaminacion * 100)
        outliers_mask = prof_promedio <= umbral_prof

        return {
            'profundidades': prof_promedio,
            'outliers': datos[outliers_mask],
            'indices_outliers': np.where(outliers_mask)[0],
            'umbral_profundidad': umbral_prof,
            'n_outliers': np.sum(outliers_mask),
            'porcentaje_outliers': np.sum(outliers_mask) / len(datos) * 100
        }

# Ejemplo: Detección de outliers en datos de latencia de API
np.random.seed(999)

# Generamos datos de latencia con outliers realistas
latencia_normal = np.random.gamma(2, 50, 900)  # Latencias normales
latencia_outliers_red = np.random.exponential(500, 80)  # Problemas de red
latencia_outliers_server = np.random.uniform(1000, 2000, 20)  # Problemas de servidor

latencias = np.concatenate([latencia_normal, latencia_outliers_red, latencia_outliers_server])
np.random.shuffle(latencias)  # Mezclar para simular orden temporal

print("=== DETECCIÓN DE OUTLIERS EN LATENCIAS DE API ===")
print(f"Total de muestras: {len(latencias)}")
print(f"Latencia media: {np.mean(latencias):.2f} ms")
print(f"Latencia mediana: {np.median(latencias):.2f} ms")

detector = DeteccionOutliers()

# Aplicar diferentes métodos
metodos = {
    'IQR (factor=1.5)': detector.metodo_iqr(latencias, 1.5),
    'Z-score (umbral=3)': detector.metodo_zscore(latencias, 3),
    'Z-score Modificado': detector.metodo_zscore_modificado(latencias, 3.5),
    'Isolation Forest': detector.metodo_isolation_forest_simple(latencias, 0.1)
}

print(f"\n=== RESULTADOS POR MÉTODO ===")
for nombre, resultado in metodos.items():
    print(f"\n{nombre}:")
    print(f"  Outliers detectados: {resultado['n_outliers']} ({resultado['porcentaje_outliers']:.1f}%)")
    if len(resultado['outliers']) > 0:
        print(f"  Rango outliers: [{np.min(resultado['outliers']):.1f}, {np.max(resultado['outliers']):.1f}] ms")

# Visualización comparativa
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

metodos_lista = list(metodos.items())
for i, (nombre, resultado) in enumerate(metodos_lista):
    ax = axes[i]

    # Histograma con outliers marcados
    ax.hist(latencias, bins=50, alpha=0.7, color='lightblue', label='Datos normales')

    if len(resultado['outliers']) > 0:
        ax.hist(resultado['outliers'], bins=20, alpha=0.8, color='red',
               label=f'Outliers ({resultado["n_outliers"]})')

    ax.set_xlabel('Latencia (ms)')
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'{nombre}\n{resultado["n_outliers"]} outliers ({resultado["porcentaje_outliers"]:.1f}%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Análisis de consenso entre métodos
print(f"\n=== ANÁLISIS DE CONSENSO ===")

# Crear matriz de detección
indices_todos = set(range(len(latencias)))
detecciones = {}

for nombre, resultado in metodos.items():
    detecciones[nombre] = set(resultado['indices_outliers'])

# Intersección (outliers detectados por múltiples métodos)
interseccion_todos = set.intersection(*detecciones.values())
union_todos = set.union(*detecciones.values())

print(f"Outliers detectados por TODOS los métodos: {len(interseccion_todos)}")
print(f"Outliers detectados por AL MENOS UN método: {len(union_todos)}")

if len(interseccion_todos) > 0:
    valores_consenso = latencias[list(interseccion_todos)]
    print(f"Valores de consenso: {sorted(valores_consenso)[:5]}... (primeros 5)")

# Matriz de consenso
print(f"\n=== MATRIZ DE CONSENSO ===")
nombres_metodos = list(metodos.keys())
for i, metodo1 in enumerate(nombres_metodos):
    for j, metodo2 in enumerate(nombres_metodos):
        if i <= j:
            if i == j:
                overlap = len(detecciones[metodo1])
            else:
                overlap = len(detecciones[metodo1].intersection(detecciones[metodo2]))
            print(f"{metodo1[:10]} ∩ {metodo2[:10]}: {overlap:3d} outliers")

print(f"\n💡 RECOMENDACIÓN:")
if len(interseccion_todos) > 0:
    print(f"   • Eliminar outliers detectados por múltiples métodos ({len(interseccion_todos)} casos)")
    print(f"   • Investigar outliers detectados por un solo método")
else:
    print(f"   • No hay consenso fuerte - revisar umbrales")
    print(f"   • Considerar contexto del dominio para decisión final")
```

## Resumen y Conexión con la Semana 4

### Conceptos Clave Dominados

Has completado un recorrido exhaustivo por los fundamentos estadísticos de machine learning:

{{< mermaid >}}
graph LR
    A[Estadística Descriptiva] --> B[Distribuciones]
    B --> C[Correlación vs Causación]
    C --> D[Teorema de Bayes]
    D --> E[Intervalos de Confianza]
    E --> F[Tests de Hipótesis]
    F --> G[Detección de Outliers]
    G --> H[Proyecto Completo]

    style A fill:#e1f5fe
    style H fill:#c8e6c9
{{< /mermaid >}}

### Checklist de Competencias ✅

- **Estadística Descriptiva**: Calcular e interpretar medidas de tendencia central y dispersión
- **Distribuciones de Probabilidad**: Aplicar distribuciones normal, binomial y uniforme en contextos de ML
- **Correlación vs Causación**: Identificar y evitar interpretaciones erróneas
- **Teorema de Bayes**: Implementar clasificación bayesiana y actualización de creencias
- **Intervalos de Confianza**: Cuantificar incertidumbre con métodos paramétricos y bootstrap
- **Tests de Hipótesis**: Comparar modelos con rigor estadístico
- **Detección de Outliers**: Aplicar múltiples métodos para identificar anomalías

### Preparación para la Semana 4: NumPy y Pandas

Los conceptos estadísticos que dominaste hoy son los **cimientos** para la manipulación eficiente de datos con NumPy y Pandas:

{{< alert "circle-info" >}}
**Próxima Semana**: Implementarás estos cálculos estadísticos de manera vectorizada y escalable usando las librerías fundamentales de data science en Python.
{{< /alert >}}

**Lo que conectarás la próxima semana:**

1. **Estadísticas con NumPy**: Vectorización de todos los cálculos que hiciste hoy
2. **Análisis exploratorio con Pandas**: Aplicar tests estadísticos a datasets reales
3. **Visualización estadística**: Crear gráficos profesionales para análisis
4. **Pipelines de preprocesamiento**: Integrar detección de outliers y transformaciones

### Ejercicios para Consolidar

{{< alert "lightbulb" >}}
**Desafío Semanal**: Antes de la próxima sesión, aplica el análisis estadístico completo que desarrollaste a un dataset público de tu elección. Algunos datasets recomendados:

- [Kaggle: House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- [UCI: Adult Income](https://archive.ics.uci.edu/ml/datasets/adult)
- [Seaborn: Tips Dataset](https://github.com/mwaskom/seaborn-data)

**Objetivo**: Generar un reporte estadístico completo que incluya todos los elementos que vimos hoy.
{{< /alert >}}

### Recursos Adicionales

Para profundizar en los conceptos estadísticos aplicados a machine learning:

{{< alert "book-open" >}}
**Lecturas Recomendadas**:

- "The Elements of Statistical Learning" - Hastie, Tibshirani & Friedman
- "Pattern Recognition and Machine Learning" - Christopher Bishop
- "Think Stats" - Allen B. Downey (enfoque práctico con Python)

**Documentación Técnica**:
- [SciPy Stats Documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [NumPy Statistical Functions](https://numpy.org/doc/stable/reference/routines.statistics.html)
{{< /alert >}}

¡Felicitaciones! Has construido una base sólida en estadística aplicada a machine learning. En la próxima semana, estas herramientas conceptuales se transformarán en código eficiente y escalable con las librerías fundamentales del ecosistema científico de Python.

---

*¿Preguntas sobre algún concepto estadístico? ¿Dudas sobre la implementación de algún test? Recordá que la estadística es la base de la toma de decisiones informadas en machine learning - cada concepto que dominaste hoy te ahorrará errores costosos en producción.*
