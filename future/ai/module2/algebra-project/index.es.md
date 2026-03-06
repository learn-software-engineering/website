<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
## Contents

- [series: ["Matemática para Machine Learning"]](#series-matem%C3%A1tica-para-machine-learning)
- [series_order: 2](#series_order-2)
  - [date: 2025-09-14
tags: ["Inteligencia Artificial", "Aprendizaje Automático", "Machine Learning", "Matemática", "Álgebra Lineal", "Vectores", "Matrices", "Recomendaciones"]](#date-2025-09-14%0Atags-inteligencia-artificial-aprendizaje-autom%C3%A1tico-machine-learning-matem%C3%A1tica-%C3%A1lgebra-lineal-vectores-matrices-recomendaciones)
  - [Objetivo del proyecto](#objetivo-del-proyecto)
    - [Datos disponibles](#datos-disponibles)
    - [Implementación Completa](#implementaci%C3%B3n-completa)
  - [Próximos pasos: optimizaciones y cálculo](#pr%C3%B3ximos-pasos-optimizaciones-y-c%C3%A1lculo)
  - [Reflexión final](#reflexi%C3%B3n-final)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

---
weight: 2
# series: ["Matemática para Machine Learning"]
# series_order: 2
title: "Proyecto usando Álgebra Lineal: Sistema de Recomendaciones"
description: "Para consolidar los conceptos aprendidos, implementaremos un sistema de recomendaciones completo usando solo álgebra lineal. Este proyecto demuestra cómo el álgebra lineal es fundamental en sistemas de machine learning reales."
authors:
  - jnonino
date: 2025-09-14
tags: ["Inteligencia Artificial", "Aprendizaje Automático", "Machine Learning", "Matemática", "Álgebra Lineal", "Vectores", "Matrices", "Recomendaciones"]
---

Para consolidar los conceptos aprendidos, implementaremos un sistema de recomendaciones completo usando solo álgebra lineal. Este proyecto demuestra cómo el álgebra lineal es fundamental en sistemas de machine learning reales.

---

## Objetivo del proyecto

Crear un sistema de recomendaciones que:
1. Represente usuarios y productos como vectores
2. Calcule similitudes entre usuarios usando productos punto
3. Haga recomendaciones de películas no vistas basadas en usuarios similares

### Datos disponibles

Para el desarrollo del sistema tenemos preparados un conjunto de datos que podés usar para verificar el funcionamiento de tu código.

**Usuarios**: [*datos/usuarios.json*](https://github.com/learn-software-engineering/examples/blob/main/ai/module2/algebra-project/datos/usuarios.json)
- *Nombre del usuario*

**Productos**: [*datos/peliculas.json*](https://github.com/learn-software-engineering/examples/blob/main/ai/module2/algebra-project/datos/peliculas.json)
- *Nombre de la película*
- *Género de la película*

**Interacciones**: [*datos/interacciones.json*](https://github.com/learn-software-engineering/examples/blob/main/ai/module2/algebra-project/datos/interacciones.json)
- *ID de la interacción*
- *ID del usuario*
- *ID de la película*
- *Valoración*: puntuación de 1-5 estrellas

### Implementación Completa

Haciendo click en el siguiente [enlace](https://github.com/learn-software-engineering/examples/tree/main/ai/module2/algebra-project), puedes encontrar una posible implementación que cumple con los objetivos planteados. Incluye explicaciones detalladas.

{{< callout type="important" >}}
¡Intentá resolverlo por tu cuenta primero! 😀😀😀
{{< /callout >}}

Al ejecutar el sistema obtenemos:

```bash
>  python main.py
===========================================
=== Prueba del Sistema de Recomendación ===
===========================================
Inicializando el sistema
   * 13 peliculas cargadas
   * 6 usuarios cargados
   * 45 interacciones cargadas
   Procesando interacciones...
   Vectores de usuarios actualizados
   Matriz de puntuaciones construida: (6, 13)
   Calculando similitudes entre 6 usuarios
   Matriz de similitud entre usuarios construida: (6, 6)
Sistema de recomendaciones inicializado correctamente

****************************
Reporte completo del sistema
****************************
Información del conjunto de datos:
   * Usuarios: 6
   * Películas: 13
   * Interacciones totales: 45
   * Calificaciones dadas: 45 de 78
   * Sparsity (densidad): 42.3%

Análisis de los usuario
   * Ana López (ID: usuario_001):
      - Películas vistas: 6
      - Puntuación promedio: 3.50
   * Carlos Mendez (ID: usuario_002):
      - Películas vistas: 7
      - Puntuación promedio: 3.57
   * María Rodriguez (ID: usuario_003):
      - Películas vistas: 8
      - Puntuación promedio: 3.00
   * Diego Silva (ID: usuario_004):
      - Películas vistas: 8
      - Puntuación promedio: 2.88
   * Laura Martínez (ID: usuario_005):
      - Películas vistas: 7
      - Puntuación promedio: 3.71
   * Roberto García (ID: usuario_006):
      - Películas vistas: 9
      - Puntuación promedio: 2.33

Matriz de puntuaciones (6, 13):
+--------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+
|        |  Duro d  |  Termin  |  Matrix  |  El exo  |  Pesadi  |  El res  |  Mi pob  |  Forres  |  ¿Qué p  |  La más  |   Eso    |  Hallow  |  El ori  |
+--------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+
| Ana Ló |      5.0 |      4.0 |      0.0 |      2.0 |      1.0 |      0.0 |      4.0 |      5.0 |      0.0 |      0.0 |      0.0 |      0.0 |      0.0 |
| Carlos |      4.0 |      5.0 |      4.0 |      3.0 |      2.0 |      0.0 |      3.0 |      4.0 |      0.0 |      0.0 |      0.0 |      0.0 |      0.0 |
| María  |      1.0 |      2.0 |      0.0 |      5.0 |      4.0 |      5.0 |      2.0 |      1.0 |      4.0 |      0.0 |      0.0 |      0.0 |      0.0 |
| Diego  |      2.0 |      1.0 |      2.0 |      1.0 |      0.0 |      3.0 |      5.0 |      4.0 |      5.0 |      0.0 |      0.0 |      0.0 |      0.0 |
| Laura  |      4.0 |      3.0 |      0.0 |      4.0 |      4.0 |      0.0 |      3.0 |      3.0 |      5.0 |      0.0 |      0.0 |      0.0 |      0.0 |
| Robert |      2.0 |      5.0 |      1.0 |      3.0 |      4.0 |      1.0 |      2.0 |      2.0 |      1.0 |      0.0 |      0.0 |      0.0 |      0.0 |
+--------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+

Matriz de similitud entre usuarios (6, 6):
+-----------------+-----------+---------------+-----------------+-------------+----------------+----------------+
|                 | Ana López | Carlos Mendez | María Rodriguez | Diego Silva | Laura Martínez | Roberto García |
+-----------------+-----------+---------------+-----------------+-------------+----------------+----------------+
|    Ana López    |     1.000 |       0.880   |        0.447    |      0.651  |       0.761    |       0.771    |
|  Carlos Mendez  |     0.880 |       1.000   |        0.503    |      0.612  |       0.739    |       0.865    |
| María Rodriguez |     0.447 |       0.503   |        1.000    |      0.656  |       0.782    |       0.750    |
|   Diego Silva   |     0.651 |       0.612   |        0.656    |      1.000  |       0.727    |       0.538    |
|  Laura Martínez |     0.761 |       0.739   |        0.782    |      0.727  |       1.000    |       0.843    |
|  Roberto García |     0.771 |       0.865   |        0.750    |      0.538  |       0.843    |       1.000    |
+-----------------+-----------+---------------+-----------------+-------------+----------------+----------------+

***************************************
Hacer recomendaciones para los usuarios
***************************************
   Recomendaciones para Ana López:
      Películas ya vistas: 6/13
      Predicción para usuario_001 - 'Matrix': 2.43 - Basada en 3 usuarios similares
      Predicción para usuario_001 - 'El resplandor': 2.65 - Basada en 3 usuarios similares
      Predicción para usuario_001 - '¿Qué pasó ayer?': 3.66 - Basada en 4 usuarios similares
      No hay usuarios similares que hayan visto 'La máscara'
      No hay usuarios similares que hayan visto 'Eso'
      No hay usuarios similares que hayan visto 'Halloween'
      No hay usuarios similares que hayan visto 'El origen'
      3 recomendaciones generadas
      1- ¿Qué pasó ayer? - Predicción de puntuación 3.66
      2- El resplandor - Predicción de puntuación 2.65
      3- Matrix - Predicción de puntuación 2.43
   Recomendaciones para Carlos Mendez:
      Películas ya vistas: 7/13
      Predicción para usuario_002 - 'El resplandor': 2.63 - Basada en 3 usuarios similares
      Predicción para usuario_002 - '¿Qué pasó ayer?': 3.54 - Basada en 4 usuarios similares
      No hay usuarios similares que hayan visto 'La máscara'
      No hay usuarios similares que hayan visto 'Eso'
      No hay usuarios similares que hayan visto 'Halloween'
      No hay usuarios similares que hayan visto 'El origen'
      2 recomendaciones generadas
      1- ¿Qué pasó ayer? - Predicción de puntuación 3.54
      2- El resplandor - Predicción de puntuación 2.63
   Recomendaciones para María Rodriguez:
      Películas ya vistas: 8/13
      Predicción para usuario_003 - 'Matrix': 2.13 - Basada en 3 usuarios similares
      No hay usuarios similares que hayan visto 'La máscara'
      No hay usuarios similares que hayan visto 'Eso'
      No hay usuarios similares que hayan visto 'Halloween'
      No hay usuarios similares que hayan visto 'El origen'
      1 recomendaciones generadas
      1- Matrix - Predicción de puntuación 2.13
   Recomendaciones para Diego Silva:
      Películas ya vistas: 8/13
      Predicción para usuario_004 - 'Pesadilla en la calle Elm': 3.00 - Basada en 5 usuarios similares
      No hay usuarios similares que hayan visto 'La máscara'
      No hay usuarios similares que hayan visto 'Eso'
      No hay usuarios similares que hayan visto 'Halloween'
      No hay usuarios similares que hayan visto 'El origen'
      1 recomendaciones generadas
      1- Pesadilla en la calle Elm - Predicción de puntuación 3.00
   Recomendaciones para Laura Martínez:
      Películas ya vistas: 7/13
      Predicción para usuario_005 - 'Matrix': 2.27 - Basada en 3 usuarios similares
      Predicción para usuario_005 - 'El resplandor': 2.95 - Basada en 3 usuarios similares
      No hay usuarios similares que hayan visto 'La máscara'
      No hay usuarios similares que hayan visto 'Eso'
      No hay usuarios similares que hayan visto 'Halloween'
      No hay usuarios similares que hayan visto 'El origen'
      2 recomendaciones generadas
      1- El resplandor - Predicción de puntuación 2.95
      2- Matrix - Predicción de puntuación 2.27
   Recomendaciones para Roberto García:
      Películas ya vistas: 9/13
      No hay usuarios similares que hayan visto 'La máscara'
      No hay usuarios similares que hayan visto 'Eso'
      No hay usuarios similares que hayan visto 'Halloween'
      No hay usuarios similares que hayan visto 'El origen'
      0 recomendaciones generadas
      - No hay recomendaciones disponibles
```

---

## Próximos pasos: optimizaciones y cálculo

El álgebra lineal que acabas de dominar es la base computacional del Machine Learning. En el próximo artículo exploraremos el **cálculo**. Allí podrás:

- **Entender intuitivamente** qué es una derivada y por qué es tan poderosa
- **Calcular gradientes** de funciones de múltiples variables
- **Implementar descenso por gradiente desde cero** en Python
- **Visualizar** cómo los algoritmos *"aprenden"* navegando funciones de costo
- **Optimizar** modelos de Machine Learning usando estos conceptos

Y lo más importante: vas a **ver** el Machine Learning de una forma completamente nueva, entendiendo el motor matemático que lo impulsa.

---

## Reflexión final

Hasta aquí aprendiste:

- ✅ **Conceptos fundamentales del Álgebra Lineal**
  - Vectores como representación de datos
  - Matrices como transformaciones
  - Espacios vectoriales y transformaciones lineales
  - Valores y vectores propios
- ✅ **Operaciones clave**
  - Producto punto para similitud
  - Multiplicación matriz-vector para transformaciones
- ✅ **Aplicaciones prácticas**
  - Sistema de recomendación completo
- ✅ **Herramientas**
  - Implementación desde cero para comprensión
  - NumPy para optimización
  - Visualización de conceptos

**Para recordar**

1. **El álgebra lineal es el lenguaje del ML**: Todo se reduce a vectores y matrices
2. **Las operaciones tienen significado**: No son solo cálculos, representan conceptos
3. **La geometría importa**: Visualizar ayuda a entender
4. **La optimización es crucial**: NumPy vs implementación casera
5. **La teoría guía la práctica**: entender el *"por qué"* antes del *"cómo"*

¡Nos vemos en el siguiente módulo! 🚀

---

{{< callout icon="sparkles" >}}
¡Gracias por llegar hasta acá! Espero que este recorrido por el universo de la programación haya sido tan apasionante para vos como lo fue para mí escribirlo.

Nos encantaría escuchar lo que pensás, así que no te quedes callado/a – dejá tus comentarios, sugerencias y todas esas ideas copadas que seguro se te ocurrieron.

Y para ir más allá de estas líneas, date una vuelta por los ejemplos prácticos que preparamos para vos. Vas a encontrar todo el código y los proyectos en nuestro repositorio de GitHub [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

¡Gracias por ser parte de esta comunidad de aprendizaje. Seguí programando y explorando nuevos territorios en este fascinante mundo del software!
{{< /callout >}}
