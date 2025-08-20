---
weight: 1
series: ["Convirtiéndome en un Ingeniero en IA"]
series_order: 1
title: "Inteligencia Artificial y Aprendizaje Automático"
authors:
  - jnonino
description: >
  Iniciá un viaje didáctico por el mundo de IA y ML: conceptos clave, ejemplos simples en Python y desafíos interactivos para aprender desde el primer día.
date: 2025-08-20
tags: ["Inteligencia Artificial", "IA", "Aprendizaje Automático", "Machine Learning", "ML"]
---

## ¿Qué vas a aprender esta semana?

- Comprender qué es **Inteligencia Artificial (IA)** y qué es **Aprendizaje Automático (Machine Learning, ML)**.
- Reconocer que ML es una parte de IA.
- Diferenciar estos conceptos aplicándolos a ejemplos reales.
- Experimentar con código en Python que te permite obtener respuestas automáticas e inmediatas, reforzando tu comprensión.

---

## ¿Qué es Inteligencia Artificial (IA)?

La **Inteligencia Artificial** se refiere a sistemas que pueden **resolver problemas o tomar decisiones** de forma similar a cómo lo haría una persona. Por ejemplo, cuando tu asistente de voz reconoce lo que decís y responde, estás usando IA.

### ¿Y qué es Aprendizaje Automático (Machine Learning, ML)?

El **Aprendizaje Automático** es un método dentro de la IA. Consiste en que un sistema **aprenda a partir de datos**, sin que le digas explícitamente qué hacer. Es lo que ocurre cuando un programa se entrena a partir de ejemplos, como fotos con etiquetas “perro” o “gato”, y luego aprende a distinguirlas solo.

### Inteligencia Artificial (IA) vs Machine Learning (ML)

| Concepto | Definición                                 | Ejemplo                                        |
|----------|--------------------------------------------|------------------------------------------------|
| **IA**   | Sistemas que imitan capacidades humanas    | Asistente de voz (usa varios métodos)          |
| **ML**   | Parte de la IA que aprende automáticamente | Clasificador de imágenes entrenado desde datos |

Entonces, todo ML es IA, pero no toda IA es ML: algunas IAs usan reglas fijas, lógica o redes neuronales.

---

## Actividad 1: Mini-test

Vas a ejecutar un programa que te dará una respuesta automática según el ejemplo que introduzcas, ayudándote a practicar la distinción entre IA y ML.

{{< codeimporter
    url="https://raw.githubusercontent.com/learn-software-engineering/examples/refs/heads/main/ai/lesson1-intro/actividad-1-mini-test.py"
    type="python"
    >}}

El ejemplo te permite ver inmediatamente cómo se categorizan casos específicos con un algoritmo muy simple mientras te ayuda a recordar la diferencia.

**Qué hace internamente**:
  1. Define una función que analiza qué ejemplo ingresaste.
  2. Según el listado que convierte en texto, lo clasifica como IA o ML.
  3. Recorrés una lista de pruebas y mostrás resultados claros.

---

## Actividad 2: Mini-quiz interactivo

Copiá este código a un archivo `.py` o notebook. Al ejecutarlo, te pedirá que escribas “IA” o “ML” para distintas situaciones prácticas, y te dirá si acertaste.

{{< codeimporter
    url="https://raw.githubusercontent.com/learn-software-engineering/examples/refs/heads/main/ai/lesson1-intro/actividad-2-mini-quiz.py"
    type="python"
    >}}

---

## Para pensar

¿Se te ocurren otros ejemplos como “reconocimiento de voz”, “traducción automática” o “recomendaciones de productos”? ¿Los asociarías a IA, ML o ambos? Prendé tu entorno y probá con tus propios casos.

---

## ¿Qué sigue en la próxima lección?

Abordaremos las **herramientas matemáticas esenciales** detrás de ML: álgebra lineal (vectores y matrices) y cálculo (derivadas, optimización). Vas a ver cómo esos conceptos permiten que los modelos aprendan de datos.
