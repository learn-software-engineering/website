---
weight: 3
series: ["Fundamentos de IA para Programadores"]
series_order: 3
title: "Ejemplo 2 - Sistema de Clasificación"
description: "Vamos a crear un clasificador de texto básico usando técnicas estadísticas simples. Esto te muestra los conceptos fundamentales detrás de algoritmos más complejos como Naive Bayes."
authors:
  - jnonino
date: 2025-08-22
tags: ["Inteligencia Artificial", "Aprendizaje Automático", "Machine Learning", "Clasificación"]
---
{{< katex >}}

{{< lead >}}
Ahora vamos a crear un clasificador de texto básico usando técnicas estadísticas simples. Esto te muestra los conceptos fundamentales detrás de algoritmos más complejos como Naive Bayes.
{{< /lead >}}

{{< codeimporter
    url="https://raw.githubusercontent.com/learn-software-engineering/examples/main/ai/lesson1-intro/sistema_de_clasificacion_simple.py"
    type="python"
    >}}

---

## ¿Qué aprendemos de este ejemplo?

1. **El pre-procesamiento es crucial**: Limpiar y normalizar el texto afecta directamente la calidad del modelo.

2. **Teorema de Bayes en acción**: Combinamos la probabilidad previa P(categoría) con la evidencia P(palabras|categoría).

3. **Suavizado de Laplace**: Técnica esencial para manejar palabras que no vimos durante el entrenamiento.

4. **Log-probabilidades**: Truco numérico para evitar underflow al multiplicar muchas probabilidades pequeñas.

5. **Interpretabilidad**: Podemos entender qué palabras son más importantes para cada categoría.

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
