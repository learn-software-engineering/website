---
weight: 4
series: ["Fundamentos de IA para Programadores"]
series_order: 4
title: "Proyecto - Sistema de Recomendaciones"
description: "Vamos a crear un sistema de recomendaciones usando reglas lógicas. Este tipo de sistema es común en e-commerce, plataformas de contenido y aplicaciones móviles."
authors:
  - jnonino
date: 2025-09-03
tags: ["Inteligencia Artificial", "Aprendizaje Automático", "Machine Learning", "Recomendaciones"]
---
{{< katex >}}

{{< lead >}}
Para consolidar todo lo aprendido, vamos a crear un sistema de recomendaciones usando reglas lógicas. Este tipo de sistema es común en e-commerce, plataformas de contenido y aplicaciones móviles.
{{< /lead >}}

---

## Problema a resolver

Eres un ingeniero de software trabajando para una empresa de e-commerce que necesita implementar un **sistema de recomendaciones** para mejorar la experiencia de sus usuarios y aumentar las ventas.

Tu misión es desarrollar un sistema que pueda sugerir productos relevantes a los usuarios basándose en:
- Su perfil demográfico (edad, ubicación, intereses)
- Su historial de compras y navegación
- El comportamiento de usuarios similares
- Características de los productos
- Reglas específicas del negocio

### Datos disponibles

Para el desarrollo del sistema tenemos preparados un conjunto de datos que podés usar para verificar el funcionamiento de tu código.

**Usuarios**: [*datos/usuarios.json*](https://github.com/learn-software-engineering/examples/blob/main/ai/module1/sistema_de_recomendaciones/datos/usuarios.json)
- *Datos demográficos*: edad, género, ubicación
- *Intereses*: lista de categorías preferidas
- *Nivel de gasto*: bajo, medio, alto
- *Historial de actividad*: compras, visualizaciones, ratings

**Productos**: [*datos/productos.json*](https://github.com/learn-software-engineering/examples/blob/main/ai/module1/sistema_de_recomendaciones/datos/productos.json)
- *Información básica*: nombre, categoría, precio
- *Métricas de calidad*: rating promedio, popularidad
- *Metadatos*: tags descriptivos, público objetivo
- *Características comerciales*: disponibilidad, promociones

**Interacciones**: [*datos/interacciones.json*](https://github.com/learn-software-engineering/examples/blob/main/ai/module1/sistema_de_recomendaciones/datos/interacciones.json)
- *Tipos de interacción*: compra, visualización, rating, wishlist
- *Datos temporales*: fecha y hora
- *Valoraciones*: puntuaciones de 1-5 estrellas
- *Contexto*: dispositivo, ubicación, sesión

### Requerimientos Funcionales

1. **Calcular similitud entre usuarios**
```
ENTRADA: ID de usuario
PROCESO: Calcular similitud con otros usuarios basándose en:
         - Productos en común que han comprado o visto
         - Similitud demográfica (edad, género, ubicación)
         - Intereses compartidos
SALIDA: Lista de usuarios similares ordenada por similitud
```

2. **Generar recomendaciones por filtrado colaborativo**
```
ENTRADA: ID de usuario, número de recomendaciones deseadas
PROCESO: - Encontrar usuarios similares
         - Identificar productos que les gustaron a esos usuarios
         - Filtrar productos ya conocidos por el usuario objetivo
         - Puntuar por similitud de usuarios y ratings
SALIDA: Lista de productos recomendados con scores
```

3. **Generar recomendaciones por contenido**
```
ENTRADA: ID de usuario, número de recomendaciones deseadas
PROCESO: - Analizar perfil del usuario (intereses, demografía)
         - Evaluar cada producto por compatibilidad:
           * Matching con intereses del usuario
           * Adecuación por edad y demografía
           * Calidad del producto (rating, popularidad)
           * Precio apropiado para nivel de gasto
SALIDA: Lista de productos recomendados con scores
```

4. **Procesar reglas de negocio**
```
ENTRADA: Usuario, lista de recomendaciones preliminares
PROCESO: Aplicar reglas como:
         - Boost a productos tech para usuarios jóvenes
         - Promocionar productos premium a usuarios de gasto alto
         - Priorizar productos económicos para presupuesto bajo
         - Aplicar preferencias regionales
SALIDA: Recomendaciones ajustadas por reglas de negocio
```

5. **Generar explicaciones para las recomendaciones**
```
ENTRADA: Usuario, producto recomendado
PROCESO: Generar explicación legible de por qué se recomienda:
         - "Coincide con tus intereses en tecnología"
         - "A usuarios similares también les gustó"
         - "Excelente rating (4.8/5)"
         - "Precio apropiado para tu perfil"
SALIDA: Lista de razones explicativas
```

### Métricas de evaluación

Evalúa tu sistema considerando:

**Precisión**
- ¿Las recomendaciones son relevantes para cada usuario?
- ¿Evita recomendar productos ya conocidos?

**Diversidad**
- ¿Ofrece variedad de categorías y precios?
- ¿Evita el sesgo hacia productos populares únicamente?

**Explicabilidad**
- ¿Las razones son claras y convincentes?

**Eficiencia**
- ¿Responde en tiempo razonable?

---

## Solución

Haciendo click en el siguiente [enlace](https://github.com/learn-software-engineering/examples/blob/main/ai/module1/sistema_de_recomendaciones), puedes encontrar una posible solución para este problema. Incluye explicaciones detalladas.

{{< alert >}}
¡Intentá resolverlo por tu cuenta primero! 😀😀😀
{{< /alert >}}

### ¿Qué nos aporta esta solución?

1. **Sistema híbrido**: Combinamos múltiples estrategias de recomendación para obtener mejores resultados.
2. **Reglas de negocio**: Implementamos lógica específica del dominio que puede adaptarse fácilmente.
3. **Explicabilidad**: El sistema puede explicar por qué recomienda cada producto, generando confianza.
4. **Escalabilidad**: La arquitectura permite agregar nuevas fuentes de datos y reglas fácilmente.

### Ejercicios para profundizar

A continuación tienes algunos ejercicios que te ayudarán a profundizar tus conocimientos.

- **Ejercicio 1**: Implementa una nueva regla de negocio que dé boost a productos en oferta durante los fines de semana.
- **Ejercicio 2**: Implementa un sistema de feedback que aprenda de las interacciones del usuario (like/dislike).

Por supuesto, no tienes porque detenerte en estas recomendaciones, dejá volar tu imaginación y utilizá los conceptos aprendidos para otros casos.

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
