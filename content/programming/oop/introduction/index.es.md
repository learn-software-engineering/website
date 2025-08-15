---
weight: 1
series: ["Programación: Aprendiendo Orientación a Objetos"]
series_order: 1
title: "Introducción"
authors:
  - jnonino
description: >
  La Programación Orientada a Objetos (POO) es un paradigma de programación que se ha vuelto indispensable en la actualidad. Este enfoque modela elementos del mundo real como "objetos" que tienen propiedades y comportamientos, lo cual permite crear programas más intuitivos y fáciles de mantener. En este artículo veremos los conceptos básicos de POO y sus ventajas frente a otros paradigmas como la programación procedural.
date: 2025-08-04
tags: ["Programación", "Programación Orientada a Objetos"]
---

La Programación Orientada a Objetos (POO) es un paradigma de programación que se ha vuelto indispensable en la actualidad. Este enfoque modela elementos del mundo real como "objetos" que tienen propiedades y comportamientos, lo cual permite crear programas más intuitivos y fáciles de mantener. En este artículo veremos los conceptos básicos de POO y sus ventajas frente a otros paradigmas como la programación procedural. ¡Empecemos!

Este paradigma se basa en dos conceptos fundamentales:
- **Objetos**: entidades que combinan estado (datos) y comportamiento (operaciones) en una misma unidad. Por ejemplo, un objeto "coche" tendría propiedades como color, número de puertas, velocidad máxima, etc. Y comportamientos como acelerar, frenar, girar, etc.
- **Clases**: especificaciones que definen la estructura y comportamiento común de un grupo de objetos. La clase "coche" serviría como molde para crear objetos coche con las mismas características.

Como explica el programador Alan Kay, uno de los creadores de la POO:

> "La idea central de POO es que los usuarios deben manipular objetos conceptuales más que máquinas de Turing. Las interfaces con el mundo real deben, por lo tanto, ser construidas en términos de objetos conceptuales."

Es decir, la POO modela conceptualmente elementos del mundo real para hacer la programación más intuitiva.

---

## Paradigmas de programación

Antes de profundizar en la POO, conviene entender que existen diferentes paradigmas o enfoques para abordar la programación. Los principales son:

### Programación procedural

Secuencia ordenada de instrucciones que el programa debe seguir paso a paso. El foco está en procedimientos y funciones. Por ejemplo, C es un lenguaje orientado a la programación procedural.

La programación procedural es mejor para:
- Problemas sencillos o algoritmos secuenciales.
- Código que no necesitará reusarse ni expandirse mucho.
- Casos donde el rendimiento y eficiencia son críticos.

### Programación orientada a objetos

Modelo basado en objetos que contienen datos y código en unidades cohesivas. El foco está en las clases y en la interacción entre objetos. Por ejemplo, Java y Python son lenguajes orientados a objetos.

La POO permite modelar de forma más directa elementos del mundo real, encapsular mejor los datos y reutilizar código a través de la herencia entre clases.

Las principales ventajas de POO frente a la programación procedural son:

- **Modularidad**: los objetos agrupan datos y operaciones relacionadas, encapsulando la complejidad interna. Esto permite trabajar con módulos independientes.
- **Ocultación de información**: Los objetos pueden exponer una interfaz simple y ocultar detalles de implementación internos. Esto reduce acoplamientos.
- **Reusabilidad**: Las clases permiten reuse de código. Una clase abstracta puede heredar a múltiples subclases.
- **Extensibilidad**: Podemos extender el comportamiento de clases padres creando nuevas subclases.
- **Mapeo conceptual**: Los objetos representan entidades del mundo real, lo cual facilita la traducción de requerimientos a código.

Sin embargo, la POO también tiene desventajas. Según el programador Paul Graham:

>"La programación orientada a objetos suele ser una molestia. Hace que las cosas sean más difíciles de lo que deberían ser."

Por ejemplo, para problemas simples la POO puede resultar excesiva. Y en proyectos grandes existe el riesgo de abusar de la herencia y el polimorfismo, volviendo el código difícil de seguir.

En definitiva, la POO es más adecuada cuando:
- El problema a modelar tiene entidades claras y estructuradas.
- Queremos reutilizar código encapsulado en clases modulares.
- Trabajamos en sistemas que deben extenderse y mantenerse con facilidad.
