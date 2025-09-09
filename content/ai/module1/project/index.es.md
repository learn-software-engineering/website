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

Haciendo click en el siguiente [enlace](https://github.com/learn-software-engineering/examples/tree/main/ai/module1/sistema_de_recomendaciones), puedes encontrar una posible solución para este problema. Incluye explicaciones detalladas.

{{< alert >}}
¡Intentá resolverlo por tu cuenta primero! 😀😀😀
{{< /alert >}}

### Ejecución

Aquí hay algunos ejemplos de ejecución:

#### Demostración completa

```bash
>  python main.py
Selecciona el tipo de demostración:
   1- Demostración completa
   2- Comparación de algoritmos
   3- Usuario específico

Opción (1-3): 1
Sistema de Recomendaciones - Demostración Completa
==================================================
Cargando la configuración desde config.yaml
Configuración cargada desde config.yaml
Configuración validada exitosamente
Inicializando Sistema de Recomendaciones
   * 6 usuarios cargados
   * 12 productos cargados
   * 22 interacciones cargadas
Sistema listo!

========================
Estadísticas del Sistema
========================
Analizando sistema
Usuarios: 6
   * Por género: {'F': 3, 'M': 3}
   * Por ubicación: {'Buenos Aires': 2, 'Córdoba': 1, 'Rosario': 1, 'Mendoza': 1, 'La Plata': 1}
   * Por nivel de gasto: {'medio': 3, 'alto': 2, 'bajo': 1}
   * Edad promedio: 31.5 años
Productos: 12
   * Por categoría: {'tecnologia': 4, 'lectura': 2, 'gaming': 2, 'hogar': 1, 'fitness': 2, 'cocina': 1}
   * Precio promedio: $37,950
   * Calificación promedio: 4.36/5
Interacciones: 22
   * Por tipo: {'compra': 11, 'view': 9, 'wishlist': 2}
   * Por usuario: {'user_001': 4, 'user_002': 4, 'user_003': 4, 'user_004': 4, 'user_005': 3, 'user_006': 3}
   * Usuarios activos: 6
Similitudes:
   * Promedio: 0.162
   * Máxima: 0.398
   * Mínima: 0.000

=============================
Reporte Detallado - ANA LÓPEZ
=============================
Perfil del usuario
   * Edad: 28 años
   * Género: F
   * Ubicación: Buenos Aires
   * Intereses: tecnologia, fitness, lectura
   * Nivel de gasto: medio
Historial de actividad (4 interacciones):
   • 2024-08-17: WISHLIST - Cámara DSLR Canon EOS Rebel T7
   • 2024-08-10: COMPRA - Proteína Whey Gold Standard (Calificación: 4/5)
   • 2024-08-05: VIEW - Auriculares Bluetooth Sport Pro
   • 2024-08-01: COMPRA - Smartphone Samsung Galaxy S24 (Calificación: 5/5)
3 usuarios más similares
   * Diego Silva: 0.398 (39.8% similar)
   * Roberto García: 0.322 (32.2% similar)
   * Laura Martínez: 0.246 (24.6% similar)
Generando recomendaciones para Ana López
========================================
   Aplicando filtrado colaborativo para Ana López...
   Usuarios similares encontrados: 5
   6 recomendaciones colaborativas generadas
   Aplicando filtrado por contenido para Ana López...
   8 recomendaciones por contenido generadas
   Combinando estrategias de recomendación
      * Productos únicos después de combinar: 8
   Aplicando reglas de negocio
      * Los usuarios jóvenes (<35) prefieren productos tecnológicos: 1 productos afectados
   1 reglas aplicadas: boost_tecnologia_jovenes
   Primeras 5 recomendaciones seleccionadas
   Recomendaciones generadas exitosamente!
5 recomendaciones generadas:
   #1 - Tablet Samsung Galaxy Tab S9
      Precio: $95,000
      Calificación: 4.4/5
      Puntuación: 3.214
      Explicación: Te gusta la categoría tecnologia; Apropiado para tu grupo de edad; Precio equilibrado para tu rango; Buena valoración (4.4/5); Producto muy popular
   #2 - Set Pesas Ajustables 40kg
      Precio: $15,000
      Calificación: 4.1/5
      Puntuación: 3.179
      Explicación: Coincide con tu interés en fitness; Te gusta la categoría fitness; Apropiado para tu grupo de edad; Precio equilibrado para tu rango; Buena valoración (4.1/5); A usuarios con perfil similar también les gustó
   #3 - Libro: El Arte de la Guerra Digital
      Precio: $4,200
      Calificación: 4.5/5
      Puntuación: 2.808
      Explicación: Coincide con tu interés en tecnologia; Te gusta la categoría lectura; Apropiado para tu grupo de edad; Excelente valoración (4.5/5)
   #4 - Libro: Cocina Mediterránea Saludable
      Precio: $3,500
      Calificación: 4.8/5
      Puntuación: 2.14
      Explicación: Te gusta la categoría lectura; Apropiado para tu grupo de edad; Excelente valoración (4.8/5)
   #5 - Cafetera Espresso DeLonghi
      Precio: $32,000
      Calificación: 4.2/5
      Puntuación: 1.792
      Explicación: Apropiado para tu grupo de edad; Precio equilibrado para tu rango; Buena valoración (4.2/5); A usuarios con perfil similar también les gustó

=================================
Reporte Detallado - CARLOS MENDEZ
=================================
Perfil del usuario
   * Edad: 35 años
   * Género: M
   * Ubicación: Córdoba
   * Intereses: deportes, gaming, musica
   * Nivel de gasto: alto
Historial de actividad (4 interacciones):
   • 2024-08-18: VIEW - Tablet Samsung Galaxy Tab S9
   • 2024-08-12: COMPRA - Auriculares Gaming HyperX Cloud III (Calificación: 5/5)
   • 2024-08-07: VIEW - Smartphone Samsung Galaxy S24
   • 2024-08-03: COMPRA - Juego: Call of Duty Modern Warfare III (Calificación: 4/5)
3 usuarios más similares
   * Roberto García: 0.192 (19.2% similar)
   * Laura Martínez: 0.168 (16.8% similar)
   * Diego Silva: 0.120 (12.0% similar)
Generando recomendaciones para Carlos Mendez
============================================
   Aplicando filtrado colaborativo para Carlos Mendez...
   Usuarios similares encontrados: 5
   6 recomendaciones colaborativas generadas
   Aplicando filtrado por contenido para Carlos Mendez...
   8 recomendaciones por contenido generadas
   Combinando estrategias de recomendación
      * Productos únicos después de combinar: 8
   Aplicando reglas de negocio
      * Usuarios con gasto alto ven productos premium primero: 8 productos afectados
   1 reglas aplicadas: usuarios_premium
   Primeras 5 recomendaciones seleccionadas
   Recomendaciones generadas exitosamente!
5 recomendaciones generadas:
   #1 - Auriculares Bluetooth Sport Pro
      Precio: $25,000
      Calificación: 4.2/5
      Puntuación: 2.392
      Explicación: Coincide con tu interés en deportes; Apropiado para tu grupo de edad; Buena valoración (4.2/5); Producto muy popular
   #2 - Cámara DSLR Canon EOS Rebel T7
      Precio: $85,000
      Calificación: 4.3/5
      Puntuación: 1.949
      Explicación: Apropiado para tu grupo de edad; Producto premium que se ajusta a tu perfil; Buena valoración (4.3/5)
   #3 - Proteína Whey Gold Standard
      Precio: $7,200
      Calificación: 4.6/5
      Puntuación: 1.697
      Explicación: Apropiado para tu grupo de edad; Excelente valoración (4.6/5); Producto muy popular
   #4 - Cafetera Espresso DeLonghi
      Precio: $32,000
      Calificación: 4.2/5
      Puntuación: 1.689
      Explicación: Apropiado para tu grupo de edad; Buena valoración (4.2/5)
   #5 - Libro: El Arte de la Guerra Digital
      Precio: $4,200
      Calificación: 4.5/5
      Puntuación: 1.633
      Explicación: Apropiado para tu grupo de edad; Excelente valoración (4.5/5)

===================================
Reporte Detallado - MARÍA RODRIGUEZ
===================================
Perfil del usuario
   * Edad: 42 años
   * Género: F
   * Ubicación: Rosario
   * Intereses: cocina, jardineria, arte
   * Nivel de gasto: bajo
Historial de actividad (4 interacciones):
   • 2024-08-20: VIEW - Cámara DSLR Canon EOS Rebel T7
   • 2024-08-15: COMPRA - Cafetera Espresso DeLonghi (Calificación: 3/5)
   • 2024-08-06: VIEW - Kit Herramientas Jardín Premium
   • 2024-08-02: COMPRA - Libro: Cocina Mediterránea Saludable (Calificación: 5/5)
3 usuarios más similares
   * Roberto García: 0.193 (19.3% similar)
   * Laura Martínez: 0.120 (12.0% similar)
   * Ana López: 0.060 (6.0% similar)
Generando recomendaciones para María Rodriguez
==============================================
   Aplicando filtrado colaborativo para María Rodriguez...
   Usuarios similares encontrados: 5
   7 recomendaciones colaborativas generadas
   Aplicando filtrado por contenido para María Rodriguez...
   8 recomendaciones por contenido generadas
   Combinando estrategias de recomendación
      * Productos únicos después de combinar: 8
   Aplicando reglas de negocio
      * Usuarios con presupuesto bajo ven productos económicos: 2 productos afectados
   1 reglas aplicadas: productos_economicos
   Primeras 5 recomendaciones seleccionadas
   Recomendaciones generadas exitosamente!
5 recomendaciones generadas:
   #1 - Proteína Whey Gold Standard
      Precio: $7,200
      Calificación: 4.6/5
      Puntuación: 2.432
      Explicación: Apropiado para tu grupo de edad; Precio económico y accesible; Excelente valoración (4.6/5); Producto muy popular
   #2 - Libro: El Arte de la Guerra Digital
      Precio: $4,200
      Calificación: 4.5/5
      Puntuación: 2.313
      Explicación: Apropiado para tu grupo de edad; Precio económico y accesible; Excelente valoración (4.5/5)
   #3 - Set Pesas Ajustables 40kg
      Precio: $15,000
      Calificación: 4.1/5
      Puntuación: 1.588
      Explicación: Apropiado para tu grupo de edad; Precio económico y accesible; Buena valoración (4.1/5)
   #4 - Smartphone Samsung Galaxy S24
      Precio: $150,000
      Calificación: 4.5/5
      Puntuación: 1.572
      Explicación: Apropiado para tu grupo de edad; Excelente valoración (4.5/5); Producto muy popular
   #5 - Tablet Samsung Galaxy Tab S9
      Precio: $95,000
      Calificación: 4.4/5
      Puntuación: 1.352
      Explicación: Apropiado para tu grupo de edad; Buena valoración (4.4/5); Producto muy popular

===============================
Reporte Detallado - DIEGO SILVA
===============================
Perfil del usuario
   * Edad: 24 años
   * Género: M
   * Ubicación: Buenos Aires
   * Intereses: tecnologia, gaming, fitness
   * Nivel de gasto: medio
Historial de actividad (4 interacciones):
   • 2024-08-19: WISHLIST - Smartphone Samsung Galaxy S24
   • 2024-08-14: COMPRA - Set Pesas Ajustables 40kg (Calificación: 4/5)
   • 2024-08-08: VIEW - Juego: Call of Duty Modern Warfare III
   • 2024-08-04: COMPRA - Auriculares Bluetooth Sport Pro (Calificación: 4/5)
3 usuarios más similares
   * Ana López: 0.398 (39.8% similar)
   * Roberto García: 0.310 (31.0% similar)
   * Carlos Mendez: 0.120 (12.0% similar)
Generando recomendaciones para Diego Silva
==========================================
   Aplicando filtrado colaborativo para Diego Silva...
   Usuarios similares encontrados: 5
   5 recomendaciones colaborativas generadas
   Aplicando filtrado por contenido para Diego Silva...
   8 recomendaciones por contenido generadas
   Combinando estrategias de recomendación
      * Productos únicos después de combinar: 8
   Aplicando reglas de negocio
      * Los usuarios jóvenes (<35) prefieren productos tecnológicos: 2 productos afectados
   1 reglas aplicadas: boost_tecnologia_jovenes
   Primeras 5 recomendaciones seleccionadas
   Recomendaciones generadas exitosamente!
5 recomendaciones generadas:
   #1 - Tablet Samsung Galaxy Tab S9
      Precio: $95,000
      Calificación: 4.4/5
      Puntuación: 3.214
      Explicación: Te gusta la categoría tecnologia; Apropiado para tu grupo de edad; Precio equilibrado para tu rango; Buena valoración (4.4/5); Producto muy popular
   #2 - Cámara DSLR Canon EOS Rebel T7
      Precio: $85,000
      Calificación: 4.3/5
      Puntuación: 3.151
      Explicación: Te gusta la categoría tecnologia; Apropiado para tu grupo de edad; Precio equilibrado para tu rango; Buena valoración (4.3/5)
   #3 - Auriculares Gaming HyperX Cloud III
      Precio: $18,000
      Calificación: 4.4/5
      Puntuación: 3.124
      Explicación: Coincide con tu interés en gaming; Te gusta la categoría gaming; Apropiado para tu grupo de edad; Precio equilibrado para tu rango; Buena valoración (4.4/5)
   #4 - Proteína Whey Gold Standard
      Precio: $7,200
      Calificación: 4.6/5
      Puntuación: 2.979
      Explicación: Coincide con tu interés en fitness; Te gusta la categoría fitness; Apropiado para tu grupo de edad; Excelente valoración (4.6/5); A usuarios con perfil similar también les gustó; Producto muy popular
   #5 - Libro: El Arte de la Guerra Digital
      Precio: $4,200
      Calificación: 4.5/5
      Puntuación: 1.592
      Explicación: Coincide con tu interés en tecnologia; Excelente valoración (4.5/5)

==================================
Reporte Detallado - LAURA MARTÍNEZ
==================================
Perfil del usuario
   * Edad: 31 años
   * Género: F
   * Ubicación: Mendoza
   * Intereses: lectura, arte, musica
   * Nivel de gasto: medio
Historial de actividad (3 interacciones):
   • 2024-08-13: VIEW - Cámara DSLR Canon EOS Rebel T7
   • 2024-08-11: COMPRA - Libro: El Arte de la Guerra Digital (Calificación: 5/5)
   • 2024-08-09: VIEW - Libro: Cocina Mediterránea Saludable
3 usuarios más similares
   * Ana López: 0.246 (24.6% similar)
   * Carlos Mendez: 0.168 (16.8% similar)
   * Roberto García: 0.144 (14.4% similar)
Generando recomendaciones para Laura Martínez
=============================================
   Aplicando filtrado colaborativo para Laura Martínez...
   Usuarios similares encontrados: 5
   7 recomendaciones colaborativas generadas
   Aplicando filtrado por contenido para Laura Martínez...
   9 recomendaciones por contenido generadas
   Combinando estrategias de recomendación
      * Productos únicos después de combinar: 9
   Aplicando reglas de negocio
      * Los usuarios jóvenes (<35) prefieren productos tecnológicos: 3 productos afectados
   1 reglas aplicadas: boost_tecnologia_jovenes
   Primeras 5 recomendaciones seleccionadas
   Recomendaciones generadas exitosamente!
5 recomendaciones generadas:
   #1 - Auriculares Bluetooth Sport Pro
      Precio: $25,000
      Calificación: 4.2/5
      Puntuación: 2.186
      Explicación: Apropiado para tu grupo de edad; Precio equilibrado para tu rango; Buena valoración (4.2/5); Producto muy popular
   #2 - Tablet Samsung Galaxy Tab S9
      Precio: $95,000
      Calificación: 4.4/5
      Puntuación: 2.174
      Explicación: Apropiado para tu grupo de edad; Precio equilibrado para tu rango; Buena valoración (4.4/5); Producto muy popular
   #3 - Smartphone Samsung Galaxy S24
      Precio: $150,000
      Calificación: 4.5/5
      Puntuación: 2.15
      Explicación: Apropiado para tu grupo de edad; Excelente valoración (4.5/5); Producto muy popular
   #4 - Juego: Call of Duty Modern Warfare III
      Precio: $12,000
      Calificación: 4.3/5
      Puntuación: 1.785
      Explicación: Apropiado para tu grupo de edad; Precio equilibrado para tu rango; Buena valoración (4.3/5); Producto muy popular
   #5 - Auriculares Gaming HyperX Cloud III
      Precio: $18,000
      Calificación: 4.4/5
      Puntuación: 1.753
      Explicación: Apropiado para tu grupo de edad; Precio equilibrado para tu rango; Buena valoración (4.4/5)

==================================
Reporte Detallado - ROBERTO GARCÍA
==================================
Perfil del usuario
   * Edad: 29 años
   * Género: M
   * Ubicación: La Plata
   * Intereses: deportes, tecnologia, cocina
   * Nivel de gasto: alto
Historial de actividad (3 interacciones):
   • 2024-08-16: VIEW - Tablet Samsung Galaxy Tab S9
   • 2024-08-09: COMPRA - Cafetera Espresso DeLonghi (Calificación: 4/5)
   • 2024-08-05: COMPRA - Smartphone Samsung Galaxy S24 (Calificación: 5/5)
3 usuarios más similares
   * Ana López: 0.322 (32.2% similar)
   * Diego Silva: 0.310 (31.0% similar)
   * María Rodriguez: 0.193 (19.3% similar)
Generando recomendaciones para Roberto García
=============================================
   Aplicando filtrado colaborativo para Roberto García...
   Usuarios similares encontrados: 5
   7 recomendaciones colaborativas generadas
   Aplicando filtrado por contenido para Roberto García...
   9 recomendaciones por contenido generadas
   Combinando estrategias de recomendación
      * Productos únicos después de combinar: 9
   Aplicando reglas de negocio
      * Los usuarios jóvenes (<35) prefieren productos tecnológicos: 2 productos afectados
      * Usuarios con gasto alto ven productos premium primero: 9 productos afectados
   2 reglas aplicadas: boost_tecnologia_jovenes, usuarios_premium
   Primeras 5 recomendaciones seleccionadas
   Recomendaciones generadas exitosamente!
5 recomendaciones generadas:
   #1 - Auriculares Bluetooth Sport Pro
      Precio: $25,000
      Calificación: 4.2/5
      Puntuación: 4.5
      Explicación: Coincide con tu interés en deportes; Te gusta la categoría tecnologia; Apropiado para tu grupo de edad; Buena valoración (4.2/5); A usuarios con perfil similar también les gustó; Producto muy popular
   #2 - Cámara DSLR Canon EOS Rebel T7
      Precio: $85,000
      Calificación: 4.3/5
      Puntuación: 3.781
      Explicación: Te gusta la categoría tecnologia; Apropiado para tu grupo de edad; Producto premium que se ajusta a tu perfil; Buena valoración (4.3/5)
   #3 - Libro: Cocina Mediterránea Saludable
      Precio: $3,500
      Calificación: 4.8/5
      Puntuación: 2.424
      Explicación: Coincide con tu interés en cocina; Apropiado para tu grupo de edad; Excelente valoración (4.8/5)
   #4 - Libro: El Arte de la Guerra Digital
      Precio: $4,200
      Calificación: 4.5/5
      Puntuación: 2.336
      Explicación: Coincide con tu interés en tecnologia; Apropiado para tu grupo de edad; Excelente valoración (4.5/5)
   #5 - Proteína Whey Gold Standard
      Precio: $7,200
      Calificación: 4.6/5
      Puntuación: 1.851
      Explicación: Apropiado para tu grupo de edad; Excelente valoración (4.6/5); A usuarios con perfil similar también les gustó; Producto muy popular

=====================================
Demostración Completada Exitosamente!
Revisa los reportes anteriores para entender cómo funciona cada componente
=====================================
```

#### Comparación de algoritmos

```bash
>  python main.py
Selecciona el tipo de demostración:
   1- Demostración completa
   2- Comparación de algoritmos
   3- Usuario específico

Opción (1-3): 2
=========================
Comparación de Algoritmos
=========================
Cargando la configuración desde config.yaml
Configuración cargada desde config.yaml
Configuración validada exitosamente
Inicializando Sistema de Recomendaciones
   * 6 usuarios cargados
   * 12 productos cargados
   * 22 interacciones cargadas
Sistema listo!
Usuario de prueba: Ana López
Recomendaciones por filtrado colaborativo:
   Aplicando filtrado colaborativo para Ana López...
   Usuarios similares encontrados: 5
   5 recomendaciones colaborativas generadas
   1. Set Pesas Ajustables 40kg (Puntuación: 0.318)
   2. Cafetera Espresso DeLonghi (Puntuación: 0.294)
   3. Libro: El Arte de la Guerra Digital (Puntuación: 0.246)
   4. Libro: Cocina Mediterránea Saludable (Puntuación: 0.060)
   5. Auriculares Gaming HyperX Cloud III (Puntuación: 0.054)
Recomendaciones por filtrado por contenido:
   Aplicando filtrado por contenido para Ana López...
   5 recomendaciones por contenido generadas
   1. Set Pesas Ajustables 40kg (Puntuación: 7.470)
   2. Libro: El Arte de la Guerra Digital (Puntuación: 6.650)
   3. Tablet Samsung Galaxy Tab S9 (Puntuación: 6.180)
   4. Libro: Cocina Mediterránea Saludable (Puntuación: 5.260)
   5. Juego: Call of Duty Modern Warfare III (Puntuación: 4.260)
Sistema híbrido
Generando recomendaciones para Ana López
========================================
   Aplicando filtrado colaborativo para Ana López...
   Usuarios similares encontrados: 5
   6 recomendaciones colaborativas generadas
   Aplicando filtrado por contenido para Ana López...
   8 recomendaciones por contenido generadas
   Combinando estrategias de recomendación
      * Productos únicos después de combinar: 8
   Aplicando reglas de negocio
      * Los usuarios jóvenes (<35) prefieren productos tecnológicos: 1 productos afectados
   1 reglas aplicadas: boost_tecnologia_jovenes
   Primeras 5 recomendaciones seleccionadas
   Recomendaciones generadas exitosamente!
   1. Tablet Samsung Galaxy Tab S9 (Puntuación: 3.214)
   2. Set Pesas Ajustables 40kg (Puntuación: 3.179)
   3. Libro: El Arte de la Guerra Digital (Puntuación: 2.808)
   4. Libro: Cocina Mediterránea Saludable (Puntuación: 2.14)
   5. Cafetera Espresso DeLonghi (Puntuación: 1.792)
Observaciones
   * Colaborativo: Basado en usuarios similares
   * Contenido: Basado en perfil del usuario
   * Híbrido: Combina ambos enfoques
```

#### Usuario específico

```bash
>  python main.py
Selecciona el tipo de demostración:
   1- Demostración completa
   2- Comparación de algoritmos
   3- Usuario específico

Opción (1-3): 3
Cargando la configuración desde config.yaml
Configuración cargada desde config.yaml
Configuración validada exitosamente
Inicializando Sistema de Recomendaciones
   * 6 usuarios cargados
   * 12 productos cargados
   * 22 interacciones cargadas
Sistema listo!

Usuarios disponibles:
   user_001: Ana López
   user_002: Carlos Mendez
   user_003: María Rodriguez
   user_004: Diego Silva
   user_005: Laura Martínez
   user_006: Roberto García

Ingresa ID de usuario: user_002

=================================
Reporte Detallado - CARLOS MENDEZ
=================================
Perfil del usuario
   * Edad: 35 años
   * Género: M
   * Ubicación: Córdoba
   * Intereses: deportes, gaming, musica
   * Nivel de gasto: alto
Historial de actividad (4 interacciones):
   • 2024-08-18: VIEW - Tablet Samsung Galaxy Tab S9
   • 2024-08-12: COMPRA - Auriculares Gaming HyperX Cloud III (Calificación: 5/5)
   • 2024-08-07: VIEW - Smartphone Samsung Galaxy S24
   • 2024-08-03: COMPRA - Juego: Call of Duty Modern Warfare III (Calificación: 4/5)
3 usuarios más similares
   * Roberto García: 0.192 (19.2% similar)
   * Laura Martínez: 0.168 (16.8% similar)
   * Diego Silva: 0.120 (12.0% similar)
Generando recomendaciones para Carlos Mendez
============================================
   Aplicando filtrado colaborativo para Carlos Mendez...
   Usuarios similares encontrados: 5
   6 recomendaciones colaborativas generadas
   Aplicando filtrado por contenido para Carlos Mendez...
   8 recomendaciones por contenido generadas
   Combinando estrategias de recomendación
      * Productos únicos después de combinar: 8
   Aplicando reglas de negocio
      * Usuarios con gasto alto ven productos premium primero: 8 productos afectados
   1 reglas aplicadas: usuarios_premium
   Primeras 5 recomendaciones seleccionadas
   Recomendaciones generadas exitosamente!
5 recomendaciones generadas:
   #1 - Auriculares Bluetooth Sport Pro
      Precio: $25,000
      Calificación: 4.2/5
      Puntuación: 2.392
      Explicación: Coincide con tu interés en deportes; Apropiado para tu grupo de edad; Buena valoración (4.2/5); Producto muy popular
   #2 - Cámara DSLR Canon EOS Rebel T7
      Precio: $85,000
      Calificación: 4.3/5
      Puntuación: 1.949
      Explicación: Apropiado para tu grupo de edad; Producto premium que se ajusta a tu perfil; Buena valoración (4.3/5)
   #3 - Proteína Whey Gold Standard
      Precio: $7,200
      Calificación: 4.6/5
      Puntuación: 1.697
      Explicación: Apropiado para tu grupo de edad; Excelente valoración (4.6/5); Producto muy popular
   #4 - Cafetera Espresso DeLonghi
      Precio: $32,000
      Calificación: 4.2/5
      Puntuación: 1.689
      Explicación: Apropiado para tu grupo de edad; Buena valoración (4.2/5)
   #5 - Libro: El Arte de la Guerra Digital
      Precio: $4,200
      Calificación: 4.5/5
      Puntuación: 1.633
      Explicación: Apropiado para tu grupo de edad; Excelente valoración (4.5/5)
```

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
