---
weight: 2
series: ["Fundamentos de IA para Programadores"]
series_order: 2
title: "Machine Learning Ejemplo 1: Sistema Experto"
description: "Vamos a crear un sistema de Inteligencia Artificial clásica para diagnosticar problemas de rendimiento en aplicaciones web. Este ejemplo te muestra cómo estructurar conocimiento en reglas lógicas."
authors:
  - jnonino
date: 2025-08-22
tags: ["Inteligencia Artificial", "Aprendizaje Automático", "Machine Learning", "Sistema Experto"]
---
{{< katex >}}

{{< lead >}}
Vamos a crear un sistema de IA clásica para diagnosticar problemas de rendimiento en aplicaciones web. Este ejemplo te muestra cómo estructurar conocimiento en reglas lógicas.
{{< /lead >}}

---

## Estructura del sistema

Como se mencionó, vamos a implementar un sistema que diagnostique problemas de rendimiento en aplicaciones web, el primer paso entonces es definir los ***síntomas*** que el sistema debe evaluar para emitir un ***diagnóstico***.

Para este caso sencillo se evaluará, el uso de CPU, uso de memoria RAM, cantidad de conexiones activas en simultáneo, tiempo de respuesta y la cantidad de solicitudes (*requests*) lentas, por poner un límite, aquellas que demoren más de un segundo en volver con una respuesta.

Con ello, podemos generar una estructura inicial del sistema y ejecutarla. Es importante validar que el sistema continua funcionando luego de cada cambio.

```python
class DiagnosticadorRendimiento:
    """
    Sistema experto para diagnosticar problemas de rendimiento
    en aplicaciones web usando reglas lógicas.
    """

    def __init__(self):
        # Base de conocimiento: conjunto de reglas de diagnóstico
        self.reglas = []

    def diagnosticar(self, sintomas):
        """
        Diagnostica problemas basándose en los síntomas reportados.

        Args:
            sintomas (dict): Diccionario con métricas del sistema
                - tiempo_respuesta: tiempo promedio en segundos
                - uso_memoria: porcentaje de memoria utilizada
                - queries_lentas: número de queries que tardan >1s
                - conexiones_activas: número de conexiones simultáneas
                - uso_cpu: porcentaje de CPU utilizada

        Returns:
            list: Lista de diagnósticos posibles con sus certezas
        """
        diagnosticos = []

        # Evaluamos cada regla en nuestra base de conocimiento
        for regla in self.reglas:
            resultado = regla(sintomas)
            if resultado:
                diagnosticos.append(resultado)

        # Ordenamos por nivel de certeza (mayor a menor)
        diagnosticos.sort(key=lambda x: x['certeza'], reverse=True)
        return diagnosticos


# Ejemplo de uso del sistema experto
if __name__ == "__main__":
    # Creamos una instancia de nuestro diagnosticador
    diagnosticador = DiagnosticadorRendimiento()

    # Recibimos del usuario los valores de cada metrica
    print("=" * 50)
    print("Ingrese los valores de rendimiento del sistema")
    cpu = int(input("Porcentaje de CPU utilizada(int): "))
    memoria = int(input("Porcentaje de memoria utilizada(int): "))
    tiempo_de_respuesta = float(input("Cantidad de segundos en promedio para recibir una respuesta(float): "))
    queries_lentas = int(input("Cantidad de queries tardan más de 1 segundo(int): "))
    conexiones_activas = int(input("Cantidad de conexiones simultáneas(int): "))

    # Simulamos métricas de un servidor con problemas
    metricas_servidor = {
        'tiempo_respuesta': tiempo_de_respuesta,
        'uso_memoria': memoria,
        'queries_lentas': queries_lentas,
        'conexiones_activas': conexiones_activas,
        'uso_cpu': cpu
    }

    # Realizamos el diagnóstico
    print("=" * 50)
    print("DIAGNÓSTICO DEL SISTEMA")
    print("=" * 50)

    resultados = diagnosticador.diagnosticar(metricas_servidor)

    if not resultados:
        print("No se detectaron problemas significativos")
    else:
        for i, diagnostico in enumerate(resultados, 1):
            print(f"\n#{i} - {diagnostico['problema']}")
            print(f"Certeza: {diagnostico['certeza']:.1%}")
            print("Recomendaciones:")
            for rec in diagnostico['recomendaciones']:
                print(f"  • {rec}")

```

Al ejecutar el programa, vemos que el sistema dice que no hay problemas significativos. Esto se debe a que el ***DiagnosticadorRendimiento*** no tiene ninguna regla implementada, a continuación veremos cómo generar estas reglas que le darán el *conocimiento* necesario al *diagnosticador* para emitir un diagnóstico.

``` bash
>  python sistema_experto_basico.py
==================================================
Ingrese los valores de rendimiento del sistema
Porcentaje de CPU utilizada(int): 45
Porcentaje de memoria utilizada(int): 92
Cantidad de segundos en promedio para recibir una respuesta(float): 4.2
Cantidad de queries tardan más de 1 segundo(int): 15
Cantidad de conexiones simultáneas(int): 800
==================================================
DIAGNÓSTICO DEL SISTEMA
==================================================
No se detectaron problemas significativos
```

---

## Reglas

Como se observa en el método *diagnosticar* de la clase *DiagnosticadorRendimiento*, las reglas son simplemente ***funciones*** o ***métodos*** de la misma clase, que reciben con argumento un diccionario llamado *sintomas* que contiene las métricas del sistema definidas anteriormente.

```python
# Evaluamos cada regla en nuestra base de conocimiento
for regla in self.reglas:
    resultado = regla(sintomas)
    if resultado:
        diagnosticos.append(resultado)
```
Para que la función *diagnosticar* evalúe la regla, esta debe ser incluida en el diccionario de reglas definido en el constructor de la clase.

```python
def __init__(self):
    # Base de conocimiento: conjunto de reglas de diagnóstico
    self.reglas = []
```

Se espera además que cada regla retorne un mapa con los campos:
- *problema*: posible causa de un mal rendimiento.
- *certeza*: nivel de probabilidad de que el problema definido sea el causante de deficiencias en el rendimiento.
- *recomendaciones*: acciones que se podrían llevar a cabo para mejorar el rendimiento.

Vale aclarar que estas reglas son ficticias y a la hora de generar un sistema experto de este tipo debes definirlas cuidadosamente en base al tipo de problema que intentas resolver y las condiciones de tu sistema.

### Uso de CPU

La primera regla que implementaremos tendrá a su cargo determinar si el nivel de uso de CPU representa un problema de rendimiento o no.

El primer factor a decidir es que nivel se considera alto para el uso de CPU, por ejemplo \(80\%\). También definimos que cuando el valor supere este límite, la certeza de que el uso de CPU es un problem será por los menos \(0.85\) y crecerá hasta \(1\) mientras el uso de CPU crezca hasta el \(100\%\).

```python
def _regla_cpu(self, sintomas):
    """
    Regla: Uso de CPU alto puede indicar procesamiento intensivo o algoritmos ineficientes.
    """
    if sintomas.get('uso_cpu', 0) > 80:
        certeza = min(0.85, sintomas['uso_cpu'] / 100)

        return {
            'problema': 'Procesamiento intensivo o algoritmos ineficientes',
            'certeza': certeza,
            'recomendaciones': [
                'Analizar el código para identificar cuellos de botella',
                'Optimizar algoritmos, por ejemplo, aquellos O(n²) o peores',
                'Implementar una cache para cálculos repetitivos',
                'Considerar procesamiento asíncrono para tareas pesadas'
            ]
        }
    return None
```

Ejecutando nuevamente el sistema con un valor de CPU del \(93\%\) se obtiene:

```bash
>  python sistema_experto_basico.py
==================================================
Ingrese los valores de rendimiento del sistema
Porcentaje de CPU utilizada(int): 93
Porcentaje de memoria utilizada(int): 92
Cantidad de segundos en promedio para recibir una respuesta(float): 4.2
Cantidad de queries tardan más de 1 segundo(int): 15
Cantidad de conexiones simultáneas(int): 800
==================================================
DIAGNÓSTICO DEL SISTEMA
==================================================

#1 - Procesamiento intensivo o algoritmos ineficientes
Certeza: 85.0%
Recomendaciones:
  • Analizar el código para identificar cuellos de botella
  • Optimizar algoritmos, por ejemplo, aquellos O(n²) o peores
  • Implementar una cache para cálculos repetitivos
  • Considerar procesamiento asíncrono para tareas pesadas
```

### Uso de memoria

Para el caso de la memoria RAM, no solo consideraremos su uso (superior al \(85\%\)) sino que también tenderemos en cuenta el tiempo de respuesta del sistema (mayor a \(3.0\) segundos).

```python
def _regla_memoria(self, sintomas):
    """
    Regla: Si el uso de memoria es alto Y el tiempo de respuesta
    es lento, probablemente hay una fuga (leak) de memoria.
    """
    if (sintomas.get('uso_memoria', 0) > 85 and
        sintomas.get('tiempo_respuesta', 0) > 3.0):

        # Calculamos certeza basada en qué tan extremos son los valores
        certeza = min(0.9, (sintomas['uso_memoria'] / 100) *
                      (sintomas['tiempo_respuesta'] / 5))

        return {
            'problema': 'Fuga de memoria o uso excesivo de memoria',
            'certeza': certeza,
            'recomendaciones': [
                'Revisar objetos no liberados en memoria',
                'Implementar un grupo de conexiones para evitar abrir y cerrar nuevas',
                'Analizar el consumo de memoria (memory_profiler)',
                'Considerar aumentar la cantidad de RAM del servidor'
            ]
        }
    return None
```

Veamos como funciona:

```bash
>  python sistema_experto_basico.py
==================================================
Ingrese los valores de rendimiento del sistema
Porcentaje de CPU utilizada(int): 45
Porcentaje de memoria utilizada(int): 92
Cantidad de segundos en promedio para recibir una respuesta(float): 4.2
Cantidad de queries tardan más de 1 segundo(int): 15
Cantidad de conexiones simultáneas(int): 800
==================================================
DIAGNÓSTICO DEL SISTEMA
==================================================

#1 - Fuga de memoria o uso excesivo de memoria
Certeza: 77.3%
Recomendaciones:
  • Revisar objetos no liberados en memoria
  • Implementar un grupo de conexiones para evitar abrir y cerrar nuevas
  • Analizar el consumo de memoria (memory_profiler)
  • Considerar aumentar la cantidad de RAM del servidor
```

Si el tiempo de respuesta no se ve impactado aunque el consumo de memoria sea alto, obtenemos:

```bash
>  python sistema_experto_basico.py
==================================================
Ingrese los valores de rendimiento del sistema
Porcentaje de CPU utilizada(int): 45
Porcentaje de memoria utilizada(int): 92
Cantidad de segundos en promedio para recibir una respuesta(float): 2.8
Cantidad de queries tardan más de 1 segundo(int): 15
Cantidad de conexiones simultáneas(int): 800
==================================================
DIAGNÓSTICO DEL SISTEMA
==================================================
No se detectaron problemas significativos
```

### Nivel de concurrencia

De manera similar, implementaremos una regla para controlar la cantidad de conexiones simultaneas al servidor. Demasiadas conexiones pueden saturarlo y hacer que deje de responder. En este ejemplo, definiremos que \(1000\) conexiones simultaneas es el límite.

```python
def _regla_concurrencia(self, sintomas):
    """
    Regla: Muchas conexiones simultáneas pueden saturar el servidor.
    """
    if sintomas.get('conexiones_activas', 0) > 1000:
        certeza = min(0.8, sintomas['conexiones_activas'] / 2000)

        return {
            'problema': 'Sobrecarga por exceso de conexiones concurrentes',
            'certeza': certeza,
            'recomendaciones': [
                'Implementar rate limiting',
                'Usar un balanceador de carga (load balancer) con múltiples instancias',
                'Utilizar un grupo de conexiones preestablecidas',
                'Implementar colas para procesos no críticos'
            ]
        }
    return None
```

Con la regla anterior, obtenemos los siguientes resultados:

```bash
>  python sistema_experto_basico.py
==================================================
Ingrese los valores de rendimiento del sistema
Porcentaje de CPU utilizada(int): 45
Porcentaje de memoria utilizada(int): 56
Cantidad de segundos en promedio para recibir una respuesta(float): 2.8
Cantidad de queries tardan más de 1 segundo(int): 15
Cantidad de conexiones simultáneas(int): 800
==================================================
DIAGNÓSTICO DEL SISTEMA
==================================================
No se detectaron problemas significativos
```

```bash
>  python sistema_experto_basico.py
==================================================
Ingrese los valores de rendimiento del sistema
Porcentaje de CPU utilizada(int): 45
Porcentaje de memoria utilizada(int): 56
Cantidad de segundos en promedio para recibir una respuesta(float): 2.8
Cantidad de queries tardan más de 1 segundo(int): 15
Cantidad de conexiones simultáneas(int): 1500
==================================================
DIAGNÓSTICO DEL SISTEMA
==================================================

#1 - Sobrecarga por exceso de conexiones concurrentes
Certeza: 75.0%
Recomendaciones:
  • Implementar rate limiting
  • Usar un balanceador de carga (load balancer) con múltiples instancias
  • Utilizar un grupo de conexiones preestablecidas
  • Implementar colas para procesos no críticos
```

### Consultas a la base de datos

Una posible causa para recibir respuestas lentas puede ser el tiempo consumido en las consultas a la base de datos. Implementaremos una regla para que el sistema reconozca estos casos.

```python
def _regla_base_datos(self, sintomas):
    """
    Regla: Si hay muchas queries lentas, el problema está
    en la base de datos.
    """
    if sintomas.get('queries_lentas', 0) > 10:
        certeza = min(0.95, sintomas['queries_lentas'] / 50)

        return {
            'problema': 'Queries de base de datos ineficientes',
            'certeza': certeza,
            'recomendaciones': [
                'Revisar índices en tablas frecuentemente consultadas',
                'Optimizar queries con EXPLAIN ANALYZE',
                'Implementar cache de queries (Redis)',
                'Considerar particionado de tablas grandes'
            ]
        }
    return None
```

Lo que nos da resultados como estos:

```bash
>  python sistema_experto_basico.py
==================================================
Ingrese los valores de rendimiento del sistema
Porcentaje de CPU utilizada(int): 45
Porcentaje de memoria utilizada(int): 56
Cantidad de segundos en promedio para recibir una respuesta(float): 1.5
Cantidad de queries tardan más de 1 segundo(int): 17
Cantidad de conexiones simultáneas(int): 800
==================================================
DIAGNÓSTICO DEL SISTEMA
==================================================

#1 - Queries de base de datos ineficientes
Certeza: 34.0%
Recomendaciones:
  • Revisar índices en tablas frecuentemente consultadas
  • Optimizar queries con EXPLAIN ANALYZE
  • Implementar cache de queries (Redis)
  • Considerar particionado de tablas grandes
```

```bash
>  python sistema_experto_basico.py
==================================================
Ingrese los valores de rendimiento del sistema
Porcentaje de CPU utilizada(int): 45
Porcentaje de memoria utilizada(int): 56
Cantidad de segundos en promedio para recibir una respuesta(float): 1.5
Cantidad de queries tardan más de 1 segundo(int): 49
Cantidad de conexiones simultáneas(int): 800
==================================================
DIAGNÓSTICO DEL SISTEMA
==================================================

#1 - Queries de base de datos ineficientes
Certeza: 95.0%
Recomendaciones:
  • Revisar índices en tablas frecuentemente consultadas
  • Optimizar queries con EXPLAIN ANALYZE
  • Implementar cache de queries (Redis)
  • Considerar particionado de tablas grandes
```

### Estado de la red

Si vemos que el tiempo de respuesta es alto pero CPU y memoria están bien, puede ser un problema de red. A continuación vemos el código para implementar esta regla.

```python
def _regla_red(self, sintomas):
    """
    Regla: Si el tiempo de respuesta es alto pero CPU y memoria
    están bien, puede ser un problema de red.
    """
    if (sintomas.get('tiempo_respuesta', 0) > 2.0 and
        sintomas.get('uso_cpu', 0) < 50 and
        sintomas.get('uso_memoria', 0) < 70):

        certeza = 0.7  # Menos certeza porque es por descarte

        return {
            'problema': 'Latencia de red o problemas de conectividad',
            'certeza': certeza,
            'recomendaciones': [
                'Verificar latencia entre servidor y clientes',
                'Implementar CDN para recursos estáticos',
                'Optimizar tamaño de respuestas (compresión)'
            ]
        }
    return None
```

Con el siguiente resultado:

```bash
>  python sistema_experto_basico.py
==================================================
Ingrese los valores de rendimiento del sistema
Porcentaje de CPU utilizada(int): 45
Porcentaje de memoria utilizada(int): 56
Cantidad de segundos en promedio para recibir una respuesta(float): 2.8
Cantidad de queries tardan más de 1 segundo(int): 15
Cantidad de conexiones simultáneas(int): 800
==================================================
DIAGNÓSTICO DEL SISTEMA
==================================================

#1 - Latencia de red o problemas de conectividad
Certeza: 70.0%
Recomendaciones:
  • Verificar latencia entre servidor y clientes
  • Implementar CDN para recursos estáticos
  • Optimizar tamaño de respuestas (compresión)
```

---

## ¿Qué aprendemos de este ejemplo?

1. **Estructura del conocimiento**: Las reglas están separadas en métodos independientes, lo que hace el sistema modular y fácil de mantener.

2. **Manejo de incertidumbre**: Cada regla calcula una *"certeza"* basada en qué tan extremos son los valores.

3. **Explicación del razonamiento**: El sistema no solo da un diagnóstico, sino que explica por qué llegó a esa conclusión.

4. **Escalabilidad**: Agregar nuevas reglas es tan simple como crear un nuevo método `_regla_*`.

---

## Código completo

Aquí tienes el código completo del sistema. También puedes encontrarlo en el repositorio de ejemplos haciendo click en el siguiente [enlace](https://github.com/learn-software-engineering/examples/blob/main/ai/module1/sistema_experto/main.py).

{{< github repo="learn-software-engineering/examples" showThumbnail=false >}}

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
