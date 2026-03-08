---
weight: 2
title: "Operaciones de Entrada y Salida"
cardImage: "featured.es.jpg"
authors:
  - jnonino
description: >
  Las operaciones de entrada y salida (input/output o I/O) permiten que un programa se comunique e intercambie datos con el mundo exterior. En este artículo veremos en detalle operaciones de entrada desde el teclado o un archivo, y salida hacia la pantalla o un archivo.
date: 2023-09-19
tags: ["Programación", "Conceptos Iniciales", "entrada/salida", "I/O"]
---

Las operaciones de entrada y salida (input/output o I/O) permiten que un programa se comunique e intercambie datos con el mundo exterior. En este artículo veremos en detalle operaciones de entrada desde el teclado o un archivo, y salida hacia la pantalla o un archivo.

## Entrada y salida estándar

### Salida a pantalla

Python provee funciones para enviar la salida de un programa a la *"salida estándar"*, generalmente la pantalla o terminal.

La función `print()` imprime en pantalla el valor pasado como parámetro:
```python
nombre = "Julian"
print(nombre) # muestra "Julian"
```

Podemos imprimir múltiples valores separados por comas:
```python
nombre = "Julian"
print("Hola", nombre, "!") # muestra "Hola Julian!"
```

También podemos usar valores literales sin asignar a variables:
```python
print("2 + 3 = ", 2 + 3) # muestra "2 + 3 = 5"
```

#### Formato de salida

Python provee varias formas de dar formato a la salida y así integrar variables a lo que deseamos imprimir en pantalla:

**f-Strings**: Permiten insertar variables dentro de una cadena:
```python
nombre = "Julian"
print(f"Hola {nombre}") # muestra "Hola Julian"
```

**%s**: Inserta cadenas de texto en una cadena de formato:
```python
nombre = "Julian"
print("Hola %s" % nombre) # muestra "Hola Julian"
```

**%d**: Inserta números enteros:
```python
valor = 15
print("El valor es %d" % valor) # muestra "El valor es 15"
```

**.format()**: Inserta valores en una cadena de formato:
```python
nombre = "Julian"
print("¡Hola {}! Bienvenido a {}.".format(nombre, "casa"))
# muestra "¡Hola Julian! Bienvenido a casa."
```

Estas opciones de formato nos permiten interpolar variables y valores en cadenas de texto para generar salidas personalizadas. Podemos combinar múltiples valores y formatos en una sola cadena de salida.

### Entrada desde el teclado

Existen funciones para leer datos ingresados por el usuario en tiempo de ejecución. Esto se conoce como *"entrada estándar"*.

La función `input()` permite leer un valor ingresado por el usuario y asignarlo a una variable. Por ejemplo:
```python
nombre = input("Ingresa tu nombre: ")
```

Esto muestra el mensaje `Ingresa tu nombre: ` y espera a que el usuario escriba un texto y presione *Enter*. Ese valor se asigna a la variable `nombre`.

La función `input()` siempre regresa una cadena de texto. Si queremos pedir un número u otro tipo de dato, debemos convertirlo usando `int()`, `float()`, etc.:

```python
edad = int(input("Ingresa tu edad: "))
pi = float(input("Ingresa el valor de pi: "))
```

#### Leyendo múltiples valores

Podemos pedir y leer varios valores en una misma línea separándolos con comas:

```python
nombre, edad = input("Ingresa nombre y edad: ").split()
```

El método `split()` divide la entrada en partes y retorna una lista de cadenas. Luego asignamos los elementos de la lista a variables separadas.

También podemos leer varias líneas de entrada con un bucle:

```python
nombres = [] # lista vacía

for x in range(3):
   nombre = input("Ingresa un nombre: ")
   nombres.append(nombre)
```

Este código lee 3 nombres ingresados por el usuario y los agrega a la lista `nombres`.

{{< callout type="info" >}}
  ¡No te preocupes por el concepto de bucle y esta nueva sintaxis. Vamos a hablar de eso en próximos artículos!
{{< /callout >}}

## Archivos

### Salida a un archivo

Además de imprimir a pantalla, podemos escribir la salida a un archivo usando la función `open()`:
```python
archivo = open("datos.txt", "w")
```

Esto abre `datos.txt` para escritura (`"w"`) y retorna un objeto archivo.

Luego usamos `archivo.write()` para escribir contenido en el:
```python
archivo.write("Hola mundo!")
archivo.write("Este texto va al archivo")
```

Debemos cerrar el archivo con `archivo.close()` cuando terminamos:
```python
archivo.close()
```

También podemos usar `with` para abrir y cerrar automáticamente:
```python
with open("datos.txt", "w") as archivo:
   archivo.write("Hola mundo!")
   # no hace falta cerrar, es automático
```

### Lectura de archivos

Para leer un archivo usamos `open()` con modo "r" y iteramos sobre el objeto archivo. Esto imprime cada línea, incluyendo los saltos de línea.
```python
with open("datos.txt", "r") as archivo:
   for linea in archivo:
      print(linea) # muestra cada línea del archivo
```

Podemos leer todas las líneas a una lista con `readlines()`:
```python
lineas = archivo.readlines()
print(lineas)
```

Para leer el contenido completo a una cadena usamos `read()`:
```python
texto = archivo.read()
print(texto)
```

También podemos leer un número determinado de bytes o caracteres con `read(n)`.

### Operaciones comunes para el manejo de archivos

Existen varias funciones incorporadas para manipular archivos en Python:
- `open()`: Abre un archivo y retorna un objeto archivo.
- `close()`: Cierra el archivo.
- `write()`: Escribe datos al archivo.
- `read()`: Lee datos del archivo.
- `readline()`: Lee una línea del archivo.
- `truncate()`: Vacía el archivo.
- `seek()`: Mueve la posición de lectura/escritura.
- `rename()`: Renombra el archivo.
- `remove()`: Elimina el archivo.

## Conclusión

En este artículo explicamos en detalle operaciones de entrada y salida en Python, incluyendo leer de entrada estándar y escribir a salida estándar o archivos. Manejar correctamente la entrada y salida es esencial para muchas aplicaciones de Python. Recomiendo practicar con ejemplos propios para dominar estas funciones.

---

{{< callout icon="sparkles" >}}
¡Gracias por llegar hasta acá! Espero que este recorrido por el universo de la programación haya sido tan apasionante para vos como lo fue para mí escribirlo.

Nos encantaría escuchar lo que pensás, así que no te quedes callado/a, dejá tus comentarios, sugerencias y todas esas ideas copadas que seguro se te ocurrieron.

Y para ir más allá de estas líneas, date una vuelta por los ejemplos prácticos que preparamos para vos. Vas a encontrar todo el código y los proyectos en nuestro repositorio de GitHub [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

¡Gracias por ser parte de esta comunidad de aprendizaje. Seguí programando y explorando nuevos territorios en este fascinante mundo de la computación!
{{< /callout >}}
