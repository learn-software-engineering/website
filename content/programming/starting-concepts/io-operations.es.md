---
weight: 2
series: ["Conceptos Iniciales"]
series_order: 2
title: "Operaciones de Entrada y Salida"
authors:
  - jnonino
description: >
  Las operaciones de entrada y salida (input/output o I/O) permiten que un programa se comunique e intercambie datos con el mundo exterior. En este artículo veremos en detalle operaciones de entrada desde el teclado o un archivo, y salida hacia la pantalla o un archivo.
date: 2023-09-19
tags: ["Programación", "Conceptos Iniciales", "entrada/salida", "I/O"]
---

## Salida a pantalla

Python también provee funciones para enviar la salida de un programa a la "salida estándar", generalmente la pantalla o terminal[^2].

La función `print()` muestra el valor pasado como parámetro:

```python
nombre = "Eric"
print(nombre) # muestra "Eric"
```

Podemos imprimir múltiples valores separados por comas[^4]:

```python
print("Hola", nombre, "!") # muestra "Hola Eric!"
```

También podemos usar valores literales sin asignar a variables[^3]:

```python
print("2 + 3 =", 2 + 3) # muestra "2 + 3 = 5"
```

### Formateo de salida

Python provee varias formas de dar formato a la salida[^1]:

**f-Strings**: Permiten insertar variables dentro de una cadena:

```python
nombre = "Eric"
print(f"Hola {nombre}") # muestra "Hola Eric"
```

**%s**: Inserta cadenas de texto en una cadena de formato:

```python
nombre = "Eric"
print("Hola %s" % nombre) # muestra "Hola Eric"
```

**%d**: Inserta números enteros:

```python
valor = 15
print("El valor es %d" % valor) # muestra "El valor es 15"
```

**.format()**: Inserta valores en una cadena de formato:

```python
nombre = "Eric"
print("Hola {}. Bienvenido".format(nombre))
# muestra "Hola Eric. Bienvenido"
```

Estas opciones de formateo nos permiten interpolar variables y valores en cadenas de texto para generar outputs personalizados. Podemos combinar múltiples valores y formateos en una sola cadena de salida[^4].

---

## Entrada desde el teclado

Python provee funciones incorporadas para leer datos ingresados por el usuario en tiempo de ejecución. Esto se conoce como "entrada estándar"[^1].

La función `input()` permite leer un valor ingresado por el usuario y asignarlo a una variable. Por ejemplo:

```python
nombre = input("Ingresa tu nombre: ")
```

Esto muestra el mensaje "Ingresa tu nombre: " y espera a que el usuario escriba un texto y presione Enter. Ese valor se asigna a la variable `nombre`[^4].

La función `input()` siempre regresa una cadena de texto. Si queremos pedir un número u otro tipo de dato, debemos convertirlo usando `int()`, `float()`, etc[^2]:

```python
edad = int(input("Ingresa tu edad: "))
pi = float(input("Ingresa el valor de pi: "))
```

### Leyendo múltiples valores

Podemos pedir y leer varios valores en una misma línea separándolos con comas[^3]:

```python
nombre, edad = input("Ingresa nombre y edad: ").split()
```

El método `split()` divide la entrada en partes y retorna una lista de cadenas. Luego asignamos los elementos de la lista a variables separadas.

También podemos leer varias líneas de entrada con un ciclo[^1]:

```python
nombres = [] # lista vacía

for x in range(3):
   nombre = input("Ingresa un nombre: ")
   nombres.append(nombre)
```

Este código lee 3 nombres ingresados por el usuario y los agrega a una lista.

---

## Salida a un archivo

Además de imprimir a pantalla, podemos escribir la salida a un archivo usando la función `open()`[^2]:

```python
archivo = open("datos.txt", "w")
```

Esto abre `datos.txt` para escritura ("w") y retorna un objeto archivo.

Luego usamos `archivo.write()` para escribir a ese archivo[^3]:

```python
archivo.write("Hola mundo!")
archivo.write("Este texto va al archivo")
```

Debemos cerrar el archivo con `archivo.close()` cuando terminamos[^1]:

```python
archivo.close()
```

También podemos usar `with` para abrir y cerrar automáticamente[^4]:

```python
with open("datos.txt", "w") as archivo:
   archivo.write("Hola mundo!")
   # no hace falta cerrar, es automático
```

---

## Lectura de archivos

Para leer un archivo usamos `open()` con modo "r" y iteramos sobre el objeto archivo[^2]:

```python
with open("datos.txt", "r") as archivo:
   for linea in archivo:
      print(linea) # muestra cada línea del archivo
```

Esto imprime cada línea, incluyendo los saltos de línea.

Podemos leer todas las líneas a una lista con `readlines()`[^3]:

```python
linenas = archivo.readlines()
print(linenas)
```

Para leer el contenido completo a una cadena usamos `read()`[^1]:

```python
texto = archivo.read()
print(texto)
```

También podemos leer un número determinado de bytes o caracteres con `read(n)`[^4].

---

## Operaciones para el manejo de archivos

Existen varias funciones incorporadas para manipular archivos en Python[^2]:

- `open()` - Abre un archivo y retorna un objeto archivo
- `close()` - Cierra el archivo
- `write()` - Escribe datos al archivo
- `read()` - Lee datos del archivo
- `readline()` - Lee una línea del archivo
- `truncate()` - Vacía el archivo
- `seek()` - Mueve la posición de lectura/escritura
- `rename()` - Renombra el archivo
- `remove()` - Elimina el archivo

Estas funciones nos permiten efectuar operaciones avanzadas para leer, escribir y mantener archivos.

---

## Conclusión

En este artículo explicamos en detalle operaciones de entrada y salida en Python, incluyendo leer de entrada estándar y escribir a salida estándar o archivos. Manejar correctamente la entrada y salida es esencial para muchas aplicaciones de Python. Recomiendo practicar con ejemplos propios para dominar estas funciones[^3].

---

{{% alert color="info" %}}
¡Felicitaciones por llegar hasta acá! Espero que este recorrido por el universo de la programación te haya resultado tan interesante como lo fue para mí al escribirlo.

Queremos conocer tu opinión, así que no dudes en compartir tus comentarios, sugerencias y esas ideas brillantes que seguro tenés.

Además, para explorar más allá de estas líneas, date una vuelta por los ejemplos prácticos que armamos para vos. Todo el código y los proyectos los encontrarás en nuestro repositorio de GitHub [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Gracias por ser parte de esta comunidad de aprendizaje. ¡Seguí programando y explorando nuevas areas en este fascinante mundo del software!
{{% /alert %}}

---

## Referencias

[^1]: Downey, A. B. (2015). *Think Python: How to think like a computer scientist*. Needham, Massachusetts: Green Tea Press.
[^2]: McKinney, W. (2018). *Python for data analysis: Data wrangling with Pandas, NumPy, and IPython*. O'Reilly Media.
[^3]: Matthes, E. (2015). *Python crash course: A hands-on, project-based introduction to programming*. No Starch Press.
[^4]: Lutz, M. (2013). *Learning Python: Powerful Object-Oriented Programming*. O'Reilly Media, Incorporated.
