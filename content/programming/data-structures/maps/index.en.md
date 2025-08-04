---
weight: 3
series: ["Data Structures"]
series_order: 3
title: "Maps (Dictionaries)"
authors:
  - jnonino
description: >
  Maps (also called hashes or dictionaries) are data structures that associate keys with values. They allow ultra fast access to elements through unique keys. In Python they are known as dictionaries.
date: 2023-10-30
tags: ["Programming", "Data Structures", "Maps", "Dictionaries"]
---

A dictionary, or map, consists of a collection of key-value pairs. The key is used to access the associated value. Keys must be unique within a dictionary. Values can be repeated.

{{< figure
    src="maps.jpg"
    alt="Diagram of a map"
    caption="Diagram of a map"
    >}}

---

## Main operations

- **Add/update:** Inserts a key-value pair. If the key existed, its value is replaced.
    ```python
    dictionary['key'] = 'value'
    ```
- **Get value:** Accesses the value given a key.
    ```python
    value = dictionary['key']
    ```
- **Delete:** Removes a key-value pair from the dictionary.
    ```python
    del dictionary['key']
    ```
- **Traverse:** Iterate over the keys, values or pairs of the dictionary.
    ```python
    for key in dictionary:
      print(key, dictionary[key]) # key, value
    ```

---

## Creating a dictionary or map

The syntax for creating maps or dictionaries in Python is:

```python
empty_dictionary = {}

person = {
  'name': 'John',
  'age': 25
}
```

---

## Usage examples

Dictionaries are useful in many cases. Below are some examples.

### Objects and mappings

We can model objects and entities with key-value attributes:

```python
product = {
  'name': 'Smartphone',
  'price': 500,
  'brand': 'XYZ'
}
```

### Counts and frequencies

Counting occurrences of elements in sequences:

```python
text = "Hello world world"

frequencies = {}

for word in text.split():
    if word in frequencies:
        frequencies[word] += 1
    else:
        frequencies[word] = 1

print(frequencies)
# {'Hello': 1, 'world': 2}
```

### Storing and accessing data

As a high performance alternative to lists and arrays.

---

## Conclusion

Dictionaries are versatile data structures thanks to their fast access based on unique keys. They have uses in almost all programs, so mastering dictionaries is essential in any language.

---

{{< alert icon="comment" cardColor="grey" iconColor="black" textColor="black" >}}
¡Felicitaciones por llegar hasta acá! Espero que este recorrido por el universo de la programación te haya resultado tan interesante como lo fue para mí al escribirlo.

Queremos conocer tu opinión, así que no dudes en compartir tus comentarios, sugerencias y esas ideas brillantes que seguro tenés.

Además, para explorar más allá de estas líneas, date una vuelta por los ejemplos prácticos que armamos para vos. Todo el código y los proyectos los encontrarás en nuestro repositorio de GitHub [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Gracias por ser parte de esta comunidad de aprendizaje. ¡Seguí programando y explorando nuevas areas en este fascinante mundo del software!
{{< /alert >}}
