---
weight: 3
series: ["Programming: Learning Data Structures"]
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
    src="images/maps.jpg"
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
Cheers for making it this far! I hope this journey through the programming universe has been as fascinating for you as it was for me to write down.

We're keen to hear your thoughts, so don't be shy – drop your comments, suggestions, and those bright ideas you're bound to have.

Also, to delve deeper than these lines, take a stroll through the practical examples we've cooked up for you. You'll find all the code and projects in our GitHub repository [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Thanks for being part of this learning community. Keep coding and exploring new territories in this captivating world of software!
{{< /alert >}}
