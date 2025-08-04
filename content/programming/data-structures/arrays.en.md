---
weight: 2
series: ["Data Structures"]
series_order: 2
title: "Arrays"
authors:
  - jnonino
description: >
  Arrays are fundamental data structures in programming that allow storing and organizing collections of data of the same type. Mastering the use of arrays is essential for any programmer.
date: 2023-10-30
tags: ["Programming", "Data Structures", "Arrays"]
---

An array is a data structure that represents a set of elements, which are accessed through contiguous numeric indices ranging from 0 to the size of the array minus 1. Arrays provide fast, direct access to elements based on their position.

In languages like Python and Ruby, arrays are known as 'lists'. In Javascript they are known as 'arrays'.

Arrays are typically homogeneous, storing elements of the same type like integers, strings, etc. Some languages allow heterogeneous arrays with values of different types.

![Diagram of an array](diagram-arrays.jpeg)

### Creating arrays

The way to create arrays varies according to the programming language:

```python
MY_ARRAY = ["A", "B", "C"] # array literal
my_array = list(range(5)) # array from range
```

When creating a literal array its elements are initialized directly. When constructing an empty array its size is specified but its elements are initialized with a default value (0 for numbers, null for objects, etc).

### Accessing and modifying elements

Individual elements are quickly accessed by their index using brackets `[]`:

```python
my_array = ['a', 'b', 'c']
print(my_array[0]) # 'a'
print(my_array[2]) # 'c'
my_array[2] = 'z'
print(my_array[2]) # 'z'
```

Indices start at 0, so in an array of size N, valid indices are between 0 and N-1.

Accessing an invalid index causes an error, for example, accessing index 3 in an array of size 3. This is known as "index out of bounds".

### Traversing an array

We can traverse all elements using a `for` loop:

```python
letters = ['a', 'b', 'c']

for i in range(len(letters)):
  print(letters[i])
```

This prints each element in order. `len()` returns the total length of the array.

Another way is by directly iterating over the elements:

```python
letters = ['a', 'b', 'c']

for letter in letters:
  print(letter)
```

---

## Searching in an array

We can search for an element in an array using a loop and comparing element by element:

```python
letters = ['a', 'b', 'c']

def search_in_array(array, element):
  for i in range(len(array)):
    if array[i] == element:
      return i
  return False

print(search_in_array(letters, 'b')) # 1
print(search_in_array(letters, 'z')) # False
```

It returns the index if found or `False` if not found.

## Multidimensional array

Arrays can have more than one dimension, for example, 2D matrices, 3D cubes, etc.

A 2D array can be seen as a table with rows and columns. To access an element two indices are specified, one for the row and one for the column:

```python
matrix = [
  [1, 2, 3],
  [4, 5, 6]
]

print(matrix[0][2]) # 3
print(matrix[1][0]) # 4
```

They can have more dimensions, for example a 3D array to represent pixels in an image.

---

## Conclusion

Arrays are fundamental data structures in programming that provide efficient access to elements in memory through numeric indices. Having a good command of arrays, matrices, and their uses is essential for any programmer.

---

{{% alert color="info" %}}
Cheers for making it this far! I hope this journey through the programming universe has been as fascinating for you as it was for me to write down.

We're keen to hear your thoughts, so don't be shy – drop your comments, suggestions, and those bright ideas you're bound to have.

Also, to delve deeper than these lines, take a stroll through the practical examples we've cooked up for you. You'll find all the code and projects in our GitHub repository [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Thanks for being part of this learning community. Keep coding and exploring new territories in this captivating world of software!
{{% /alert %}}
