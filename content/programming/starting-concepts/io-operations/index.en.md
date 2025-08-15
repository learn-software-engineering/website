---
weight: 2
series: ["Programming: Learning the Starting Concepts"]
series_order: 2
title: "Input and output operations"
authors:
  - jnonino
description: >
  Input/output operations (I/O) allow a program to communicate and exchange data with the outside world. In this article we will see in detail input operations from the keyboard or a file, and output to the screen or a file.
date: 2023-09-19
tags: ["Programming", "Starting Concepts", "input/output", "I/O"]
---

## Screen output

Python also provides functions to send program output to "standard output", usually the screen or terminal.

The `print()` function displays the value passed as a parameter:

```python
name = "Eric"
print(name) # displays "Eric"
```

We can print multiple values separated by commas:

```python
print("Hello", name, "!") # displays "Hello Eric!"
```

We can also use literal values without assigning to variables:

```python
print("2 + 3 =", 2 + 3) # displays "2 + 3 = 5"
```

### Output formatting

Python provides various ways to format output:

**f-Strings**: Allow inserting variables into a string:

```python
name = "Eric"
print(f"Hello {name}") # displays "Hello Eric"
```

**%s**: Inserts string text into a format string:

```python
name = "Eric"
print("Hello %s" % name) # displays "Hello Eric"
```

**%d**: Inserts integer numbers:

```python
value = 15
print("The value is %d" % value) # displays "The value is 15"
```

**.format()**: Inserts values into a format string:

```python
name = "Eric"
print("Hello {}. Welcome".format(name))
# displays "Hello Eric. Welcome"
```

These formatting options allow us to interpolate variables and values into text strings to generate custom outputs. We can combine multiple values and formats in a single output string.

---

## Keyboard input

Python provides built-in functions to read data entered by the user at runtime. This is known as "standard input".

The `input()` function allows reading a value entered by the user and assigning it to a variable. For example:

```python
name = input("Enter your name: ")
```

This displays the message "Enter your name: " and waits for the user to enter text and press Enter. That value is assigned to the `name` variable.

The `input()` function always returns a string. If we want to ask for a number or other data type, we must convert it using `int()`, `float()`, etc:

```python
age = int(input("Enter your age: "))
pi = float(input("Enter the value of pi: "))
```

### Reading multiple values

We can ask for and read multiple values on the same line separating them with commas:

```python
name, age = input("Enter name and age: ").split()
```

The `split()` method divides the input into parts and returns a list of strings. We then assign the list elements to separate variables.

We can also read multiple lines of input with a loop:

```python
names = [] # empty list

for x in range(3):
   name = input("Enter a name: ")
   names.append(name)
```

This code reads 3 names entered by the user and adds them to a list.

---

## Output to a file

In addition to printing to the screen, we can write output to a file using the `open()` function:

```python
file = open("data.txt", "w")
```

This opens `data.txt` for writing ("w") and returns a file object.

Then we use `file.write()` to write to that file:

```python
file.write("Hello World!")
file.write("This text goes to the file")
```

We must close the file with `file.close()` when finished:

```python
file.close()
```

We can also use `with` to open and automatically close:

```python
with open("data.txt", "w") as file:
   file.write("Hello World!")
   # no need to close, it's automatic
```

---

## Reading files

To read a file we use `open()` in "r" mode and iterate over the file object:

```python
with open("data.txt", "r") as file:
   for line in file:
      print(line) # prints each line of the file
```

This prints each line, including newlines.

We can read all lines to a list with `readlines()`:

```python
lines = file.readlines()
print(lines)
```

To read the full content to a string we use `read()`:

```python
text = file.read()
print(text)
```

We can also read a specific number of bytes or characters with `read(n)`.

---

## File handling operations

There are several built-in functions to handle files in Python:

- `open()` - Opens a file and returns a file object
- `close()` - Closes the file
- `write()` - Writes data to the file
- `read()` - Reads data from the file
- `readline()` - Reads a line from the file
- `truncate()` - Empties the file
- `seek()` - Changes the reading/writing position
- `rename()` - Renames the file
- `remove()` - Deletes the file

These allow us to perform advanced operations to read, write and maintain files.

---

## Conclusion

In this article we explained Python input and output operations in detail, including reading from standard input and writing to standard output or files. Properly handling input and output is essential for many Python applications. I recommend practising with your own examples to master these functions.

---

## References

1. Downey, A. B. (2015). *Think Python: How to think like a computer scientist*. Needham, Massachusetts: Green Tea Press.
2. McKinney, W. (2018). *Python for data analysis: Data wrangling with Pandas, NumPy, and IPython*. O'Reilly Media.
3. Matthes, E. (2015). *Python crash course: A hands-on, project-based introduction to programming*. No Starch Press.
4. Lutz, M. (2013). *Learning Python: Powerful Object-Oriented Programming*. O'Reilly Media, Incorporated.

---

{{< alert icon="comment" cardColor="grey" iconColor="black" textColor="black" >}}
Cheers for making it this far! I hope this journey through the programming universe has been as fascinating for you as it was for me to write down.

We're keen to hear your thoughts, so don't be shy – drop your comments, suggestions, and those bright ideas you're bound to have.

Also, to delve deeper than these lines, take a stroll through the practical examples we've cooked up for you. You'll find all the code and projects in our GitHub repository [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Thanks for being part of this learning community. Keep coding and exploring new territories in this captivating world of software!
{{< /alert >}}

---
