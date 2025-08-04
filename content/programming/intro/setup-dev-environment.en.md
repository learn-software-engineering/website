---
weight: 5
series: ["Introduction to Programming"]
series_order: 5
title: "Set Up your Development Environment"
authors:
  - jnonino
description: >
  Venturing into the world of programming might seem like a Herculean task, especially when faced with the initial decision: Where to begin? This article will guide you through the essential steps to set up your programming environment, ensuring a solid foundation for your coding journey.
date: 2023-09-10
tags: ["Programming", "Python", "IDE", "Tools"]
# images:
#   - /images/banners/content/programming/0100-intro/0104-setup-dev-environment.png
---

## Choosing a programming language

Choosing a programming language is the first and perhaps the most crucial step in the learning process. Several factors to consider when selecting a language include:

1. **Purpose**: What do you want to code for? If it's web development, JavaScript or PHP might be good options. If you're into data science, R or Python might be more appropriate.
2. **Community**: A language with an active community can be vital for beginners. A vibrant community usually means more resources, tutorials, and solutions available online.
3. **Learning curve**: Some languages are easier to pick up than others. It's essential to pick one that matches your experience level and patience.
4. **Job opportunities**: If you're eyeing a career in programming, researching the job market demand for various languages can be insightful.

While there are many valuable and potent languages, for the purpose of this course, we've chosen [**Python**](https://www.python.org/). This language is renowned for its simplicity and readability, making it ideal for those just starting out. Moreover, Python boasts an active community and a wide range of applications, from web development to artificial intelligence.

### Installing Python

#### For Windows users:

1. Download the installer:
   - Visit the official Python website at [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)
   - Click on the download link for the latest version of Python 3.x.
2. Run the installer:
   - Once the download is complete, locate and run the installer `.exe` file.
   - Make sure to check the box that says "Add Python to PATH" during installation. This step is crucial for making Python accessible from the Command Prompt.
   - Follow the installation prompts.
3. Verify installation:
   - Open the Command Prompt and type:
     ```bash
     python --version
     ```
   - This should display the version of Python you just installed.

#### For Mac users:

1. Download the installer:
   - Visit the official Python website at [https://www.python.org/downloads/mac-osx/](https://www.python.org/downloads/mac-osx/)
   - Click on the download link for the latest version of Python 3.x.
2. Run the installer:
   - Once the download is complete, locate and run the `.pkg` file.
   - Follow the installation prompts.
3. Verify installation:
   - Open the Terminal and type:
     ```bash
     python3 --version
     ```
   - This should display the version of Python you just installed.

#### For Linux (Ubuntu/Debian) users:

1. Update packages:
   ```bash
   sudo apt update
   ```
2. Install Python:
   ```bash
   sudo apt install python3
   ```
3. Verify installation:
   - After installation, you can check the version of Python installed by typing:
     ```bash
     python3 --version
     ```

---

## Integrated Development Environments (IDEs)

An IDE is a tool that streamlines application development by combining commonly-used functionalities into a single software package: code editor, compiler, debugger, and more. Choosing the right IDE can make the programming process more fluid and efficient.

When evaluating IDEs, consider:

1. **Language compatibility**: Not all IDEs are compatible with every programming language.
2. **Features**: Some IDEs offer features like auto-completion, syntax highlighting, and debugging tools.
3. **Extensions and plugins**: Being able to customize and extend your IDE through plugins can be extremely beneficial.
4. **Price**: There are free and paid IDEs. Evaluate whether the additional features of a paid IDE justify its cost.

For this course, we've selected [**Visual Studio Code (VS Code)**](https://code.visualstudio.com/). It's a popular IDE that's free and open-source. It's known for its straightforward interface, a vast array of plugins, and its capability to handle multiple programming languages. Its active community ensures regular updates and a plethora of learning resources.

### Installing Visual Studio Code

#### For Windows users:

1. Download the installer:
   - Visit the official VS Code website at [https://code.visualstudio.com/](https://code.visualstudio.com/)
   - Click on the "Download for Windows" button.
2. Run the installer:
   - Once the download is complete, locate and run the installer `.exe` file.
   - Follow the installation prompts, including accepting the license agreement and choosing the installation location.
3. Launch VS Code:
   - After installation, you can find VS Code in your Start menu.
   - Launch it, and you're ready to start coding!

#### For Mac users:

1. Download the installer:
   - Visit the official VS Code website at [https://code.visualstudio.com/](https://code.visualstudio.com/)
   - Click on the "Download for Mac" button.
2. Install VS Code:
   - Once the download is complete, open the downloaded `.zip` file.
   - Drag the Visual Studio Code `.app` to the `Applications` folder, making it available in the Launchpad.
3. Launch VS Code:
   - Use Spotlight search or navigate to your Applications folder to launch VS Code.

#### For Linux (Ubuntu/Debian) users:

1. Update packages and install dependencies:
   ```bash
   sudo apt update
   sudo apt install software-properties-common apt-transport-https wget
   ```
2. Download and install the key:
   ```bash
   wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -
   ```
3. Add the VS Code repository:
   ```bash
   sudo add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"
   ```
4. Install Visual Studio Code:
   ```bash
   sudo apt update
   sudo apt install code
   ```
5. Launch VS Code:
   - You can start VS Code from the terminal by typing `code` or find it in your list of installed applications.

---

## Write and execute your first program

Once you've set up your programming environment, it's time to dive into coding.

### Hello, World!

This is arguably the most iconic program for beginners. It's simple, but it introduces you to the process of writing and executing code.

```python
print("Hello, World!")
```

![Running the Hello World program](/images/content/programming/intro/running-hello-world.png)

### Triangle area and perimeter calculation

This program is a tad more intricate. It doesn't just print out a message; it also performs mathematical calculations.

```python
# User input
side1 = float(input("Enter the length of the first side: "))
side2 = float(input("Enter the length of the second side: "))
side3 = float(input("Enter the length of the third side: "))

# Perimeter calculation
perimeter = side1 + side2 + side3

# Area calculation using Heron's formula
s = perimeter / 2
area = (s*(s-side1)*(s-side2)*(s-side3)) ** 0.5

print(f"The triangle's perimeter is: {perimeter}")
print(f"The triangle's area is: {area:.2f}")
```

![Running the Triangle program](/images/content/programming/intro/running-triangle.png)

---

## Conclusion

Setting up a programming environment might appear daunting at first, but with the right tools and resources, it becomes a manageable and rewarding task. We hope this article provided you with a solid foundation to kickstart your programming journey. Happy coding!

---

## References

1. Lutz, M. (2013). *Learning Python*. O'Reilly Media.
2. Microsoft. (2020). *Visual Studio Code Documentation*. Microsoft Docs.

---

{{% alert color="info" %}}
Cheers for making it this far! I hope this journey through the programming universe has been as fascinating for you as it was for me to write down.

We're keen to hear your thoughts, so don't be shy – drop your comments, suggestions, and those bright ideas you're bound to have.

Also, to delve deeper than these lines, take a stroll through the practical examples we've cooked up for you. You'll find all the code and projects in our GitHub repository [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Thanks for being part of this learning community. Keep coding and exploring new territories in this captivating world of software!
{{% /alert %}}

---
