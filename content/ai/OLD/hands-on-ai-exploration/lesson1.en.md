---
draft: true
weight: 1
series: ["AI: Hands-on Exploration"]
series_order: 1
title: "Lesson 1: Build your first AI application"
authors:
  - jnonino
description: >
  Welcome to your AI journey! In this lesson, you'll create your first functioning AI application without needing any advanced mathematics or extensive programming background. By the end of this lesson, you'll have built a simple but powerful chatbot using modern AI technologies, set up your development environment, and deployed your creation online where you can share it with others. The goal is to give you an immediate, tangible experience with AI.
date: 2025-03-11
tags: ["Artificial Intelligence", "Hands-on AI Exploration"]
---

{{< lead >}}
Welcome to your AI journey! In this lesson, you'll create your first functioning AI application without needing any advanced mathematics or extensive programming background. By the end of this lesson, you'll have built a simple but powerful chatbot using modern AI technologies, set up your development environment, and deployed your creation online where you can share it with others. The goal is to give you an immediate, tangible experience with AI. Let's get started!
{{< /lead >}}

## Setting up your development environment

For your developer environment you'll need:
- A computer with internet access
- Basic familiarity with using a computer and web browser
- Curiosity and willingness to experiment!

### Installing Python

Python is the programming language of choice for AI development due to its simplicity and the vast ecosystem of AI libraries available.

1. Visit [python.org](https://www.python.org/downloads/) and download the latest stable version for your operating system.
2. Follow the installation instructions, making sure to check "Add Python to PATH" during installation on Windows.
3. Verify your installation by opening a terminal/command prompt and typing:
   ```bash
   python --version
   ```
   or on some systems:
   ```bash
   python3 --version
   ```

### Setting up a virtual environment

Virtual environments help keep your projects organized and prevent conflicts between different projects.

1. Open your terminal/command prompt and navigate to where you want to create your project folder:
   ```bash
   mkdir my_first_ai_app
   cd my_first_ai_app
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env
   ```
   or on some systems:
   ```bash
   python3 -m venv env
   ```

3. Activate the virtual environment:
   - On Windows:
     ```cmd
     env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source env/bin/activate
     ```

When activated, you should see `(env)` at the beginning of your command line prompt. To deactivate, just type `deactivate` on the terminal.

### Installing a code editor

While you can write code in any text editor, specialized code editors make programming much easier.

**Recommended Option: Visual Studio Code**
1. Download and install [Visual Studio Code](https://code.visualstudio.com/).
2. Open VS Code and install the Python extension (search for "Python" in the Extensions tab).
3. Open your project folder in VS Code: File > Open Folder > select `my_first_ai_app`.

---

## Understanding AI APIs and Language Models

### What are Large Language Models?

Large Language Models (LLMs) like GPT-4, Claude, and Llama are AI systems trained on vast amounts of text data. They can understand and generate human-like text, answer questions, write creative content, translate languages, and much more.

These models work by predicting what text should come next given a starting prompt or conversation. While the underlying technology is complex, we can use them through simple APIs (Application Programming Interfaces) without needing to understand all the technical details.

### API keys and access

To use commercial AI services, you'll need an API key, which is like a password that allows your application to access the service.

For this lesson, we'll use OpenAI's API, which powers ChatGPT, but the concepts apply to other services like Anthropic's Claude, Cohere, or open-source models.

1. Visit [OpenAI's website](https://platform.openai.com/signup) and create an account.
2. Navigate to API Keys section and create a new secret key.
3. **IMPORTANT**: Copy your API key and store it securely, you won't be able to view it again!

---

## Lesson completion checklist

- [x] Set up your Python development environment.
- [ ] Created a basic chatbot using the OpenAI API.
- [ ] Built a web interface with Flask.
- [ ] Deployed your chatbot online.
- [ ] Customized your chatbot's personality.

---
