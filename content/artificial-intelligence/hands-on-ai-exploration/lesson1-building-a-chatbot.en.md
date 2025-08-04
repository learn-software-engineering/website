---
draft: true
weight: 2
series: ["Hands-on AI Exploration"]
series_order: 2
title: "Lesson 1: Building a Chatbot"
authors:
  - jnonino
description: >
  In this document we are going to see the files and the code required to implement our first AI Chatbot. We are going to use Python, some HTML, Open AI and Flask.
date: 2025-03-11
tags: ["Artificial Intelligence", "Hands-on AI Exploration", "Chatbot"]
---

{{< lead >}}
In this document we are going to see the files and the code required to implement our first AI Chatbot. We are going to use Python, some HTML, Open AI and Flask.
{{< /lead >}}

### Setting up the project

1. Create a new file in your project folder called `requirements.txt` and add these dependencies:
   ```
   openai
   python-dotenv
   flask
   ```

2. In your terminal (with virtual environment activated), install these packages:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in your project folder to securely store your API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
   Replace `your_api_key_here` with the API key you copied earlier.

4. Create a file called `.gitignore` (if you plan to use Git) and add:
   ```
   .env
   __pycache__/
   env/
   ```
   This prevents your API key from being accidentally shared.

### Creating the Chatbot backend

Create a new file called `app.py` with the following code:

```python
import os
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the chatbot's personality and capabilities
SYSTEM_PROMPT = """
You are a helpful AI assistant designed to answer questions about artificial intelligence.
You are friendly, informative, and concise in your responses.
"""

conversation_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')

    # Add user message to conversation history
    conversation_history.append({"role": "user", "content": user_message})

    # Prepare messages for API call
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

    try:
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # We're using gpt-4o-mini to keep costs down (check https://openai.com/api/pricing/ before any executions)
            messages=messages,
            max_tokens=150  # Limit response length to keep the costs at minimum
        )

        # Extract the AI's response
        ai_message = response.choices[0].message.content

        # Add AI response to conversation history
        conversation_history.append({"role": "assistant", "content": ai_message})

        return jsonify({"response": ai_message})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

This code:
1. Sets up a Flask web server
2. Creates an endpoint that accepts messages from users
3. Sends those messages to the OpenAI API
4. Returns the AI's response
5. Keeps track of the conversation history

### Creating the User Interface

Create a folder called `templates` in your project folder, then create a file called `index.html` inside it:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My First AI Chatbot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        #chat-container {
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
            height: 400px;
            overflow-y: auto;
            margin-bottom: 10px;
            background-color: #f9f9f9;
        }
        .message {
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 18px;
            max-width: 70%;
            clear: both;
        }
        .user-message {
            background-color: #DCF8C6;
            float: right;
        }
        .ai-message {
            background-color: #FFFFFF;
            float: left;
            box-shadow: 0 1px 1px rgba(0,0,0,0.1);
        }
        #message-form {
            display: flex;
        }
        #user-input {
            flex-grow: 1;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            margin-right: 10px;
        }
        button {
            padding: 10px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        #thinking {
            display: none;
            margin: 10px 0;
            font-style: italic;
            color: #888;
        }
    </style>
</head>
<body>
    <h1>My First AI Chatbot</h1>
    <p>Ask me anything about artificial intelligence!</p>

    <div id="chat-container"></div>
    <div id="thinking">AI is thinking...</div>

    <form id="message-form">
        <input type="text" id="user-input" placeholder="Type your message here..." required>
        <button type="submit">Send</button>
    </form>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const chatContainer = document.getElementById('chat-container');
            const messageForm = document.getElementById('message-form');
            const userInput = document.getElementById('user-input');
            const thinking = document.getElementById('thinking');

            messageForm.addEventListener('submit', function(e) {
                e.preventDefault();

                const message = userInput.value.trim();
                if (!message) return;

                // Add user message to chat
                addMessage(message, 'user');
                userInput.value = '';

                // Show "thinking" indicator
                thinking.style.display = 'block';

                // Send message to backend
                fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message }),
                })
                .then(response => response.json())
                .then(data => {
                    // Hide "thinking" indicator
                    thinking.style.display = 'none';

                    if (data.error) {
                        addMessage('Sorry, an error occurred: ' + data.error, 'ai');
                    } else {
                        addMessage(data.response, 'ai');
                    }
                })
                .catch(error => {
                    thinking.style.display = 'none';
                    addMessage('Sorry, an error occurred while connecting to the server.', 'ai');
                    console.error('Error:', error);
                });
            });

            function addMessage(text, sender) {
                const messageElement = document.createElement('div');
                messageElement.classList.add('message');
                messageElement.classList.add(sender + '-message');
                messageElement.textContent = text;

                chatContainer.appendChild(messageElement);

                // Scroll to bottom of chat
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        });
    </script>
</body>
</html>
```

This HTML creates a simple but attractive chat interface that communicates with your Flask backend.

### Running your Chatbot locally

1. In your terminal (with virtual environment activated), run:
   ```
   python app.py
   ```

2. Open your web browser and go to `http://127.0.0.1:5000/`

3. You should see your chatbot interface! Try asking it questions like:
   - "What is artificial intelligence?"
   - "How do neural networks work?"
   - "What can I do with AI?"

---

## Lesson completion checklist

- [x] Set up your Python development environment.
- [x] Created a basic chatbot using the OpenAI API.
- [x] Built a web interface with Flask.
- [ ] Deployed your chatbot online.
- [ ] Customized your chatbot's personality.

---
