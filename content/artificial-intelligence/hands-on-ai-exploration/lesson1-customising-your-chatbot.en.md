---
draft: true
weight: 3
series: ["Hands-on AI Exploration"]
series_order: 3
title: "Lesson 1: Customising your Chatbot"
authors:
  - jnonino
description: >
  TO COMPLETE
date: 2025-03-11
tags: ["Artificial Intelligence", "Hands-on AI Exploration", "Chatbot"]
---

{{< lead >}}
Now that you have a working chatbot, let's make it uniquely yours!
{{< /lead >}}

### Modifying the System Prompt

The system prompt defines your chatbot's personality and capabilities. Try changing the `SYSTEM_PROMPT` in `app.py` to create different types of chatbots:

**Expert AI Teacher**:
```python
SYSTEM_PROMPT = """
You are an AI expert tasked with teaching beginners about artificial intelligence.
Explain concepts in simple terms with helpful analogies.
When possible, suggest follow-up topics they might find interesting.
"""
```

**Creative Writing Assistant**:
```python
SYSTEM_PROMPT = """
You are a creative writing assistant who helps generate story ideas, characters, and plots.
Be imaginative, original, and inspiring in your responses.
Ask thought-provoking questions to help develop the user's ideas further.
"""
```

**Personal Fitness Coach**:
```python
SYSTEM_PROMPT = """
You are a friendly fitness coach who provides workout advice and motivation.
Give practical, safe exercise suggestions and emphasize the importance of proper form.
Be encouraging and supportive in your tone.
"""
```

### Changing the Model

The model parameter in the OpenAI API call determines which AI model processes your requests. Different models have different capabilities, speeds, and costs:

```python
# More capable but more expensive
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    max_tokens=150
)

# Faster and cheaper
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    max_tokens=150
)
```

### Enhancing the User Interface

You can improve your chatbot's interface by modifying the HTML and CSS in `index.html`. Some ideas:

- Add a dark mode toggle
- Include a clear conversation button
- Show the chat history on page reload
- Add a typing animation when the AI is responding

Example of adding a clear button:

```html
<!-- Add this button below the heading -->
<button id="clear-chat">Clear Conversation</button>

<!-- Add this JavaScript to your script tag -->
<script>
    // Add this inside your DOMContentLoaded function
    const clearButton = document.getElementById('clear-chat');

    clearButton.addEventListener('click', function() {
        chatContainer.innerHTML = '';

        // Reset conversation on server
        fetch('/clear', {
            method: 'POST'
        });
    });
</script>
```

Then add this route to `app.py`:

```python
@app.route('/clear', methods=['POST'])
def clear():
    global conversation_history
    conversation_history = []
    return jsonify({"status": "cleared"})
```

---

## Lesson completion checklist

- [x] Set up your Python development environment.
- [x] Created a basic chatbot using the OpenAI API.
- [x] Built a web interface with Flask.
- [x] Deployed your chatbot online.
- [x] Customized your chatbot's personality.

---
