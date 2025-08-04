---
draft: true
weight: 4
series: ["Hands-on AI Exploration"]
series_order: 4
title: "Lesson 1: Deploying your Chatbot online"
authors:
  - jnonino
description: >
  TO COMPLETE
date: 2025-03-11
tags: ["Artificial Intelligence", "Hands-on AI Exploration", "Chatbot", "Deploy"]
---

{{< lead >}}
Deployment means making your application available online for others to use. There are many services that make this easy, even for beginners.
{{< /lead >}}

## Deploying with Render (Free Option)

[Render](https://render.com/) offers a free tier that's perfect for simple projects.

1. Create a file called `requirements.txt` (if you haven't already) with the required packages:
   ```
   flask
   openai
   python-dotenv
   gunicorn
   ```

2. Create a new file called `wsgi.py`:
   ```python
   from app import app

   if __name__ == "__main__":
       app.run()
   ```

3. Sign up for a free account on [Render](https://render.com/)

4. Create a new `"Web Service"` and connect your GitHub repository (you'll need to push your code to GitHub first).

5. Configure your service:
   - Name: `my-first-ai-chatbot` (or any name you prefer)
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn wsgi:app`

6. Add your OPENAI_API_KEY as an environment variable in the Render dashboard

7. Deploy! Render will provide you with a URL where your chatbot is accessible

## Alternative: Deploying with Replit (Beginner-Friendly)

[Replit](https://replit.com/) is an online IDE that makes deployment extremely easy:

1. Create an account on Replit
2. Create a new Repl, choosing "Python" as the language
3. Upload your files or copy/paste them into Replit
4. Add your API key in the Secrets tab (key: OPENAI_API_KEY, value: your_api_key)
5. Click "Run" to deploy your application
6. Replit will provide a URL to your running application

---

## Lesson completion checklist

- [x] Set up your Python development environment.
- [x] Created a basic chatbot using the OpenAI API.
- [x] Built a web interface with Flask.
- [x] Deployed your chatbot online.
- [ ] Customized your chatbot's personality.

---
