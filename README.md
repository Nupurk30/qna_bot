# Simple RAG Project

This project provides two ways to run an "Ask Your Docs" application.

1.  **Mode 1: API Backend (Docker)**: A scalable backend API (FastAPI) that runs in Docker. You must build a separate frontend (like `ui.py`) to talk to it.
2.  **Mode 2: All-in-One App (Streamlit)**: A single, simple web app (also `ui.py`) that lets users upload their *own* PDFs. **This is the easiest way to demo the project.**

---

## 🚀 Mode 2: Run the All-in-One Streamlit App (Easiest)

This version lets anyone upload their own PDF and ask questions.

### 1. Setup
Clone the repo and set up the environment:
```bash
git clone [https://github.com/Nupurk30/qna_bot.git](https://github.com/Nupurk30/qna_bot.git) -b branch1
cd qna_bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt