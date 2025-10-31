# Simple RAG Project

This is a simple, end-to-end RAG (Retrieval-Augmented Generation) application. It uses a FastAPI backend, local AI models, and a local vector store to answer questions based on your private documents.

## ✨ Features

* **Ingests Local Files:** Reads `.pdf` and `.txt` files from a `/data` folder.
* **100% Local:** Uses free, local Hugging Face models (`sentence-transformers` and `flan-t5-base`) for embeddings and generation. No API keys needed.
* **Vector Search:** Uses `ChromaDB` to store and search for document chunks.
* **Web API:** Provides a robust FastAPI server with an `/ask` endpoint.
* **Easy Setup:** Fully containerized with Docker. Your manager can run it with one command.

---

## 🚀 How to Run (Recommended: Docker)

This is the easiest and most reliable way to run the project.

### Prerequisites
* [Git](https://git-scm.com/downloads)
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) (must be running)

### 1. Clone the Repository

```bash
git clone [https://github.com/Nupurk30/qna_bot.git](https://github.com/Nupurk30/qna_bot.git)
cd qna_bot