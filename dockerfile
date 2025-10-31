# 1. Start with a Python 3.10 base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file first (for caching)
COPY requirements.txt .

# 4. Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your project code
COPY . .

# 6. Run the ingestion script to build the 'db/' folder
RUN python ingest.py

# 7. Expose the port your API runs on
EXPOSE 8000

# 8. The command to run your app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]