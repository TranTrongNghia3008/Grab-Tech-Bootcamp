# BE/Dockerfile

# Use a slim Python image (replace 3.10.11 with your exact version if different)
FROM python:3.10.11-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker cache
# This layer is rebuilt only if requirements.txt changes
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your backend code into the container
# This layer is rebuilt if any other files in BE/ change
COPY . .

# Command to run your FastAPI application using Uvicorn
# Cloud Run provides the PORT environment variable, your app MUST listen on it.
# Replace 'main:app' with the actual module:variable for your FastAPI app.
# If your app instance is 'my_app' in 'src/api.py', use 'src.api:my_app'
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]

# EXPOSE is optional but good practice to indicate which port the container listens on
EXPOSE 8000