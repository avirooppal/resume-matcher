# Use a base image with Python
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_trf

# Create necessary directories
RUN mkdir -p temp_uploads data

# Copy the rest of the application code
COPY . .

# Set permissions
RUN chmod -R 755 temp_uploads data

# Expose the port your app runs on
EXPOSE 8000

# Command to run the application using uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"] 