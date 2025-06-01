# Resume Analyzer API (v1.1.0)

This project provides a modular API for parsing resumes and matching them against job descriptions using NLP techniques.

## Features (Enhanced)

*   Parses resumes from PDF, DOCX, and TXT formats.
*   Extracts key information into a structured JSON format (based on JSON Resume schema), with **enhanced parsing for work experience dates, education details (GPA, courses), projects, and certificates**.
*   Matches resumes against job descriptions using:
    *   Semantic similarity (Sentence Transformers) for overall content comparison.
    *   Granular field matching for:
        *   **Skills (including exact and semantic matching)**
        *   **Years of Experience (calculated from non-overlapping work periods)**
        *   **Education Level**
*   Provides a weighted final match score.
*   **Returns detailed matching results**, including matched/missing/semantically-matched skills, suitable for frontend display.
*   Exposes functionality via a FastAPI backend API.
*   Modular codebase for easier maintenance and extension.
*   **Includes robust logging and error handling**.

## Project Structure

```
/
|-- resume_analyzer/         # Core logic module
|   |-- __init__.py
|   |-- config.py            # Configuration settings (models, keywords)
|   |-- models.py            # NLP model loading functions
|   |-- text_extractor.py    # File text extraction logic
|   |-- utils.py             # Utility functions (regex, section splitting)
|   |-- parser.py            # Resume parsing and information extraction (Enhanced)
|   |-- matcher.py           # Resume-JD matching logic (Enhanced)
|-- api.py                   # FastAPI application server (Enhanced with logging/error handling)
|-- main_analyzer.py         # Command-line interface (Optional - may need updates)
|-- requirements.txt         # Python dependencies
|-- temp_uploads/            # Temporary directory for file uploads (created by api.py)
|-- README.md                # This file
```

## Setup Instructions

1.  **Clone/Download:** Obtain the project files.
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download spaCy Model:** The system uses `en_core_web_trf`. Download it:
    ```bash
    python -m spacy download en_core_web_trf
    ```
    *Note: The first time models are loaded (spaCy or Sentence Transformer), they might be downloaded automatically, which can take some time.*

## Running the API Server

1.  **Start the Server:** Use uvicorn to run the FastAPI application:
    ```bash
    uvicorn api:app --host 0.0.0.0 --port 8000
    ```
    *   `--host 0.0.0.0` makes the server accessible on your network.
    *   `--port 8000` specifies the port (change if needed).
    *   Add `--reload` during development for automatic code reloading (may not work in all environments).

2.  **Access API Docs:** Once running, open your browser and go to `http://<your-server-ip>:8000/docs` (or `http://127.0.0.1:8000/docs` if running locally) to access the interactive Swagger UI documentation and test the `/analyze/` endpoint.

## API Response Structure

The `/analyze/` endpoint returns a JSON object like this:

```json
{
  "message": "Analysis successful",
  "parsed_resume": { /* ... Standard JSON Resume structure ... */ },
  "match_results": {
    "match_percentage": 91.5,
    "details": {
      "semantic_score": 0.85,
      "skill_score": 0.9,
      "matched_skills": ["React", "TypeScript", "JavaScript", ...],
      "missing_skills": ["GraphQL"],
      "semantically_matched_skills": [("RESTful APIs", "REST API", 0.88)],
      "experience_score": 1.0,
      "education_score": 1.0,
      "calculated_resume_experience_years": 8.5,
      "weights": { /* ... weights used ... */ },
      "parsed_jd": { /* ... parsed JD details ... */ }
    }
  }
}
```

# Resume Analyzer API -  Guide

## Setup Instructions

1. **Install Python**
   - Download and install Python 3.8 or higher from [python.org](https://python.org)
   - During installation, make sure to check "Add Python to PATH"

2. **Set Up the Environment**
   ```bash
   # Create a virtual environment
   python -m venv venv

   # Activate the virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt

   # Download required spaCy model
   python -m spacy download en_core_web_trf
   ```

3. **Start the API Server**
   ```bash
   python api.py
   ```
   The server will start at http://localhost:8000

## Testing the API

1. **Access the API Documentation**
   - Open your web browser and go to http://localhost:8000/docs
   - This will show you the interactive API documentation

2. **Test the Resume Analysis**
   - Use the `/analyze/` endpoint
   - Upload a resume file (PDF, DOCX, or TXT)
   - Upload a job description file (PDF, DOCX, or TXT)
   - Click "Execute" to run the analysis

3. **Expected Response**
   The API will return a JSON response containing:
   - Parsed resume data
   - Match score against the job description
   - Detailed matching information


