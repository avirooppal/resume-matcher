#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""FastAPI application for serving the Resume Analyzer."""

import os
import uuid
import shutil
import logging
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
from resume_analyzer.advanced_parser import AdvancedResumeParser
from resume_analyzer.advanced_matcher import AdvancedResumeMatcher

# Ensure the analyzer module is importable
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

logger = logging.getLogger(__name__)

# Import analyzer modules
from resume_analyzer.text_extractor import extract_text
from resume_analyzer.parser import parse_resume_text
from resume_analyzer.matcher import match_resume_to_jd

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Resume Analyzer API",
    description="API for analyzing resumes and matching them against job descriptions",
    version="2.0.0"
)

# Initialize components
parser = AdvancedResumeParser()
matcher = AdvancedResumeMatcher()

class AnalysisRequest(BaseModel):
    resume_text: str
    job_description: str

# --- Temporary File Handling ---
TEMP_DIR = os.path.join(project_root, "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

def save_upload_file(upload_file: UploadFile) -> str:
    """Saves an uploaded file temporarily and returns its path."""
    try:
        ext = os.path.splitext(upload_file.filename)[1]
        if not ext:
            ext = ".tmp"
        temp_filename = f"{uuid.uuid4()}{ext}"
        temp_filepath = os.path.join(TEMP_DIR, temp_filename)
        
        logger.info(f"Saving uploaded file {upload_file.filename} to {temp_filepath}")
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
            
        return temp_filepath
    except Exception as e:
        logger.error(f"Error saving upload file {upload_file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not save file: {upload_file.filename}")
    finally:
        if upload_file and hasattr(upload_file, 'file') and not upload_file.file.closed:
            upload_file.file.close()

def cleanup_temp_file(filepath: str):
    """Removes a temporary file."""
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Cleaned up temp file: {filepath}")
    except Exception as e:
        logger.warning(f"Error cleaning up temp file {filepath}: {e}", exc_info=True)

@app.post("/analyze/")
async def analyze_resume(request: AnalysisRequest):
    """Analyze a resume against a job description."""
    try:
        # Parse resume
        logger.info("Parsing resume...")
        resume_data = parser.parse(request.resume_text)
        
        # Match against job description
        logger.info("Matching against job description...")
        match_results = matcher.match_resume_to_jd(
            request.resume_text,
            request.job_description
        )
        
        # Combine results
        response = {
            "resume_data": resume_data,
            "match_results": match_results
        }
        
        logger.info("Analysis complete")
        return response
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    logger.info("Starting Resume Analyzer API server...")
    logger.info(f"API Documentation available at http://127.0.0.1:8000/docs")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False, log_config=None)

