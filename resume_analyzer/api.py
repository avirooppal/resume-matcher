"""Resume Analyzer API using FastAPI."""

import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Dict, Optional, Union, List
import uvicorn
import PyPDF2
import docx
import io
from .advanced_parser import AdvancedResumeParser
from .advanced_matcher import AdvancedResumeMatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Resume Analyzer API",
    description="Advanced resume analysis and job matching API",
    version="2.0.0"
)

# Singleton instances
_parser = None
_matcher = None

def get_parser():
    global _parser
    if _parser is None:
        logger.info("Initializing AdvancedResumeParser...")
        _parser = AdvancedResumeParser()
    return _parser

def get_matcher():
    global _matcher
    if _matcher is None:
        logger.info("Initializing AdvancedResumeMatcher...")
        _matcher = AdvancedResumeMatcher()
    return _matcher

class AnalysisRequest(BaseModel):
    resume_text: str
    job_description: str

async def extract_text_from_file(file: UploadFile) -> str:
    """Extract text from uploaded file (PDF, DOCX, or TXT)."""
    content = await file.read()
    
    if file.filename.endswith('.pdf'):
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error reading PDF file: {str(e)}"
            )
            
    elif file.filename.endswith('.docx'):
        try:
            docx_file = io.BytesIO(content)
            doc = docx.Document(docx_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error reading DOCX file: {str(e)}"
            )
            
    elif file.filename.endswith('.txt'):
        try:
            return content.decode('utf-8')
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error reading TXT file: {str(e)}"
            )
            
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload PDF, DOCX, or TXT files."
        )

def format_resume_data(parsed_data: Dict) -> Dict:
    """Format parsed resume data into the desired structure."""
    return {
        "basics": {
            "name": parsed_data.get("name", ""),
            "label": parsed_data.get("title", ""),
            "picture": "",
            "email": parsed_data.get("email", ""),
            "phone": parsed_data.get("phone", ""),
            "summary": parsed_data.get("summary", ""),
            "location": {
                "address": "",
                "postalCode": "",
                "city": parsed_data.get("location", {}).get("city", ""),
                "countryCode": parsed_data.get("location", {}).get("country", ""),
                "region": parsed_data.get("location", {}).get("state", "")
            },
            "profiles": [],
            "url": "",
            "image": ""
        },
        "work": [
            {
                "position": exp.get("position", ""),
                "name": exp.get("name", ""),
                "startDate": exp.get("startDate", ""),
                "endDate": exp.get("endDate", ""),
                "summary": "",
                "highlights": exp.get("highlights", []),
                "location": exp.get("location", ""),
                "url": "",
                "description": ""
            }
            for exp in parsed_data.get("experience", [])
        ],
        "education": [
            {
                "institution": edu.get("institution", ""),
                "area": edu.get("field", ""),
                "studyType": edu.get("degree", ""),
                "startDate": edu.get("startDate", ""),
                "endDate": edu.get("endDate", ""),
                "gpa": "",
                "score": "",
                "courses": [],
                "location": edu.get("location", "")
            }
            for edu in parsed_data.get("education", [])
        ],
        "publications": [],
        "skills": [
            {
                "name": skill.get("name", ""),
                "level": skill.get("level", ""),
                "keywords": [],
                "category": skill.get("category", "")
            }
            for skill in parsed_data.get("skills", [])
        ],
        "languages": [],
        "projects": [],
        "certificates": []
    }

def format_match_results(match_data: Dict) -> Dict:
    """Format match results into the desired structure."""
    return {
        "match_percentage": match_data.get("overall_score", 0) * 100,
        "details": {
            "semantic_score": match_data.get("semantic_similarity", 0),
            "skill_score": match_data.get("skill_match", {}).get("score", 0),
            "matched_skills": [
                skill.get("skill", "")
                for skill in match_data.get("skill_match", {}).get("matched", [])
            ],
            "missing_skills": match_data.get("skill_match", {}).get("missing", []),
            "semantically_matched_skills": [
                {
                    "skill": match.get("skill", ""),
                    "matched_with": match.get("matched_with", ""),
                    "confidence": match.get("confidence", 0)
                }
                for match in match_data.get("skill_match", {}).get("semantic", [])
            ],
            "experience_score": match_data.get("experience_match", {}).get("score", 0),
            "education_score": 0,  # To be implemented
            "calculated_resume_experience_years": sum(
                exp.get("years", 0)
                for exp in match_data.get("experience_match", {}).get("matched", [])
                if exp.get("type") == "years"
            ),
            "weights": {
                "semantic": 0.3,
                "skills": 0.4,
                "experience": 0.3,
                "education": 0
            },
            "parsed_jd": {
                "requirements_text": "",
                "responsibilities_text": "",
                "required_skills": [
                    skill.get("name", "")
                    for skill in match_data.get("required_skills", [])
                ],
                "required_experience_years": None,
                "required_education": "",
                "match_text": match_data.get("match_text", "")
            }
        }
    }

@app.post("/analyze/")
async def analyze_resume(
    resume: Optional[UploadFile] = File(None),
    job_description: Optional[UploadFile] = File(None),
    resume_text: Optional[str] = Form(None),
    job_description_text: Optional[str] = Form(None)
) -> Dict:
    """Analyze resume against job description.
    
    Accepts either:
    1. File uploads (PDF, DOCX, TXT) for both resume and job description
    2. Direct text input for both resume and job description
    """
    try:
        # Get singleton instances
        parser = get_parser()
        matcher = get_matcher()
        
        # Process resume
        if resume:
            resume_text = await extract_text_from_file(resume)
        elif not resume_text:
            raise HTTPException(
                status_code=400,
                detail="Either resume file or resume text must be provided"
            )
            
        # Process job description
        if job_description:
            job_description_text = await extract_text_from_file(job_description)
        elif not job_description_text:
            raise HTTPException(
                status_code=400,
                detail="Either job description file or text must be provided"
            )
        
        # Parse and match
        logger.info("Parsing resume and job description...")
        parsed_resume = parser.parse(resume_text)
        match_result = matcher.match_resume_to_jd(
            resume_text,
            job_description_text
        )
        
        return {
            "message": "Resume analysis completed successfully",
            "parsed_resume": format_resume_data(parsed_resume),
            "match_results": format_match_results(match_result)
        }
    except Exception as e:
        logger.error(f"Error analyzing resume: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing resume: {str(e)}"
        )

@app.get("/health")
async def health_check() -> Dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.0.0"
    }

if __name__ == "__main__":
    logger.info("Starting Resume Analyzer API server...")
    uvicorn.run(
        "resume_analyzer.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # Disable reload to prevent multiple model loads
    ) 