"""Advanced resume parsing using modern NLP techniques."""

import spacy
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime

logger = logging.getLogger(__name__)

# Shared model instances
_nlp = None
_skill_model = None
_text_model = None

def get_nlp():
    global _nlp
    if _nlp is None:
        logger.info("Loading spaCy model...")
        _nlp = spacy.load("en_core_web_trf")
    return _nlp

def get_skill_model():
    global _skill_model
    if _skill_model is None:
        logger.info("Loading skill model...")
        _skill_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _skill_model

def get_text_model():
    global _text_model
    if _text_model is None:
        logger.info("Loading text model...")
        _text_model = SentenceTransformer('all-mpnet-base-v2')
    return _text_model

class AdvancedResumeParser:
    def __init__(self):
        # Use shared model instances
        self.nlp = get_nlp()
        self.skill_model = get_skill_model()
        self.text_model = get_text_model()
        
        # Common skill patterns
        self.skill_patterns = {
            "programming": r"(?i)(python|java|javascript|typescript|c\+\+|c#|ruby|php|swift|kotlin|go|rust)",
            "frameworks": r"(?i)(react|angular|vue|django|flask|spring|express|node\.js|asp\.net|laravel|rails)",
            "databases": r"(?i)(sql|mysql|postgresql|mongodb|redis|oracle|sqlite|cassandra)",
            "tools": r"(?i)(git|docker|kubernetes|jenkins|jira|confluence|vscode|eclipse|intellij)",
            "cloud": r"(?i)(aws|azure|gcp|heroku|digitalocean|cloudflare)",
            "devops": r"(?i)(ci/cd|jenkins|github actions|gitlab ci|travis ci|ansible|terraform)"
        }
        
        # Experience patterns
        self.exp_patterns = {
            "date_range": r"(?i)(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\s*[-–—to]+\s*(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Present|Current)[a-z]*\.?\s+\d{4}\b|Present|Current)",
            "position": r"(?i)(Senior|Lead|Principal)?\s*(Software|Frontend|Backend|Full Stack|DevOps|Data|ML|AI|Cloud|Security)?\s*(Engineer|Developer|Architect|Consultant|Analyst|Scientist)",
            "company": r"(?i)(Inc\.|LLC|Ltd\.|Corp\.|Corporation|Company|Co\.)"
        }

    def extract_skills(self, text: str) -> List[Dict]:
        """Extract skills using both pattern matching and semantic similarity."""
        skills = []
        
        # Pattern-based extraction
        for category, pattern in self.skill_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                skill = match.group(0)
                skills.append({
                    "name": skill,
                    "category": category,
                    "confidence": 0.9  # High confidence for pattern matches
                })
        
        # Extract skills from highlights and descriptions
        doc = self.nlp(text)
        for sent in doc.sents:
            # Look for skill mentions in the text
            for token in sent:
                if token.pos_ in ["NOUN", "PROPN"]:
                    # Check if it's likely a skill
                    if self._is_likely_skill(token.text):
                        skills.append({
                            "name": token.text,
                            "category": self._categorize_skill(token.text),
                            "confidence": 0.7
                        })
        
        # Extract skills from comma-separated lists
        skill_lists = re.findall(r'(?i)(?:skills|technologies|tools|frameworks|languages)[:]\s*([^.\n]+)', text)
        for skill_list in skill_lists:
            for skill in skill_list.split(','):
                skill = skill.strip()
                if skill and len(skill) > 1:
                    skills.append({
                        "name": skill,
                        "category": self._categorize_skill(skill),
                        "confidence": 0.8
                    })
        
        # Remove duplicates and sort by confidence
        unique_skills = self._deduplicate_skills(skills)
        return sorted(unique_skills, key=lambda x: x["confidence"], reverse=True)

    def extract_experience(self, text: str) -> List[Dict]:
        """Extract work experience using advanced NLP techniques."""
        experiences = []
        
        # Split into potential experience blocks
        blocks = self._split_into_blocks(text)
        
        for block in blocks:
            # Extract dates
            dates = self._extract_dates(block)
            
            # Extract position and company
            position, company = self._extract_position_company(block)
            
            # Extract highlights
            highlights = self._extract_highlights(block)
            
            # Extract skills from highlights
            skills = []
            for highlight in highlights:
                skills.extend(self.extract_skills(highlight))
            
            # Calculate confidence scores
            confidence = self._calculate_confidence(block, position, company, dates)
            
            if confidence > 0.5:  # Only include high-confidence entries
                experiences.append({
                    "position": position,
                    "name": company,
                    "startDate": dates.get("start"),
                    "endDate": dates.get("end"),
                    "highlights": highlights,
                    "skills": skills,
                    "confidence": confidence
                })
        
        return sorted(experiences, key=lambda x: x["confidence"], reverse=True)

    def _is_likely_skill(self, text: str) -> bool:
        """Determine if a text is likely a skill using semantic similarity."""
        # Compare with known skills
        known_skills = ["python", "java", "javascript", "react", "angular", "vue"]
        text_embedding = self.skill_model.encode([text])[0]
        known_embeddings = self.skill_model.encode(known_skills)
        
        similarities = np.dot(known_embeddings, text_embedding) / (
            np.linalg.norm(known_embeddings, axis=1) * np.linalg.norm(text_embedding)
        )
        
        return np.max(similarities) > 0.7

    def _categorize_skill(self, skill: str) -> str:
        """Categorize a skill using semantic similarity."""
        categories = list(self.skill_patterns.keys())
        skill_embedding = self.skill_model.encode([skill])[0]
        category_embeddings = self.skill_model.encode(categories)
        
        similarities = np.dot(category_embeddings, skill_embedding) / (
            np.linalg.norm(category_embeddings, axis=1) * np.linalg.norm(skill_embedding)
        )
        
        return categories[np.argmax(similarities)]

    def _split_into_blocks(self, text: str) -> List[str]:
        """Split text into potential experience blocks."""
        # Use blank lines and date patterns as separators
        blocks = re.split(r"\n\s*\n", text)
        return [block.strip() for block in blocks if block.strip()]

    def _extract_dates(self, text: str) -> Dict:
        """Extract date ranges from text."""
        match = re.search(self.exp_patterns["date_range"], text)
        if match:
            return {
                "start": match.group(1),
                "end": match.group(2)
            }
        return {}

    def _extract_position_company(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract position and company using NLP and patterns."""
        doc = self.nlp(text)
        
        # Try pattern matching first
        position_match = re.search(self.exp_patterns["position"], text)
        company_match = re.search(self.exp_patterns["company"], text)
        
        position = position_match.group(0) if position_match else None
        company = company_match.group(0) if company_match else None
        
        # If pattern matching fails, use NER
        if not position or not company:
            for ent in doc.ents:
                if ent.label_ == "WORK_OF_ART" and not position:
                    position = ent.text
                elif ent.label_ == "ORG" and not company:
                    company = ent.text
        
        return position, company

    def _extract_highlights(self, text: str) -> List[str]:
        """Extract highlights using bullet points and semantic analysis."""
        highlights = []
        
        # Extract bullet points
        bullet_points = re.findall(r"^[\-\*•]\s*(.+)$", text, re.MULTILINE)
        highlights.extend(bullet_points)
        
        # Extract numbered lists
        numbered_points = re.findall(r"^\d+\.\s*(.+)$", text, re.MULTILINE)
        highlights.extend(numbered_points)
        
        # Extract skills from comma-separated lists
        skill_lists = re.findall(r'(?i)(?:skills|technologies|tools|frameworks|languages)[:]\s*([^.\n]+)', text)
        for skill_list in skill_lists:
            highlights.extend(skill_list.split(','))
        
        return [h.strip() for h in highlights if h.strip()]

    def _calculate_confidence(self, block: str, position: Optional[str], 
                            company: Optional[str], dates: Dict) -> float:
        """Calculate confidence score for an experience entry."""
        confidence = 0.0
        
        # Position and company presence
        if position:
            confidence += 0.3
        if company:
            confidence += 0.3
            
        # Date presence
        if dates.get("start"):
            confidence += 0.2
        if dates.get("end"):
            confidence += 0.1
            
        # Block length and structure
        if len(block.split()) > 20:  # Sufficient content
            confidence += 0.1
            
        return confidence

    def _deduplicate_skills(self, skills: List[Dict]) -> List[Dict]:
        """Remove duplicate skills while preserving the highest confidence version."""
        unique_skills = {}
        for skill in skills:
            name = skill["name"].lower()
            if name not in unique_skills or skill["confidence"] > unique_skills[name]["confidence"]:
                unique_skills[name] = skill
        return list(unique_skills.values())

    def parse(self, text: str) -> Dict:
        """Main parsing function that combines all extractions."""
        # Extract basic information
        doc = self.nlp(text)
        
        # Extract name (first PERSON entity)
        name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "")
        
        # Extract email with improved pattern
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        email = next((match.group(0) for match in re.finditer(email_pattern, text)), "")
        
        # Extract phone with improved pattern
        phone_pattern = r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phone = next((match.group(0) for match in re.finditer(phone_pattern, text)), "")
        
        # Extract location
        location = next((ent.text for ent in doc.ents if ent.label_ == "GPE"), "")
        
        # Extract summary with multiple patterns
        summary_patterns = [
            r'(?i)(?:summary|profile|about|objective)[:]\s*([^#\n]+?)(?=\n\s*\n|\Z)',
            r'(?i)(?:summary|profile|about|objective)[:]\s*([^#\n]+?)(?=\n\s*[A-Z][a-z]+[:]|\Z)',
            r'(?i)(?:summary|profile|about|objective)[:]\s*([^#\n]+?)(?=\n\s*[A-Z][a-z]+\s*$|\Z)'
        ]
        
        summary = ""
        for pattern in summary_patterns:
            match = re.search(pattern, text)
            if match:
                summary = match.group(1).strip()
                break
        
        # If no summary found, try to extract from the first paragraph
        if not summary:
            paragraphs = text.split('\n\n')
            if paragraphs:
                first_para = paragraphs[0].strip()
                if len(first_para.split()) > 10 and not any(keyword in first_para.lower() for keyword in ['experience', 'education', 'skills']):
                    summary = first_para
        
        # Extract skills and experience
        skills = self.extract_skills(text)
        experience = self.extract_experience(text)
        
        # Extract skills from experience highlights
        for exp in experience:
            if "highlights" in exp:
                for highlight in exp["highlights"]:
                    skills.extend(self.extract_skills(highlight))
        
        # Remove duplicate skills
        unique_skills = {}
        for skill in skills:
            name = skill["name"].lower()
            if name not in unique_skills or skill["confidence"] > unique_skills[name]["confidence"]:
                unique_skills[name] = skill
        
        return {
            "name": name,
            "email": email,
            "phone": phone,
            "summary": summary,
            "location": {"city": location},
            "skills": list(unique_skills.values()),
            "experience": experience,
            "education": []  # To be implemented
        } 