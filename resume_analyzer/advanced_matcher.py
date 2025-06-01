"""Advanced resume matching using embeddings and semantic similarity."""

import logging
from typing import Dict, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from datetime import datetime
import re
from .advanced_parser import AdvancedResumeParser, get_skill_model, get_text_model

logger = logging.getLogger(__name__)

# Shared parser instance
_parser = None

def get_parser():
    global _parser
    if _parser is None:
        logger.info("Initializing AdvancedResumeParser...")
        _parser = AdvancedResumeParser()
    return _parser

class AdvancedResumeMatcher:
    def __init__(self):
        # Use shared instances
        self.parser = get_parser()
        self.skill_model = get_skill_model()
        self.text_model = get_text_model()
        
        # Move models to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def match_resume_to_jd(self, resume_text: str, jd_text: str) -> Dict:
        """Match resume against job description using advanced techniques."""
        # Parse both resume and JD
        resume_data = self.parser.parse(resume_text)
        jd_data = self._parse_job_description(jd_text)
        
        # Calculate various match scores
        skill_score, skill_matches = self._match_skills(
            resume_data["skills"],
            jd_data["required_skills"]
        )
        
        experience_score, exp_matches = self._match_experience(
            resume_data["experience"],
            jd_data["required_experience"]
        )
        
        semantic_score = self._calculate_semantic_similarity(
            resume_text,
            jd_text
        )
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            skill_score,
            experience_score,
            semantic_score
        )
        
        # Log detailed results
        logger.info(f"Overall score: {overall_score:.2f}")
        logger.info(f"Skill score: {skill_score:.2f}")
        logger.info(f"Experience score: {experience_score:.2f}")
        logger.info(f"Semantic score: {semantic_score:.2f}")
        
        return {
            "overall_score": overall_score,
            "skill_match": {
                "score": skill_score,
                "matched_skills": skill_matches["matched"],
                "missing_skills": skill_matches["missing"],
                "semantic_matches": skill_matches["semantic"]
            },
            "experience_match": {
                "score": experience_score,
                "matched_experience": exp_matches["matched"],
                "missing_experience": exp_matches["missing"]
            },
            "semantic_similarity": semantic_score,
            "required_skills": jd_data["required_skills"],
            "match_text": jd_text
        }

    def _parse_job_description(self, jd_text: str) -> Dict:
        """Parse job description to extract required skills and experience."""
        # Extract requirements and responsibilities sections
        requirements_text = self._extract_section(jd_text, "requirements")
        responsibilities_text = self._extract_section(jd_text, "responsibilities")
        
        # Extract skills from both sections
        required_skills = []
        
        # Extract from requirements section
        if requirements_text:
            required_skills.extend(self._extract_skills_from_section(requirements_text))
        
        # Extract from responsibilities section
        if responsibilities_text:
            required_skills.extend(self._extract_skills_from_section(responsibilities_text))
        
        # Extract from the entire text if sections are empty
        if not required_skills:
            required_skills.extend(self._extract_required_skills(jd_text))
        
        # Extract required experience
        required_experience = self._extract_required_experience(jd_text)
        
        return {
            "required_skills": required_skills,
            "required_experience": required_experience,
            "requirements_text": requirements_text,
            "responsibilities_text": responsibilities_text,
            "match_text": jd_text
        }

    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a specific section from the job description."""
        patterns = [
            f"(?i){section_name}[:]?\s*([^#\n]+)",
            f"(?i){section_name}[:]?\s*([^#\n]+?)(?=\n\s*\n|\Z)",
            f"(?i){section_name}[:]?\s*([^#\n]+?)(?=\n\s*[A-Z][a-z]+[:]|\Z)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                section_text = match.group(1).strip()
                # Clean up bullet points and numbering
                section_text = re.sub(r'^[\-\*•]\s*', '', section_text, flags=re.MULTILINE)
                section_text = re.sub(r'^\d+\.\s*', '', section_text, flags=re.MULTILINE)
                return section_text
        return ""

    def _extract_skills_from_section(self, text: str) -> List[Dict]:
        """Extract skills from a specific section of text."""
        skills = []
        
        # Extract from bullet points
        bullet_points = re.findall(r'^[\-\*•]\s*([^.\n]+)', text, re.MULTILINE)
        for point in bullet_points:
            # Extract skills from each bullet point
            for category, pattern in self.parser.skill_patterns.items():
                matches = re.finditer(pattern, point)
                for match in matches:
                    skill = match.group(0)
                    skills.append({
                        "name": skill,
                        "category": category,
                        "confidence": 0.9
                    })
        
        # Extract from comma-separated lists
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
        
        # Remove duplicates
        unique_skills = {}
        for skill in skills:
            name = skill["name"].lower()
            if name not in unique_skills or skill["confidence"] > unique_skills[name]["confidence"]:
                unique_skills[name] = skill
        
        return list(unique_skills.values())

    def _extract_required_skills(self, jd_text: str) -> List[Dict]:
        """Extract required skills from job description."""
        required_skills = []
        
        # Common skill patterns
        skill_patterns = {
            "programming": r"(?i)(python|java|javascript|typescript|c\+\+|c#|ruby|php|swift|kotlin|go|rust)",
            "frameworks": r"(?i)(react|angular|vue|django|flask|spring|express|node\.js|asp\.net|laravel|rails)",
            "databases": r"(?i)(sql|mysql|postgresql|mongodb|redis|oracle|sqlite|cassandra)",
            "tools": r"(?i)(git|docker|kubernetes|jenkins|jira|confluence|vscode|eclipse|intellij)",
            "cloud": r"(?i)(aws|azure|gcp|heroku|digitalocean|cloudflare)",
            "devops": r"(?i)(ci/cd|jenkins|github actions|gitlab ci|travis ci|ansible|terraform)"
        }
        
        # Extract skills using patterns
        for category, pattern in skill_patterns.items():
            matches = re.finditer(pattern, jd_text)
            for match in matches:
                skill = match.group(0)
                required_skills.append({
                    "name": skill,
                    "category": category,
                    "confidence": 0.9
                })
        
        # Extract skills from requirements section
        requirements_text = self._extract_section(jd_text, "requirements")
        if requirements_text:
            doc = self.parser.nlp(requirements_text)
            for sent in doc.sents:
                for token in sent:
                    if token.pos_ in ["NOUN", "PROPN"]:
                        if self._is_likely_skill(token.text):
                            required_skills.append({
                                "name": token.text,
                                "category": self._categorize_skill(token.text),
                                "confidence": 0.7
                            })
        
        # Extract skills from comma-separated lists
        skill_lists = re.findall(r'(?i)(?:skills|technologies|tools|frameworks|languages)[:]\s*([^.\n]+)', jd_text)
        for skill_list in skill_lists:
            for skill in skill_list.split(','):
                skill = skill.strip()
                if skill and len(skill) > 1:
                    required_skills.append({
                        "name": skill,
                        "category": self._categorize_skill(skill),
                        "confidence": 0.8
                    })
        
        # Remove duplicates
        unique_skills = {}
        for skill in required_skills:
            name = skill["name"].lower()
            if name not in unique_skills or skill["confidence"] > unique_skills[name]["confidence"]:
                unique_skills[name] = skill
        
        return list(unique_skills.values())

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
        categories = ["programming", "frameworks", "databases", "tools", "cloud", "devops"]
        skill_embedding = self.skill_model.encode([skill])[0]
        category_embeddings = self.skill_model.encode(categories)
        
        similarities = np.dot(category_embeddings, skill_embedding) / (
            np.linalg.norm(category_embeddings, axis=1) * np.linalg.norm(skill_embedding)
        )
        
        return categories[np.argmax(similarities)]

    def _match_skills(self, resume_skills: List[Dict], jd_skills: List[Dict]) -> Tuple[float, Dict]:
        """Match skills using both exact and semantic matching."""
        matched_skills = []
        missing_skills = []
        semantic_matches = []
        
        if not jd_skills:
            return 0.0, {"matched": [], "missing": [], "semantic": []}
        
        # Convert skills to embeddings
        resume_skill_embeddings = self.skill_model.encode([s["name"] for s in resume_skills])
        jd_skill_embeddings = self.skill_model.encode([s["name"] for s in jd_skills])
        
        # Calculate similarity matrix
        similarity_matrix = util.pytorch_cos_sim(
            torch.tensor(resume_skill_embeddings),
            torch.tensor(jd_skill_embeddings)
        )
        
        # Find matches
        for i, jd_skill in enumerate(jd_skills):
            matched = False
            
            # Check for exact matches first
            for resume_skill in resume_skills:
                if resume_skill["name"].lower() == jd_skill["name"].lower():
                    matched_skills.append({
                        "skill": jd_skill["name"],
                        "confidence": 1.0,
                        "type": "exact"
                    })
                    matched = True
                    break
            
            if not matched:
                # Check for semantic matches
                similarities = similarity_matrix[:, i]
                max_sim_idx = torch.argmax(similarities).item()
                max_sim = similarities[max_sim_idx].item()
                
                if max_sim > 0.7:  # High similarity threshold
                    semantic_matches.append({
                        "skill": jd_skill["name"],
                        "matched_with": resume_skills[max_sim_idx]["name"],
                        "confidence": max_sim,
                        "type": "semantic"
                    })
                    matched = True
                else:
                    missing_skills.append({
                        "skill": jd_skill["name"],
                        "category": jd_skill.get("category", "unknown"),
                        "confidence": 0.0
                    })
        
        # Calculate score
        total_skills = len(jd_skills)
        if total_skills == 0:
            return 0.0, {"matched": [], "missing": [], "semantic": []}
            
        score = (len(matched_skills) + 0.5 * len(semantic_matches)) / total_skills
        
        # Log matching results
        logger.info(f"Matched skills: {len(matched_skills)}")
        logger.info(f"Semantic matches: {len(semantic_matches)}")
        logger.info(f"Missing skills: {len(missing_skills)}")
        
        return score, {
            "matched": matched_skills,
            "missing": missing_skills,
            "semantic": semantic_matches
        }

    def _extract_required_experience(self, jd_text: str) -> List[Dict]:
        """Extract required experience from job description."""
        required_exp = []
        
        # Look for experience requirements
        exp_patterns = [
            r"(?i)(\d+)\+?\s*years?\s+of\s+experience",
            r"(?i)experience\s+with\s+([^.,]+)",
            r"(?i)proven\s+experience\s+in\s+([^.,]+)"
        ]
        
        for pattern in exp_patterns:
            matches = re.finditer(pattern, jd_text)
            for match in matches:
                if "years" in pattern:
                    required_exp.append({
                        "type": "years",
                        "value": int(match.group(1)),
                        "text": match.group(0)
                    })
                else:
                    required_exp.append({
                        "type": "skill",
                        "value": match.group(1).strip(),
                        "text": match.group(0)
                    })
        
        return required_exp

    def _match_experience(self, resume_exp: List[Dict], required_exp: List[Dict]) -> Tuple[float, Dict]:
        """Match experience using semantic similarity."""
        matched_exp = []
        missing_exp = []
        
        for req in required_exp:
            matched = False
            
            if req["type"] == "years":
                # Check if any experience block meets the year requirement
                for exp in resume_exp:
                    if self._calculate_experience_years(exp) >= req["value"]:
                        matched_exp.append({
                            "requirement": req["text"],
                            "matched_with": exp["position"],
                            "type": "years"
                        })
                        matched = True
                        break
            else:
                # Semantic match for skill-based requirements
                req_embedding = self.text_model.encode([req["value"]])[0]
                exp_embeddings = self.text_model.encode([
                    f"{exp['position']} {exp['name']} {' '.join(exp['highlights'])}"
                    for exp in resume_exp
                ])
                
                similarities = util.pytorch_cos_sim(
                    torch.tensor([req_embedding]),
                    torch.tensor(exp_embeddings)
                )[0]
                
                max_sim_idx = torch.argmax(similarities).item()
                max_sim = similarities[max_sim_idx].item()
                
                if max_sim > 0.6:  # Lower threshold for experience matching
                    matched_exp.append({
                        "requirement": req["text"],
                        "matched_with": resume_exp[max_sim_idx]["position"],
                        "confidence": max_sim,
                        "type": "semantic"
                    })
                    matched = True
            
            if not matched:
                missing_exp.append(req["text"])
        
        # Calculate score
        total_req = len(required_exp)
        if total_req == 0:
            return 0.0, {"matched": [], "missing": []}
            
        score = len(matched_exp) / total_req
        
        return score, {
            "matched": matched_exp,
            "missing": missing_exp
        }

    def _calculate_experience_years(self, experience: Dict) -> float:
        """Calculate years of experience from a single experience entry."""
        if not experience.get("startDate") or not experience.get("endDate"):
            return 0.0
            
        try:
            start = datetime.strptime(experience["startDate"], "%Y-%m")
            end = datetime.now() if experience["endDate"] == "Present" else \
                  datetime.strptime(experience["endDate"], "%Y-%m")
            
            years = (end - start).days / 365.25
            return max(0.0, years)
        except (ValueError, TypeError):
            return 0.0

    def _calculate_semantic_similarity(self, resume_text: str, jd_text: str) -> float:
        """Calculate semantic similarity between resume and job description."""
        # Use the more powerful model for longer texts
        resume_embedding = self.text_model.encode([resume_text])[0]
        jd_embedding = self.text_model.encode([jd_text])[0]
        
        similarity = util.pytorch_cos_sim(
            torch.tensor([resume_embedding]),
            torch.tensor([jd_embedding])
        )[0][0].item()
        
        return similarity

    def _calculate_overall_score(self, skill_score: float, 
                               experience_score: float, 
                               semantic_score: float) -> float:
        """Calculate overall match score using weighted combination."""
        weights = {
            "skill": 0.4,
            "experience": 0.3,
            "semantic": 0.3
        }
        
        return (
            weights["skill"] * skill_score +
            weights["experience"] * experience_score +
            weights["semantic"] * semantic_score
        ) 