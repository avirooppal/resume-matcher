# -*- coding: utf-8 -*-
"""Functions for matching resumes against job descriptions.

Enhanced version with granular JD parsing, field-level matching, 
semantic skill matching, refined experience/education matching, 
and frontend-compatible output.
"""

import re
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sentence_transformers import util
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .models import load_sentence_transformer_model, load_spacy_model
from .utils import extract_section_text
from .config import JD_REQUIREMENTS_KEYWORDS, JD_RESPONSIBILITIES_KEYWORDS, SKILLS_KEYWORDS
from .parser import extract_skills_from_section, parse_date_range # Import parse_date_range if needed here?

# --- Setup Logging ---
logger = logging.getLogger(__name__)

# --- Configuration for Semantic Matching ---
SEMANTIC_SIMILARITY_THRESHOLD = 0.75

# --- Job Description Parsing (Enhanced) ---

def parse_job_description(jd_text):
    # ... (Keep existing logic) ...
    if not jd_text:
        logger.warning("JD text is empty, cannot parse.")
        return {"requirements_text": "", "responsibilities_text": "", "required_skills": [], "required_experience_years": None, "required_education": None, "full_text": ""}

    logger.info("Parsing job description...")
    nlp = load_spacy_model()
    if not nlp:
        logger.error("spaCy model not loaded, cannot perform NER on JD.")
        doc = None
    else:
        doc = nlp(jd_text)

    requirements_text = extract_section_text(jd_text, JD_REQUIREMENTS_KEYWORDS)
    responsibilities_text = extract_section_text(jd_text, JD_RESPONSIBILITIES_KEYWORDS)

    jd_skills_text = requirements_text if requirements_text else jd_text
    required_skills = extract_skills_from_section(jd_skills_text)
    logger.info(f"Extracted {len(required_skills)} required skills from JD.")

    required_experience_years = None
    try:
        exp_match = re.search(r"(\d+)\+?\s+years?\s+of\s+experience", jd_text, re.IGNORECASE)
        if exp_match:
            required_experience_years = int(exp_match.group(1))
        else:
            exp_match_min = re.search(r"minimum\s+(\d+)\s+years?", jd_text, re.IGNORECASE)
            if exp_match_min:
                required_experience_years = int(exp_match_min.group(1))
        if required_experience_years is not None:
            logger.info(f"Extracted required experience: {required_experience_years} years.")
    except Exception as e:
        logger.warning(f"Could not parse required experience years: {e}")

    required_education = None
    edu_keywords = {
        "bachelor": ["bachelor", "b.s", "bs", "b.a", "ba"],
        "master": ["master", "m.s", "ms", "m.a", "ma", "mba"],
        "phd": ["ph.d", "phd", "doctorate"]
    }
    jd_lower = jd_text.lower()
    if any(keyword in jd_lower for keyword in edu_keywords["phd"]):
        required_education = "phd"
    elif any(keyword in jd_lower for keyword in edu_keywords["master"]):
        required_education = "master"
    elif any(keyword in jd_lower for keyword in edu_keywords["bachelor"]):
        required_education = "bachelor"
    if required_education:
        logger.info(f"Extracted required education level: {required_education}")

    jd_match_text = f"{requirements_text}\n\n{responsibilities_text}".strip()
    if not jd_match_text:
        jd_match_text = jd_text

    return {
        "requirements_text": requirements_text,
        "responsibilities_text": responsibilities_text,
        "required_skills": required_skills,
        "required_experience_years": required_experience_years,
        "required_education": required_education,
        "match_text": jd_match_text
    }

# --- Granular Matching Functions (Refined) ---

def compare_skills(resume_skills, jd_skills):
    results = {"score": 0.0, "matched_skills": [], "missing_skills": [], "semantically_matched": []}
    sentence_model = load_sentence_transformer_model()

    if not jd_skills:
        logger.info("No required skills found in JD. Skill match score: 1.0")
        results["score"] = 1.0
        return results
        
    # Extract skill names from JD skills
    jd_skill_names_orig = {skill.strip(): skill.strip().lower() for skill in jd_skills if skill and skill.strip()}
    jd_skill_names_lower = set(jd_skill_names_orig.values())
    if not jd_skill_names_lower:
        logger.info("JD skills list empty after cleaning. Skill match score: 1.0")
        results["score"] = 1.0
        return results
        
    original_jd_skills_list = sorted(list(jd_skill_names_orig.keys()))
    results["missing_skills"] = original_jd_skills_list

    if not resume_skills:
        logger.warning("No skills found in resume. Skill match score: 0.0")
        return results

    # Extract skill names from resume skills, handling the new format
    resume_skill_names_orig = {}
    for skill in resume_skills:
        if isinstance(skill, dict) and "name" in skill:
            skill_name = skill["name"].strip()
            if skill_name:
                resume_skill_names_orig[skill_name] = skill_name.lower()
        elif isinstance(skill, str):
            skill_name = skill.strip()
            if skill_name:
                resume_skill_names_orig[skill_name] = skill_name.lower()
                
    resume_skill_names_lower = set(resume_skill_names_orig.values())
    if not resume_skill_names_lower:
        logger.warning("Resume skills list empty after cleaning. Skill match score: 0.0")
        return results

    exact_matched_lower = resume_skill_names_lower.intersection(jd_skill_names_lower)
    matched_skills_exact_orig = sorted([orig for orig, lower in jd_skill_names_orig.items() if lower in exact_matched_lower])
    results["matched_skills"] = matched_skills_exact_orig
    logger.info(f"Found {len(matched_skills_exact_orig)} exact skill matches.")

    remaining_jd_lower = jd_skill_names_lower - exact_matched_lower
    remaining_resume_lower = resume_skill_names_lower - exact_matched_lower
    semantically_matched_pairs = []

    if remaining_jd_lower and remaining_resume_lower and sentence_model:
        logger.info(f"Attempting semantic matching on {len(remaining_jd_lower)} remaining JD skills and {len(remaining_resume_lower)} remaining resume skills.")
        try:
            remaining_jd_list = sorted(list(remaining_jd_lower))
            remaining_resume_list = sorted(list(remaining_resume_lower))
            jd_orig_map = {lower: orig for orig, lower in jd_skill_names_orig.items() if lower in remaining_jd_lower}
            resume_orig_map = {lower: orig for orig, lower in resume_skill_names_orig.items() if lower in remaining_resume_lower}

            jd_embeddings = sentence_model.encode(remaining_jd_list, convert_to_tensor=True)
            resume_embeddings = sentence_model.encode(remaining_resume_list, convert_to_tensor=True)
            cosine_scores = util.cos_sim(jd_embeddings, resume_embeddings).cpu().numpy()

            matched_resume_indices = set()
            for i, jd_skill_lower in enumerate(remaining_jd_list):
                best_match_idx = np.argmax(cosine_scores[i])
                best_score = cosine_scores[i, best_match_idx]

                if best_score >= SEMANTIC_SIMILARITY_THRESHOLD and best_match_idx not in matched_resume_indices:
                    resume_skill_lower = remaining_resume_list[best_match_idx]
                    jd_skill_orig = jd_orig_map.get(jd_skill_lower, jd_skill_lower)
                    resume_skill_orig = resume_orig_map.get(resume_skill_lower, resume_skill_lower)
                    
                    logger.info(f"Semantic Match Found: {jd_skill_orig} (JD) ~= {resume_skill_orig} (Resume) with score {best_score:.3f}")
                    semantically_matched_pairs.append((jd_skill_orig, resume_skill_orig, round(best_score, 3)))
                    matched_resume_indices.add(best_match_idx)
                    
        except Exception as e:
            logger.error(f"Error during semantic skill matching: {e}", exc_info=True)
    elif not sentence_model:
         logger.warning("Sentence model not loaded, skipping semantic skill matching.")

    semantically_matched_jd_orig = {pair[0] for pair in semantically_matched_pairs}
    results["semantically_matched"] = sorted(semantically_matched_pairs)
    
    all_matched_jd_orig = set(matched_skills_exact_orig) | semantically_matched_jd_orig
    results["missing_skills"] = sorted([orig for orig, lower in jd_skill_names_orig.items() if orig not in all_matched_jd_orig])

    score = len(all_matched_jd_orig) / len(jd_skill_names_lower)
    results["score"] = round(score, 4)

    logger.info(f"Final Skill Match: Total matched (exact + semantic) = {len(all_matched_jd_orig)}/{len(jd_skill_names_lower)}. Score: {results['score']:.2f}")
    logger.info(f"Exact Matched Skills: {results['matched_skills']}")
    logger.info(f"Semantic Matched Pairs (JD Skill, Resume Skill, Score): {results['semantically_matched']}")
    logger.info(f"Final Missing Skills: {results['missing_skills']}")
    return results

def calculate_total_experience(work_experience):
    """Calculates approximate total years of relevant experience from parsed work entries."""
    total_months = 0
    processed_intervals = [] # To handle overlapping intervals

    # Convert work experience entries to intervals
    intervals = []
    today = datetime.now()
    for entry in work_experience:
        start_str = entry.get("startDate")
        end_str = entry.get("endDate")
        
        if not start_str:
            continue # Cannot calculate without start date
            
        try:
            start_date = datetime.strptime(start_str, "%Y-%m")
            if end_str and end_str.lower() == "present":
                end_date = today
            elif end_str:
                end_date = datetime.strptime(end_str, "%Y-%m")
                # Ensure end date is not before start date
                if end_date < start_date:
                     logger.warning(f"End date {end_str} is before start date {start_str} in experience entry. Skipping.")
                     continue
            else:
                # If no end date, assume it ended shortly after start? Or skip?
                # For simplicity, let's skip entries without an end date or 'Present'
                logger.debug(f"Experience entry missing end date or Present. Skipping duration calculation for: {entry.get(position)}")
                continue
                
            intervals.append((start_date, end_date))
            
        except ValueError as e:
            logger.warning(f"Could not parse dates {start_str} or {end_str}: {e}. Skipping entry.")
            continue
            
    if not intervals:
        return 0
        
    # Sort intervals by start date
    intervals.sort(key=lambda x: x[0])
    
    # Merge overlapping intervals
    merged = []
    if intervals:
        current_start, current_end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start <= current_end: # Overlap or adjacent
                current_end = max(current_end, next_end) # Extend the current interval
            else:
                merged.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        merged.append((current_start, current_end))
        
    # Calculate total duration from merged intervals
    total_duration = relativedelta()
    for start, end in merged:
        total_duration += relativedelta(end, start)
        
    # Convert total duration to approximate years
    total_years = total_duration.years + total_duration.months / 12.0
    logger.info(f"Calculated total non-overlapping experience: {total_years:.2f} years")
    return total_years

def compare_experience_years(resume_work_experience, jd_required_years):
    """Compares calculated resume experience with required years. Returns score 0.0-1.0."""
    if jd_required_years is None or jd_required_years <= 0:
        logger.info("No specific experience years required by JD. Score: 1.0")
        return 1.0 # No specific requirement

    resume_total_years = calculate_total_experience(resume_work_experience)

    # Scoring logic: 
    # 1.0 if resume years >= required years
    # Linear score from 0.0 to 1.0 if resume years < required years
    # Example: If 5 years required, 4 years gets 4/5 = 0.8 score
    if resume_total_years >= jd_required_years:
        score = 1.0
    else:
        score = resume_total_years / jd_required_years
        
    score = max(0.0, min(1.0, score)) # Ensure score is within [0, 1]
    logger.info(f"Experience Years Match: Resume={resume_total_years:.2f}y, Required={jd_required_years}y. Score: {score:.2f}")
    return round(score, 4)

def compare_education(resume_education, jd_required_education):
    """Compares highest education level achieved with required level. Returns score 0.0-1.0."""
    if jd_required_education is None:
        logger.info("No specific education level required by JD. Score: 1.0")
        return 1.0 # No specific requirement

    # Define numerical levels for comparison
    level_map = {"associate": 1, "bachelor": 2, "master": 3, "phd": 4}
    jd_req_level = level_map.get(jd_required_education.lower(), 0)
    if jd_req_level == 0:
         logger.warning(f"Could not map JD required education {jd_required_education} to a level. Assuming no requirement.")
         return 1.0

    highest_resume_level = 0
    for edu in resume_education:
        study_type = edu.get("studyType")
        if study_type:
            level = level_map.get(study_type.lower(), 0)
            highest_resume_level = max(highest_resume_level, level)

    # Scoring: 1.0 if resume level >= required level, 0.0 otherwise
    # Could potentially give partial score if level is just below?
    score = 1.0 if highest_resume_level >= jd_req_level else 0.0
    logger.info(f"Education Match: Resume Level={highest_resume_level}, Required Level={jd_req_level}. Score: {score:.2f}")
    return round(score, 4)

# --- Semantic Matching Function ---

def get_resume_text_for_matching(parsed_resume_data):
    # ... (keep existing logic) ...
    basics = parsed_resume_data.get("basics", {})
    resume_summary = basics.get("summary", "") or ""
    resume_skills = " ".join([s.get("name", "") for s in parsed_resume_data.get("skills", [])])
    experience_parts = []
    for work_item in parsed_resume_data.get("work", []):
        position = work_item.get("position", "")
        desc = work_item.get("summary", "") or work_item.get("description", "")
        highlights = "\n".join(work_item.get("highlights", []))
        experience_parts.append(f"{position}\n{desc}\n{highlights}")
    resume_experience = "\n\n".join(experience_parts).strip()
    # if not resume_experience:
    #      resume_experience = parsed_resume_data.get("_experience_text", "") or "" # Fallback removed

    resume_full_text_for_matching = f"{resume_summary}\n\n{resume_skills}\n\n{resume_experience}".strip()
    return resume_full_text_for_matching

def calculate_semantic_match(resume_text, jd_text):
    # ... (keep existing logic) ...
    sentence_model = load_sentence_transformer_model()
    if not sentence_model or not resume_text or not jd_text:
        logger.warning("Cannot calculate semantic match due to missing model or text.")
        return 0.0
    try:
        logger.info("Encoding texts for semantic similarity calculation...")
        resume_embedding = sentence_model.encode(resume_text, convert_to_tensor=True)
        jd_embedding = sentence_model.encode(jd_text, convert_to_tensor=True)
        cosine_scores = util.cos_sim(resume_embedding, jd_embedding)
        similarity_score = cosine_scores.item()
        logger.info(f"Calculated Semantic Similarity Score: {similarity_score:.4f}")
        return round(max(0.0, min(1.0, similarity_score)), 4)
    except Exception as e:
        logger.error(f"Error during semantic similarity calculation: {e}", exc_info=True)
        return 0.0

# --- Main Matching Function (Combined Score) ---

def match_resume_to_jd(parsed_resume_data, jd_text):
    """Performs matching using both semantic and granular field comparisons."""
    logger.info("\nStarting enhanced resume-JD matching process...")
    parsed_jd_data = parse_job_description(jd_text)
    resume_match_text = get_resume_text_for_matching(parsed_resume_data)

    # --- Calculate individual scores (0.0 to 1.0) --- 
    semantic_score = calculate_semantic_match(resume_match_text, parsed_jd_data["match_text"])
    skill_match_results = compare_skills(parsed_resume_data.get("skills", []), parsed_jd_data["required_skills"])
    skill_score = skill_match_results["score"]
    
    # Use refined experience and education matching
    experience_score = compare_experience_years(parsed_resume_data.get("work", []), parsed_jd_data["required_experience_years"])
    education_score = compare_education(parsed_resume_data.get("education", []), parsed_jd_data["required_education"])

    # --- Combine scores with weights --- 
    # Weights might need adjustment based on importance and reliability of scores
    weights = {
        "semantic": 0.30, # Reduced weight as granular scores improve
        "skills": 0.40,   # Increased weight for skills
        "experience": 0.20,
        "education": 0.10
    }
    # Ensure weights sum to 1 (or normalize if they don't)
    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0):
        logger.warning(f"Score weights do not sum to 1.0 (sum={total_weight}). Normalizing weights.")
        weights = {k: v / total_weight for k, v in weights.items()}

    combined_score = (
        semantic_score * weights["semantic"] +
        skill_score * weights["skills"] +
        experience_score * weights["experience"] +
        education_score * weights["education"]
    )

    combined_score = max(0.0, min(1.0, combined_score))
    final_match_percentage = round(combined_score * 100, 2)

    logger.info(f"\nCombined Weighted Score: {combined_score:.4f}")
    logger.info(f"Final Match Percentage: {final_match_percentage:.2f}%")
    logger.info("Matching process finished.")

    return {
        "match_percentage": final_match_percentage,
        "details": { 
            "semantic_score": semantic_score,
            "skill_score": skill_score,
            "matched_skills": skill_match_results["matched_skills"],
            "missing_skills": skill_match_results["missing_skills"],
            "semantically_matched_skills": skill_match_results["semantically_matched"],
            "experience_score": experience_score,
            "education_score": education_score,
            "calculated_resume_experience_years": round(calculate_total_experience(parsed_resume_data.get("work", [])), 2), # Add calculated years
            "weights": weights,
            "parsed_jd": parsed_jd_data
        }
    }

