# -*- coding: utf-8 -*-
"""Configuration settings for the Resume Analyzer."""

import os

# --- Model Configuration ---
# spaCy model for NER
SPACY_MODEL_NAME = "en_core_web_trf"

# Sentence Transformer model for semantic similarity
# Options: 'all-MiniLM-L6-v2', 'multi-qa-MiniLM-L6-cos-v1', 'paraphrase-MiniLM-L6-v2'
SENTENCE_MODEL_NAME = 'all-MiniLM-L6-v2'

# Device for running models ('cuda' or 'cpu')
# Auto-detect will be handled in the loading functions
DEFAULT_DEVICE = "cpu"

# --- Extraction Keywords ---
# Keywords to identify different sections in resumes/JDs
SUMMARY_KEYWORDS = ["Summary", "Objective", "Profile", "About Me"]
SKILLS_KEYWORDS = ["Skills", "Technical Skills", "Proficiencies", "Expertise"]
EXPERIENCE_KEYWORDS = ["Experience", "Work History", "Employment", "Professional Experience"]
EDUCATION_KEYWORDS = ["Education", "Academic Background", "Qualifications"]
PROJECTS_KEYWORDS = ["Projects", "Personal Projects"]
CERTIFICATES_KEYWORDS = ["Certificates", "Certifications", "Licenses"]
LANGUAGES_KEYWORDS = ["Languages"]
PUBLICATIONS_KEYWORDS = ["Publications", "Papers"]

# Keywords for Job Descriptions
JD_REQUIREMENTS_KEYWORDS = ["Requirements", "Qualifications", "Skills Required", "Must Have"]
JD_RESPONSIBILITIES_KEYWORDS = ["Responsibilities", "Duties", "Role", "What You Will Do"]

# --- Output Configuration ---
DEFAULT_OUTPUT_FILENAME = "analysis_output.json"

# --- Other Settings ---
# Add any other configurable parameters here

