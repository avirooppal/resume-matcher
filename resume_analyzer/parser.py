# -*- coding: utf-8 -*-
"""Functions for parsing resume text and extracting structured information.

Enhanced version focusing on more detailed extraction for Work Experience, Education, Projects, Certificates.
"""

import re
import spacy
import logging
from dateutil import parser as date_parser
from dateutil.parser import ParserError

from .models import load_spacy_model
from .utils import extract_section_text, EMAIL_REGEX, PHONE_REGEX, URL_REGEX
from .config import (
    SUMMARY_KEYWORDS, SKILLS_KEYWORDS, EXPERIENCE_KEYWORDS,
    EDUCATION_KEYWORDS, PROJECTS_KEYWORDS, CERTIFICATES_KEYWORDS,
    LANGUAGES_KEYWORDS, PUBLICATIONS_KEYWORDS # Assuming PUBLICATIONS might be needed later
)

# --- Setup Logging ---
logger = logging.getLogger(__name__)

# --- Date Parsing Helper (Improved) ---

def parse_date_range(text):
    """Attempts to parse start and end dates from a text string using multiple patterns and dateutil."""
    startDate, endDate = None, None
    text_lower = text.lower()

    # Prioritize patterns with explicit start and end
    patterns = [
        # Month YYYY - Month YYYY / Present / Current
        r"(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\s*[-–—to]+\s*(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Present|Current)[a-z]*\.?\s+\d{4}\b|Present|Current)",
        # YYYY - YYYY / Present / Current
        r"(\b\d{4}\b)\s*[-–—to]+\s*(\b\d{4}\b|Present|Current)",
        # Month YYYY (treat as start, end=Present if context suggests)
        # r"(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})",
        # YYYY (treat as start, end=Present if context suggests)
        # r"(\b\d{4}\b)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                start_str = match.group(1)
                startDate = date_parser.parse(start_str).strftime("%Y-%m") # Use YYYY-MM for consistency
            except (ParserError, ValueError) as e:
                logger.debug(f"Failed parsing start date 	'{start_str}	' with dateutil: {e}")
                startDate = None

            if len(match.groups()) > 1:
                end_str = match.group(2)
                if end_str and end_str.lower() in ["present", "current"]:
                    endDate = "Present"
                elif end_str:
                    try:
                        endDate = date_parser.parse(end_str).strftime("%Y-%m")
                    except (ParserError, ValueError) as e:
                         logger.debug(f"Failed parsing end date 	'{end_str}	' with dateutil: {e}")
                         endDate = None
            # If only start date found in this pattern, assume ongoing if text contains "present"/"current"
            # elif startDate and any(term in text_lower for term in ["present", "current"]):
            #      endDate = "Present"

            if startDate or endDate == "Present": # Found a plausible range
                 logger.debug(f"Parsed date range: {startDate} - {endDate} from text: 	'{text[:50]}...	'")
                 return startDate, endDate

    # Fallback: If no range found, look for single dates (less reliable)
    try:
        # Find all potential date strings using NER first
        nlp = load_spacy_model()
        doc = nlp(text)
        date_entities = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        parsed_dates = []
        for date_str in date_entities:
             try:
                 # Try parsing, ignore ambiguous dates without year
                 parsed = date_parser.parse(date_str, ignoretz=True)
                 # Check if year seems reasonable (e.g., > 1950)
                 if parsed.year > 1950:
                      parsed_dates.append(parsed)
             except (ParserError, ValueError):
                 continue
        
        if len(parsed_dates) == 1:
            startDate = parsed_dates[0].strftime("%Y-%m")
            # Assume ongoing if text contains "present"/"current"
            if any(term in text_lower for term in ["present", "current"]):
                 endDate = "Present"
        elif len(parsed_dates) >= 2:
            parsed_dates.sort()
            startDate = parsed_dates[0].strftime("%Y-%m")
            endDate = parsed_dates[-1].strftime("%Y-%m")
            # Check if latest date is very recent and text mentions present/current
            # if parsed_dates[-1].year >= (datetime.now().year -1) and any(term in text_lower for term in ["present", "current"]):
            #      endDate = "Present"

    except Exception as e:
        logger.warning(f"Error during fallback date parsing: {e}")

    if startDate or endDate == "Present":
        logger.debug(f"Parsed date range (fallback): {startDate} - {endDate} from text: 	'{text[:50]}...	'")

    return startDate, endDate

# --- Extraction Helper Functions (Refined) ---

def extract_name_and_label(doc):
    name, label = None, None
    name_ent = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            name_ent = ent
            break
    if not name:
         for token in doc[:5]:
            if token.is_title and token.is_alpha and len(token.text) > 1:
                 if token.i + 1 < len(doc) and doc[token.i+1].is_title and doc[token.i+1].is_alpha:
                     name = f"{token.text} {doc[token.i+1].text}"
                 else:
                     name = token.text
                 break
    if name_ent:
        name_end_char = name_ent.end_char
        potential_label_text = doc.text[name_end_char : min(name_end_char + 150, len(doc.text))]
        lines_after_name = potential_label_text.split("\n")
        for line in lines_after_name:
             line = line.strip()
             if line and line.istitle() and 1 < len(line.split()) < 7 and "@" not in line and not re.search(PHONE_REGEX, line) and not re.search(URL_REGEX, line):
                 label = line
                 break
    if not label:
        for line in doc.text.split("\n")[:5]:
            line = line.strip()
            if line and line != name and line.istitle() and 1 < len(line.split()) < 7 and "@" not in line and not re.search(PHONE_REGEX, line) and not re.search(URL_REGEX, line):
                 label = line
                 break
    return name, label

def extract_contact_info(text):
    emails = list(set(re.findall(EMAIL_REGEX, text)))
    phones = list(set(re.findall(PHONE_REGEX, text)))
    email = emails[0] if emails else None
    phone = phones[0] if phones else None
    return email, phone

def extract_urls(text):
    urls = re.findall(URL_REGEX, text)
    profiles = []
    website = None
    processed_urls = set()
    for url in urls:
        if url in processed_urls:
            continue
        processed_urls.add(url)
        url_lower = url.lower()
        network = None
        username = None
        if "linkedin.com/in/" in url_lower:
            network = "LinkedIn"
            match = re.search(r"linkedin.com/in/([^/?#]+)", url)
            if match: username = match.group(1)
        elif "github.com/" in url_lower:
            network = "GitHub"
            match = re.search(r"github.com/([^/?#]+)", url)
            if match: username = match.group(1)
        elif "stackoverflow.com/users/" in url_lower:
             network = "StackOverflow"
             match = re.search(r"stackoverflow.com/users/(\d+)/([^/?#]+)", url)
             if match: username = match.group(2)
        if network:
            profiles.append({"network": network, "username": username, "url": url})
        elif not website and all(pf not in url_lower for pf in ["linkedin", "github", "stackoverflow"]):
             website = url
    return profiles, website

def extract_skills_from_section(text):
    """Extracts skills from a section of text, handling various formats and delimiters."""
    skills_text = extract_section_text(text, SKILLS_KEYWORDS)
    if not skills_text:
        return []
        
    # Common skill categories and their keywords
    skill_categories = {
        "programming": ["programming", "languages", "coding", "development"],
        "frameworks": ["frameworks", "libraries", "technologies"],
        "databases": ["databases", "database", "db"],
        "tools": ["tools", "software", "applications"],
        "cloud": ["cloud", "aws", "azure", "gcp"],
        "devops": ["devops", "ci/cd", "deployment"],
        "soft_skills": ["soft skills", "interpersonal", "communication"]
    }
    
    # Common programming languages and technologies
    common_skills = {
        "programming": ["python", "java", "javascript", "typescript", "c++", "c#", "ruby", "php", "swift", "kotlin", "go", "rust"],
        "frameworks": ["react", "angular", "vue", "django", "flask", "spring", "express", "node.js", "asp.net", "laravel", "rails"],
        "databases": ["sql", "mysql", "postgresql", "mongodb", "redis", "oracle", "sqlite", "cassandra"],
        "tools": ["git", "docker", "kubernetes", "jenkins", "jira", "confluence", "vscode", "eclipse", "intellij"],
        "cloud": ["aws", "azure", "gcp", "heroku", "digitalocean", "cloudflare"],
        "devops": ["ci/cd", "jenkins", "github actions", "gitlab ci", "travis ci", "ansible", "terraform"]
    }
    
    # Split text into potential skill entries
    potential_skills = []
    
    # First try splitting by common delimiters
    skill_entries = re.split(r"[\n,;•*-]+", skills_text)
    
    # Process each entry
    for entry in skill_entries:
        entry = entry.strip()
        if not entry or len(entry) < 2:  # Skip empty or very short entries
            continue
            
        # Handle entries with skill levels
        level_match = re.match(r"(.+?)(?:\s*[-–()]+\s*([A-Za-z]+)\)?)?$", entry)
        if level_match:
            skill = level_match.group(1).strip()
            level = level_match.group(2).strip() if level_match.group(2) else None
        else:
            skill = entry
            level = None
            
        # Skip if skill is too long (likely not a skill)
        if len(skill) > 50:
            continue
            
        # Clean up the skill name
        skill = re.sub(r"^\d+\.\s*", "", skill)  # Remove numbered lists
        skill = re.sub(r"^[\-\*•]\s*", "", skill)  # Remove bullet points
        skill = skill.strip()
        
        # Skip if skill is too short after cleaning
        if len(skill) < 2:
            continue
            
        # Skip if skill is just a single word that's too short
        if len(skill.split()) == 1 and len(skill) < 3:
            continue
            
        # Skip if skill is just a common word
        if skill.lower() in ['and', 'or', 'the', 'with', 'in', 'on', 'at', 'to', 'for']:
            continue
            
        if skill:
            potential_skills.append({"name": skill, "level": level})
            
    # Group skills by category if possible
    categorized_skills = []
    uncategorized_skills = []
    
    for skill in potential_skills:
        skill_lower = skill["name"].lower()
        categorized = False
        
        # First check against common skills
        for category, skills in common_skills.items():
            if any(s in skill_lower for s in skills):
                categorized_skills.append({
                    "name": skill["name"],
                    "level": skill["level"],
                    "category": category
                })
                categorized = True
                break
                
        # If not found in common skills, check category keywords
        if not categorized:
            for category, keywords in skill_categories.items():
                if any(keyword in skill_lower for keyword in keywords):
                    categorized_skills.append({
                        "name": skill["name"],
                        "level": skill["level"],
                        "category": category
                    })
                    categorized = True
                    break
                    
        if not categorized:
            uncategorized_skills.append(skill)
            
    # Combine categorized and uncategorized skills
    all_skills = categorized_skills + uncategorized_skills
    
    # Remove duplicates while preserving order
    seen = set()
    unique_skills = []
    for skill in all_skills:
        skill_name = skill["name"].lower()
        if skill_name not in seen:
            seen.add(skill_name)
            unique_skills.append(skill)
            
    return unique_skills

def extract_languages(text):
    lang_text = extract_section_text(text, LANGUAGES_KEYWORDS)
    languages = []
    if lang_text:
        potential_langs = re.split(r"[\n,;]+", lang_text)
        for lang_entry in potential_langs:
            lang_entry = lang_entry.strip()
            if not lang_entry:
                continue
            match = re.match(r"([a-zA-Z\s]+)(?:\s*[-–()]+\s*([a-zA-Z]+)\)?)?", lang_entry)
            if match:
                language = match.group(1).strip()
                fluency = match.group(2).strip() if match.group(2) else None
            else:
                language = lang_entry
                fluency = None
            if language and len(language) > 1:
                 languages.append({"language": language, "fluency": fluency})
    return languages

# --- Enhanced Section Parsing --- 

def parse_experience_section(text):
    experience_text = extract_section_text(text, EXPERIENCE_KEYWORDS)
    if not experience_text:
        return []
        
    # Split into potential entries based on common separators
    blocks = re.split(r"\n\s*\n", experience_text) # Split by blank lines first
    if len(blocks) <= 1:
        # If no blank lines, try splitting before lines that look like titles/dates
        blocks = re.split(r"(?=\n(?:[A-Z][a-zA-Z\s\-/,&]+?\s*(?:\bat\b|\|)\s*[A-Z])|(?=\n\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b))", experience_text)
         
    experiences = []
    nlp = load_spacy_model()
    
    for entry_text in blocks:
        entry_text = entry_text.strip()
        if not entry_text or len(entry_text) < 20: # Skip very short blocks
            continue
            
        logger.debug(f"Processing experience block: {entry_text[:100]}...")
        doc = nlp(entry_text)
        
        # Extract details using NER and patterns
        position = None
        name = None # Company
        location = None
        startDate, endDate = parse_date_range(entry_text)
        
        # Extract organization and location using NER
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        gpes = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        
        # Look for job titles and company names using patterns
        lines = entry_text.split("\n")
        first_line = lines[0].strip() if lines else ""
        
        # Try to extract position and company from first line
        if first_line:
            # Common patterns: "Position at Company" or "Position | Company"
            position_company_match = re.search(r"^([^|@]+?)(?:\s+(?:at|@|\|)\s+|\s+)([^|@]+)$", first_line)
            if position_company_match:
                position = position_company_match.group(1).strip()
                name = position_company_match.group(2).strip()
            else:
                # If no clear separator, check if it's a valid position
                # Skip if it's too short or doesn't look like a position
                if len(first_line) > 3 and not first_line.endswith('.') and not first_line.isupper():
                    position = first_line
                
        # If no position found yet, look for common job titles
        if not position:
            potential_titles = []
            for sent in doc.sents:
                for i, token in enumerate(sent):
                    if token.is_title and token.pos_ in ["PROPN", "NOUN"]:
                        # Check if part of an ORG or GPE
                        is_org_gpe = any(token.idx >= ent.start_char and token.idx < ent.end_char 
                                       for ent in doc.ents if ent.label_ in ["ORG", "GPE"])
                        if not is_org_gpe:
                            # Extend to multi-word titles
                            current_title = token.text
                            j = i + 1
                            while j < len(sent) and sent[j].is_title and sent[j].pos_ in ["PROPN", "NOUN", "ADP"]:
                                current_title += " " + sent[j].text
                                j += 1
                            # Check for common job title keywords
                            if any(t in current_title.lower() for t in [
                                "engineer", "developer", "manager", "analyst", "specialist", 
                                "lead", "architect", "consultant", "director", "officer",
                                "coordinator", "associate", "assistant", "executive",
                                "developer", "programmer", "designer", "administrator",
                                "consultant", "advisor", "researcher", "scientist",
                                "instructor", "teacher", "professor", "lecturer"
                            ]):
                                potential_titles.append(current_title)
                break # Only check first sentence usually
                
            if potential_titles:
                position = potential_titles[0]
                
        # If no company found yet, use first ORG entity
        if not name and orgs:
            name = orgs[0]
            
        # Use first GPE entity as location if found
        if gpes:
            location = gpes[0]

        # Extract highlights/summary
        highlights = []
        summary_lines = []
        in_highlights = False
        
        for line in lines[1:]: # Skip first line (title/company)
            stripped_line = line.strip()
            # Skip empty lines and date lines
            if not stripped_line or re.search(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b", stripped_line):
                continue
                
            # Handle bullet points and numbered lists
            if re.match(r"^[\d\.\-\*•]+[\s]*", stripped_line):
                highlight = re.sub(r"^[\d\.\-\*•]+[\s]*", "", stripped_line)
                if highlight:
                    highlights.append(highlight)
                    in_highlights = True
            elif len(stripped_line) > 10:
                if in_highlights:
                    # If we were in highlights and hit a non-bullet line, it might be a new section
                    break
                summary_lines.append(stripped_line)
                
        summary = "\n".join(summary_lines).strip()

        # Only add if we found a valid position and company
        if position and name and len(position) > 3 and len(name) > 3:
            # Additional validation to avoid false positives
            if not position.endswith('.') and not name.endswith('.') and \
               not position.isupper() and not name.isupper() and \
               not any(word in position.lower() for word in ['boot', 'camp', 'course', 'training']):
                experiences.append({
                    "position": position,
                    "name": name,
                    "startDate": startDate,
                    "endDate": endDate,
                    "summary": summary,
                    "highlights": highlights,
                    "location": location,
                    "url": None,
                    "description": summary if not highlights else None
                })
            else:
                logger.debug(f"Skipping experience block, position or company looks invalid: {position} at {name}")
        else:
            logger.debug(f"Skipping experience block, couldn't identify valid position or company: {entry_text[:100]}...")

    return experiences

def parse_education_section(text):
    education_text = extract_section_text(text, EDUCATION_KEYWORDS)
    if not education_text:
        return []
    # Split by blank lines or lines starting with institution/degree
    blocks = re.split(r"\n\s*\n", education_text)
    if len(blocks) <= 1:
         blocks = re.split(r"(?=\n(?:[A-Z][a-zA-Z\s.,]+(?:University|Institute|College|School))|(?:\n(?:B\.S\.|M\.S\.|Ph\.D\.)))", education_text)

    educations = []
    nlp = load_spacy_model()
    for entry_text in blocks:
        entry_text = entry_text.strip()
        if not entry_text or len(entry_text) < 10:
            continue
        logger.debug(f"Processing education block: {entry_text[:100]}...")
        doc = nlp(entry_text)
        
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        studies = [ent.text for ent in doc.ents if ent.label_ == "WORK_OF_ART"] # Often catches degrees/majors
        gpes = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        
        institution = orgs[0] if orgs else None
        area = studies[0] if studies else None
        studyType = None
        location = gpes[0] if gpes else None
        
        # Refine area/studyType
        potential_areas = studies
        # Look for keywords like 'Major:', 'Minor:'
        major_match = re.search(r"(?:Major|Minor)[:\s]+([A-Za-z\s,&]+)", entry_text, re.IGNORECASE)
        if major_match: potential_areas.append(major_match.group(1).strip())
        if potential_areas: area = potential_areas[0] # Simplistic choice
        
        if area:
            area_lower = area.lower()
            if any(t in area_lower for t in ["bachelor", "b.s", "b.a"]): studyType = "Bachelor"
            elif any(t in area_lower for t in ["master", "m.s", "m.a", "mba"]): studyType = "Master"
            elif any(t in area_lower for t in ["ph.d", "doctorate"]): studyType = "Doctorate"
            elif any(t in area_lower for t in ["associate"]): studyType = "Associate"
            
        startDate, endDate = parse_date_range(entry_text)
        
        gpa = None
        score = None
        courses = []
        gpa_match = re.search(r"GPA[:\s]*([\d.]+)(?:\s*/\s*([\d.]+))?", entry_text, re.IGNORECASE)
        if gpa_match: gpa = gpa_match.group(1)
        # Look for courses section/keywords
        courses_match = re.search(r"(?:Relevant Coursework|Courses)[:\s]+([\w\s,]+(?:\n[\w\s,]+)*)", entry_text, re.IGNORECASE | re.MULTILINE)
        if courses_match:
             courses = [c.strip() for c in re.split(r"[\n,;]+", courses_match.group(1)) if c.strip()]

        if institution or area:
            educations.append({
                "institution": institution.strip() if institution else None,
                "area": area.strip() if area else None,
                "studyType": studyType,
                "startDate": startDate,
                "endDate": endDate,
                "gpa": gpa,
                "score": score,
                "courses": courses,
                "location": location # Added location if found
            })
        else:
             logger.debug(f"Skipping education block, couldn't identify institution or area: {entry_text[:100]}...")
             
    return educations

def parse_projects_section(text):
    projects_text = extract_section_text(text, PROJECTS_KEYWORDS)
    if not projects_text:
        return []
    # Assume projects are separated by blank lines or lines starting with a clear project name
    blocks = re.split(r"\n\s*\n", projects_text)
    if len(blocks) <= 1:
         blocks = re.split(r"(?=\n[A-Z][\w\s]+:) # Look for Name: pattern or similar", projects_text)
         
    projects = []
    nlp = load_spacy_model()
    for entry_text in blocks:
        entry_text = entry_text.strip()
        if not entry_text or len(entry_text) < 15:
            continue
        logger.debug(f"Processing project block: {entry_text[:100]}...")
        doc = nlp(entry_text)
        
        name = None
        description = []
        highlights = []
        keywords = [] # e.g., technologies used
        startDate, endDate = parse_date_range(entry_text)
        url = None
        roles = [] # If applicable
        
        lines = entry_text.split("\n")
        # Assume first non-empty line is the name
        for i, line in enumerate(lines):
             line = line.strip()
             if line:
                 name = line
                 lines = lines[i+1:] # Process remaining lines
                 break
                 
        # Extract URL if present
        project_urls = re.findall(URL_REGEX, entry_text)
        if project_urls: url = project_urls[0]
        
        # Extract highlights and description
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith(("-", "*", "•")):
                highlight = stripped_line[1:].strip()
                if highlight: highlights.append(highlight)
            elif stripped_line:
                 description.append(stripped_line)
                 
        # Extract keywords (look for specific labels like 'Technologies:', 'Stack:')
        keywords_match = re.search(r"(?:Technologies|Stack|Keywords)[:\s]+([\w\s,]+(?:\n[\w\s,]+)*)", entry_text, re.IGNORECASE | re.MULTILINE)
        if keywords_match:
             keywords = [k.strip() for k in re.split(r"[\n,;]+", keywords_match.group(1)) if k.strip()]

        if name:
            projects.append({
                "name": name,
                "description": "\n".join(description).strip(),
                "highlights": highlights,
                "keywords": keywords,
                "startDate": startDate,
                "endDate": endDate,
                "url": url,
                "roles": roles # Role extraction would need more specific logic
            })
        else:
             logger.debug(f"Skipping project block, couldn't identify name: {entry_text[:100]}...")
             
    return projects

def parse_certificates_section(text):
    certs_text = extract_section_text(text, CERTIFICATES_KEYWORDS)
    if not certs_text:
        return []
    # Assume one cert per line or separated by common delimiters
    potential_certs = re.split(r"[\n,;]+", certs_text)
    certificates = []
    nlp = load_spacy_model()
    for entry_text in potential_certs:
        entry_text = entry_text.strip()
        if not entry_text or len(entry_text) < 5:
            continue
        logger.debug(f"Processing certificate entry: {entry_text[:100]}...")
        doc = nlp(entry_text)
        
        name = entry_text # Default to full line
        date = None
        issuer = None
        
        # Try to find issuer (ORG)
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        if orgs: issuer = orgs[0]
        
        # Try to find date
        startDate, _ = parse_date_range(entry_text) # Use start date as the issue date
        date = startDate
        
        # Refine name (remove issuer/date if found in the string)
        if issuer and issuer in name: name = name.replace(issuer, "").strip(",\s")
        # Add more robust date removal if needed
        
        certificates.append({
            "name": name.strip(),
            "date": date,
            "issuer": issuer.strip() if issuer else None
        })
        
    return certificates

# --- Main Resume Parsing Function ---

def parse_resume_text(text):
    nlp = load_spacy_model()
    if not nlp or not text:
        logger.error("spaCy model not loaded or text is empty. Cannot parse.")
        return None

    logger.info("Processing text with spaCy model...")
    doc = nlp(text)
    logger.info("spaCy processing complete.")

    resume_data = {
        "basics": {
            "name": None, "label": None, "picture": None, "email": None, "phone": None,
            "summary": None, "location": {"address": None, "postalCode": None, "city": None, "countryCode": None, "region": None},
            "profiles": [], "url": None, "image": None
        },
        "work": [], "education": [], "publications": [], "skills": [],
        "languages": [], "projects": [], "certificates": []
    }

    logger.info("Extracting basic information...")
    name, label = extract_name_and_label(doc)
    resume_data["basics"]["name"] = name
    resume_data["basics"]["label"] = label
    email, phone = extract_contact_info(text)
    resume_data["basics"]["email"] = email
    resume_data["basics"]["phone"] = phone
    profiles, website = extract_urls(text)
    resume_data["basics"]["profiles"] = profiles
    resume_data["basics"]["url"] = website
    resume_data["basics"]["summary"] = extract_section_text(text, SUMMARY_KEYWORDS)
    gpe_entities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    if gpe_entities:
        resume_data["basics"]["location"]["city"] = gpe_entities[0]

    logger.info("Extracting skills...")
    skills_from_section = extract_skills_from_section(text)
    # Convert skills to the format expected by the matcher
    resume_data["skills"] = [
        {
            "name": skill["name"],
            "level": skill["level"],
            "keywords": [skill["name"]],  # Use skill name as keyword for matching
            "category": skill.get("category")  # Include category if available
        }
        for skill in skills_from_section
    ]
    
    logger.info("Extracting languages...")
    resume_data["languages"] = extract_languages(text)

    logger.info("Extracting work experience...")
    resume_data["work"] = parse_experience_section(text)

    logger.info("Extracting education...")
    resume_data["education"] = parse_education_section(text)
    
    logger.info("Extracting projects...")
    resume_data["projects"] = parse_projects_section(text)
    
    logger.info("Extracting certificates...")
    resume_data["certificates"] = parse_certificates_section(text)

    logger.info("Resume parsing finished.")
    return resume_data

