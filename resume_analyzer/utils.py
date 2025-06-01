# -*- coding: utf-8 -*-
"""Utility functions for the Resume Analyzer."""

import re

# --- Regex Patterns ---
EMAIL_REGEX = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
PHONE_REGEX = r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
URL_REGEX = r"https?://[\w\-\.]+(?:\:[0-9]+)?(?:/[^\s]*)?"

# --- Section Extraction Logic ---

def find_section_indices(lines, keywords):
    """Finds start and end line indices for a section based on keywords."""
    start_index = -1
    end_index = len(lines)
    found_start = False

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped: # Skip empty lines for header detection
            continue

        line_lower = line_stripped.lower()

        # Check if the current line matches any keyword
        is_keyword_match = any(keyword.lower() in line_lower for keyword in keywords)

        if not found_start and is_keyword_match:
            # Found the start of the target section
            start_index = i
            found_start = True
        elif found_start and is_keyword_match:
            # Found the start of the *next* section with the *same* keywords (e.g., multiple project sections)
            # Treat this as the end of the previous section
            end_index = i
            break
        elif found_start and not is_keyword_match:
            # Check if this line looks like a header for *any* common section
            # Simple heuristic: ALL CAPS or Title Case and relatively short
            # This helps find the end boundary when the next section has different keywords
            is_potential_header = (line_stripped.isupper() and len(line_stripped.split()) < 6) or \
                                  (line_stripped.istitle() and len(line_stripped.split()) < 6)
            # Avoid matching very short lines that aren't headers
            if is_potential_header and len(line_stripped) > 4:
                # Found the start of a different section, so end the current one
                end_index = i
                break

    return start_index, end_index

def extract_section_text(text, keywords):
    """Extracts text content for a specific section using keywords.

    Args:
        text (str): The full text of the document.
        keywords (list): A list of keywords (case-insensitive) that typically
                         start the desired section (e.g., ["Experience", "Work History"]).

    Returns:
        str: The extracted text of the section, or an empty string if not found.
    """
    lines = text.split("\n")
    start_index, end_index = find_section_indices(lines, keywords)

    if start_index != -1:
        # Extract lines from *after* the header line up to the end index
        section_lines = lines[start_index + 1 : end_index]
        # Join lines and clean up leading/trailing whitespace
        section_text = "\n".join(section_lines).strip()
        
        # For skills section, try to handle bullet points and lists better
        if any(k.lower() in ["skills", "technical skills", "proficiencies", "expertise"] for k in keywords):
            # Split by common list delimiters and clean up
            skills_lines = []
            for line in section_lines:
                # Handle bullet points and numbered lists
                line = re.sub(r'^[\d\.\-\*•]+[\s]*', '', line.strip())
                if line:
                    skills_lines.append(line)
            section_text = "\n".join(skills_lines).strip()
            
        # For experience section, try to preserve structure better
        elif any(k.lower() in ["experience", "work history", "employment", "professional experience"] for k in keywords):
            # Keep the structure but clean up extra whitespace
            exp_lines = []
            for line in section_lines:
                line = line.strip()
                if line:
                    exp_lines.append(line)
            section_text = "\n".join(exp_lines).strip()
            
        return section_text
    else:
        return "" # Section not found

# Add other utility functions as needed, e.g., text cleaning, date parsing

