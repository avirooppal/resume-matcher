#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main inference script for the Modular Resume Analyzer."""

import argparse
import json
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from resume_analyzer.text_extractor import extract_text
from resume_analyzer.parser import parse_resume_text
from resume_analyzer.matcher import match_resume_to_jd
from resume_analyzer.config import DEFAULT_OUTPUT_FILENAME

def main():
    parser = argparse.ArgumentParser(
        description="Parse a resume, match it against a job description, and output structured JSON."
    )
    parser.add_argument(
        "resume_file",
        help="Path to the resume file (PDF, DOCX, TXT)."
    )
    parser.add_argument(
        "jd_file",
        help="Path to the job description file (PDF, DOCX, TXT)."
    )
    parser.add_argument(
        "-o", "--output",
        help=f"Optional path to save the output JSON file. Defaults to {DEFAULT_OUTPUT_FILENAME} in the current directory.",
        default=DEFAULT_OUTPUT_FILENAME
    )

    args = parser.parse_args()

    # --- Input Validation ---
    if not os.path.exists(args.resume_file):
        print(f"Error: Resume file not found at {args.resume_file}")
        sys.exit(1)
    if not os.path.exists(args.jd_file):
        print(f"Error: Job description file not found at {args.jd_file}")
        sys.exit(1)

    print(f"--- Starting Resume Analysis ---")
    print(f"Resume: {args.resume_file}")
    print(f"Job Description: {args.jd_file}")

    # --- Processing Steps ---
    # 1. Extract Text
    print("\nStep 1: Extracting text...")
    resume_text = extract_text(args.resume_file)
    jd_text = extract_text(args.jd_file)

    if not resume_text:
        print("Error: Failed to extract text from resume. Aborting.")
        sys.exit(1)
    if not jd_text:
        print("Warning: Failed to extract text from job description. Matching will be skipped.")

    # 2. Parse Resume
    print("\nStep 2: Parsing resume...")
    parsed_resume_data = parse_resume_text(resume_text)
    if not parsed_resume_data:
        print("Error: Failed to parse resume data. Aborting.")
        sys.exit(1)

    # 3. Match Resume to JD
    match_results = {"match_percentage": 0.0, "details": {}}
    if jd_text:
        print("\nStep 3: Matching resume to job description...")
        match_results = match_resume_to_jd(parsed_resume_data, jd_text)
    else:
        print("\nStep 3: Skipping matching due to missing JD text.")

    # --- Combine Results --- 
    # Remove temporary fields before final output
    parsed_resume_data.pop("_experience_text", None)

    final_output = {
        "parsed_resume": parsed_resume_data,
        "match_results": match_results
    }

    # --- Output --- 
    print("\n--- Analysis Complete ---")
    print(f"Overall Match Percentage: {match_results.get(	'match_percentage', 0.0):.2f}%")

    output_path = args.output
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        print(f"Output JSON saved to: {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"Error saving JSON to {output_path}: {e}")
        print("\n--- Analysis Result (JSON) ---")
        print(json.dumps(final_output, indent=2, ensure_ascii=False))

    print("--- Finished ---")

if __name__ == "__main__":
    main()

