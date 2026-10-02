#!/usr/bin/env python3
"""
Merge Flash-2.0 and Gemini Pro-3 Targeted Extraction Results

This script combines:
- Flash-2.0 results (baseline extraction with all fields)
- Gemini Pro-3 targeted results (only specific fields where Gemini excels)

Usage:
    python3 merge_flash_gemini.py --flash flash_results/ --gemini gemini_results/ --output merged_results/
"""

import json
import os
import sys
from pathlib import Path
import argparse


def load_json_safe(filepath):
    """Load JSON file with error handling"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON decode error in {filepath}: {e}")
        return None
    except Exception as e:
        print(f"ERROR: Could not load {filepath}: {e}")
        return None


def merge_results(flash_data, gemini_data):
    """
    Merge Flash-2.0 (baseline) with Gemini Pro-3 (targeted) results.

    Gemini Pro-3 targeted fields override Flash-2.0 for:
    - research_context.heritage_values_discussed
    - evaluation.qualitative_results
    - evaluation.error_analysis
    - findings_and_contributions.* (ALL fields)
    - ethics_and_reproducibility.privacy_and_data_protection
    - ethics_and_reproducibility.bias_fairness_and_representativeness
    - ethics_and_reproducibility.cultural_and_community_considerations
    - ethics_and_reproducibility.limitations_acknowledged

    All other fields: Keep Flash-2.0 results
    """

    if not flash_data:
        print("WARNING: No Flash data, using Gemini only")
        return gemini_data

    if not gemini_data:
        print("WARNING: No Gemini data, using Flash only")
        return flash_data

    # Start with Flash-2.0 as baseline
    merged = flash_data.copy()

    # BLOCK 2: Replace heritage_values_discussed from Gemini
    if 'research_context' in gemini_data:
        if 'heritage_values_discussed' in gemini_data['research_context']:
            gemini_values = gemini_data['research_context']['heritage_values_discussed']
            if gemini_values and gemini_values != "SKIP":
                if 'research_context' not in merged:
                    merged['research_context'] = {}
                merged['research_context']['heritage_values_discussed'] = gemini_values
                print("  ✓ Merged heritage_values_discussed from Gemini")

    # BLOCK 6: Replace qualitative evaluation fields from Gemini
    if 'evaluation' in gemini_data:
        if 'evaluation' not in merged:
            merged['evaluation'] = {}

        # qualitative_results
        if 'qualitative_results' in gemini_data['evaluation']:
            gemini_qual = gemini_data['evaluation']['qualitative_results']
            if gemini_qual and gemini_qual != "SKIP":
                merged['evaluation']['qualitative_results'] = gemini_qual
                print("  ✓ Merged qualitative_results from Gemini")

        # error_analysis
        if 'error_analysis' in gemini_data['evaluation']:
            gemini_error = gemini_data['evaluation']['error_analysis']
            if gemini_error and gemini_error != "SKIP":
                merged['evaluation']['error_analysis'] = gemini_error
                print("  ✓ Merged error_analysis from Gemini")

    # BLOCK 7: Replace ALL findings_and_contributions from Gemini
    if 'findings_and_contributions' in gemini_data:
        gemini_findings = gemini_data['findings_and_contributions']
        if gemini_findings and gemini_findings != "SKIP":
            merged['findings_and_contributions'] = gemini_findings
            print("  ✓ Merged findings_and_contributions (COMPLETE BLOCK) from Gemini")

    # BLOCK 8: Replace selected ethics fields from Gemini
    if 'ethics_and_reproducibility' in gemini_data:
        if 'ethics_and_reproducibility' not in merged:
            merged['ethics_and_reproducibility'] = {}

        gemini_ethics = gemini_data['ethics_and_reproducibility']

        # privacy_and_data_protection
        if 'privacy_and_data_protection' in gemini_ethics:
            if gemini_ethics['privacy_and_data_protection'] != "SKIP":
                merged['ethics_and_reproducibility']['privacy_and_data_protection'] = \
                    gemini_ethics['privacy_and_data_protection']
                print("  ✓ Merged privacy_and_data_protection from Gemini")

        # bias_fairness_and_representativeness
        if 'bias_fairness_and_representativeness' in gemini_ethics:
            if gemini_ethics['bias_fairness_and_representativeness'] != "SKIP":
                merged['ethics_and_reproducibility']['bias_fairness_and_representativeness'] = \
                    gemini_ethics['bias_fairness_and_representativeness']
                print("  ✓ Merged bias_fairness_and_representativeness from Gemini")

        # cultural_and_community_considerations
        if 'cultural_and_community_considerations' in gemini_ethics:
            if gemini_ethics['cultural_and_community_considerations'] != "SKIP":
                merged['ethics_and_reproducibility']['cultural_and_community_considerations'] = \
                    gemini_ethics['cultural_and_community_considerations']
                print("  ✓ Merged cultural_and_community_considerations from Gemini")

        # limitations_acknowledged
        if 'limitations_acknowledged' in gemini_ethics:
            if gemini_ethics['limitations_acknowledged'] != "SKIP":
                merged['ethics_and_reproducibility']['limitations_acknowledged'] = \
                    gemini_ethics['limitations_acknowledged']
                print("  ✓ Merged limitations_acknowledged from Gemini")

    # Add metadata about merge
    merged['_merge_info'] = {
        'baseline_model': 'Gemini-2.0-flash-exp',
        'targeted_model': 'Gemini-2.5-pro or compatible targeted model',
        'merged_fields': [
            'research_context.heritage_values_discussed',
            'evaluation.qualitative_results',
            'evaluation.error_analysis',
            'findings_and_contributions.* (ALL)',
            'ethics_and_reproducibility.privacy_and_data_protection',
            'ethics_and_reproducibility.bias_fairness_and_representativeness',
            'ethics_and_reproducibility.cultural_and_community_considerations',
            'ethics_and_reproducibility.limitations_acknowledged'
        ]
    }

    return merged


def process_directory(flash_dir, gemini_dir, output_dir):
    """Process all files in directories"""

    flash_path = Path(flash_dir)
    gemini_path = Path(gemini_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all Flash files
    flash_files = list(flash_path.glob("*.txt"))

    print(f"\nFound {len(flash_files)} Flash-2.0 files")
    print(f"Processing...\n")

    success_count = 0
    error_count = 0

    for flash_file in flash_files:
        filename = flash_file.name

        # Find corresponding Gemini file
        gemini_file = gemini_path / filename.replace('.txt', '_GEMINI_PRO3.txt')
        if not gemini_file.exists():
            gemini_file = gemini_path / filename  # Try without suffix

        print(f"Processing: {filename}")

        # Load both files
        flash_data = load_json_safe(flash_file)
        gemini_data = load_json_safe(gemini_file) if gemini_file.exists() else None

        if not flash_data:
            print(f"  ✗ ERROR: Could not load Flash file")
            error_count += 1
            continue

        if not gemini_data:
            print(f"  ⚠ WARNING: No Gemini file found, using Flash only")

        # Merge
        merged_data = merge_results(flash_data, gemini_data)

        # Save merged result
        output_file = output_path / filename.replace('.txt', '_MERGED.txt')
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Saved to: {output_file.name}\n")
            success_count += 1
        except Exception as e:
            print(f"  ✗ ERROR saving: {e}\n")
            error_count += 1

    print(f"\n{'=' * 60}")
    print(f"SUMMARY:")
    print(f"  Success: {success_count}")
    print(f"  Errors:  {error_count}")
    print(f"  Total:   {len(flash_files)}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Merge baseline Flash and targeted higher-capability extraction results'
    )
    parser.add_argument(
        '--flash',
        required=True,
        help='Directory containing baseline Flash JSON files'
    )
    parser.add_argument(
        '--gemini',
        required=True,
        help='Directory containing targeted JSON files'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Directory for merged output files'
    )

    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"BASELINE + TARGETED EXTRACTION MERGE TOOL")
    print(f"{'=' * 60}")
    print(f"Flash directory:  {args.flash}")
    print(f"Gemini directory: {args.gemini}")
    print(f"Output directory: {args.output}")

    process_directory(args.flash, args.gemini, args.output)


if __name__ == '__main__':
    main()
