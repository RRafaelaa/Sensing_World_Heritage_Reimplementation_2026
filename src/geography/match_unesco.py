#!/usr/bin/env python3
"""
Match places from CSV against UNESCO World Heritage name dictionary.

Usage:
    python match_unesco.py places_complete.csv unesco_name_dictionary.json output.csv

Requirements:
    - places_complete.csv: Your input CSV with place_name column
    - unesco_name_dictionary.json: The UNESCO dictionary file
    - output.csv: Where to save the results
"""

import json
import csv
import re
import sys
from typing import Dict, List, Tuple


def load_unesco_dictionary(json_path: str) -> Dict[str, List[str]]:
    """Load the UNESCO name dictionary."""
    print(f"Loading UNESCO dictionary from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  → Loaded {len(data)} UNESCO World Heritage sites")
    return data


def normalize_name(name: str) -> str:
    """
    Normalize a name for matching.
    - Convert to lowercase
    - Remove special punctuation
    - Normalize whitespace
    - Preserve word boundaries
    """
    name = name.lower()
    # Remove quotes and apostrophes
    name = re.sub(r'[''`"]', '', name)
    # Remove other special characters but keep hyphens, spaces, and commas
    name = re.sub(r'[^\w\s\-,]', ' ', name)
    # Normalize whitespace
    name = ' '.join(name.split())
    return name.strip()


def is_exact_match(place_name: str, unesco_name: str) -> bool:
    """Check if place_name exactly matches unesco_name after normalization."""
    return normalize_name(place_name) == normalize_name(unesco_name)


def find_unesco_matches(place_name: str, unesco_dict: Dict[str, List[str]]) -> Tuple[str, List[str], bool]:
    """
    Find matches between a place name and UNESCO dictionary.

    Returns:
        (original_unesco_name, list_of_alternative_names, is_exact_match)
        Returns (None, [], False) if no match found
    """
    if not place_name or not place_name.strip():
        return (None, [], False)

    normalized_place = normalize_name(place_name)

    # Strategy 1: Exact matches (most reliable)
    for unesco_name, alternatives in unesco_dict.items():
        # Check official UNESCO name
        if normalized_place == normalize_name(unesco_name):
            return (unesco_name, alternatives, True)

        # Check all alternatives
        for alt in alternatives:
            if normalized_place == normalize_name(alt):
                return (unesco_name, alternatives, True)

    # Strategy 2: Check if place name is a complete word/phrase within UNESCO name
    # This requires word boundaries to avoid false positives like "Albania" matching "Ani"
    if len(place_name) > 10:
        place_words = normalized_place.split()

        for unesco_name, alternatives in unesco_dict.items():
            normalized_unesco = normalize_name(unesco_name)
            unesco_words = normalized_unesco.split()

            # Check if all words from place appear consecutively in UNESCO name
            # This prevents "southeastern Albania" from matching "Ani"
            if len(place_words) <= len(unesco_words):
                # Look for consecutive word sequences
                for i in range(len(unesco_words) - len(place_words) + 1):
                    if unesco_words[i:i + len(place_words)] == place_words:
                        return (unesco_name, alternatives, False)

            # Also check if UNESCO name words appear consecutively in place name
            if len(unesco_words) <= len(place_words):
                for i in range(len(place_words) - len(unesco_words) + 1):
                    if place_words[i:i + len(unesco_words)] == unesco_words:
                        return (unesco_name, alternatives, False)

            # Check alternatives with same logic
            for alt in alternatives:
                normalized_alt = normalize_name(alt)
                alt_words = normalized_alt.split()

                if len(place_words) <= len(alt_words):
                    for i in range(len(alt_words) - len(place_words) + 1):
                        if alt_words[i:i + len(place_words)] == place_words:
                            return (unesco_name, alternatives, False)

                if len(alt_words) <= len(place_words):
                    for i in range(len(place_words) - len(alt_words) + 1):
                        if place_words[i:i + len(alt_words)] == alt_words:
                            return (unesco_name, alternatives, False)

    return (None, [], False)


def main():
    # Check command line arguments
    if len(sys.argv) != 4:
        print("Usage: python match_unesco.py <input_csv> <unesco_json> <output_csv>")
        print("\nExample:")
        print("  python match_unesco.py places_complete.csv unesco_name_dictionary.json output.csv")
        sys.exit(1)

    input_csv = sys.argv[1]
    unesco_json = sys.argv[2]
    output_csv = sys.argv[3]

    print("=" * 80)
    print("UNESCO PLACE MATCHING TOOL")
    print("=" * 80)

    # Load UNESCO dictionary
    try:
        unesco_dict = load_unesco_dictionary(unesco_json)
    except FileNotFoundError:
        print(f"ERROR: Could not find UNESCO dictionary file: {unesco_json}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"ERROR: Invalid JSON in UNESCO dictionary file: {unesco_json}")
        sys.exit(1)

    # Read input CSV
    print(f"\nReading input CSV: {input_csv}")
    try:
        with open(input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            places = list(reader)
        print(f"  → Found {len(places)} places to match")
    except FileNotFoundError:
        print(f"ERROR: Could not find input CSV file: {input_csv}")
        sys.exit(1)

    # Check if place_name column exists
    if places and 'place_name' not in places[0]:
        print("ERROR: Input CSV must have a 'place_name' column")
        sys.exit(1)

    # Match places
    print("\nMatching places to UNESCO World Heritage sites...")
    results = []
    matches_found = 0
    exact_matches = 0

    for i, place in enumerate(places):
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i + 1}/{len(places)} places ({(i + 1) / len(places) * 100:.1f}%)")

        place_name = place.get('place_name', '')
        unesco_name, alternatives, is_exact = find_unesco_matches(place_name, unesco_dict)

        if unesco_name:
            matches_found += 1
            if is_exact:
                exact_matches += 1
            # Join alternatives with semicolon
            alternatives_str = '; '.join(alternatives) if alternatives else ''
        else:
            unesco_name = ''
            alternatives_str = ''
            is_exact = False

        # Keep all original columns and add UNESCO columns
        result = dict(place)
        result['unesco_original_name'] = unesco_name
        result['unesco_alternative_names'] = alternatives_str
        result['is_exact_match'] = 'TRUE' if is_exact else 'FALSE'
        results.append(result)

    # Write results
    print(f"\nWriting results to: {output_csv}")
    try:
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            if results:
                fieldnames = list(results[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
        print(f"  → Successfully wrote {len(results)} rows")
    except IOError as e:
        print(f"ERROR: Could not write output file: {e}")
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 80)
    print("MATCHING COMPLETE!")
    print("=" * 80)
    print(f"Total places processed: {len(places):,}")
    print(f"UNESCO matches found: {matches_found:,}")
    print(f"  - Exact matches: {exact_matches:,}")
    print(f"  - Partial matches: {matches_found - exact_matches:,}")
    print(f"Match rate: {matches_found / len(places) * 100:.2f}%")
    print(f"No match: {len(places) - matches_found:,}")

    # Show some examples
    print("\n" + "=" * 80)
    print("SAMPLE MATCHES (first 10):")
    print("=" * 80)
    sample_count = 0
    for result in results:
        if result['unesco_original_name'] and sample_count < 10:
            print(f"\n{sample_count + 1}. Place: {result['place_name']}")
            print(f"   UNESCO: {result['unesco_original_name']}")
            print(f"   Exact Match: {result['is_exact_match']}")
            if result['unesco_alternative_names']:
                alts = result['unesco_alternative_names'].split('; ')
                print(f"   Alternatives: {len(alts)} names")
                if len(alts) <= 3:
                    for alt in alts:
                        print(f"     - {alt}")
                else:
                    for alt in alts[:3]:
                        print(f"     - {alt}")
                    print(f"     ... and {len(alts) - 3} more")
            sample_count += 1

    if sample_count == 0:
        print("\nNo matches found. This could mean:")
        print("  - The place names in your CSV don't match UNESCO site names")
        print("  - Try checking the spelling or using official UNESCO names")

    print("\n" + "=" * 80)
    print(f"Results saved to: {output_csv}")
    print("=" * 80)


if __name__ == '__main__':
    main()