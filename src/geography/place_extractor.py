#!/usr/bin/env python3
"""
Place Name Extractor and Summarizer
Extracts place names from JSON extraction files and creates a summary CSV
that tracks which papers mention which places.
Includes detection and parsing of geo-coordinates.
"""

import json
import csv
import os
import re
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict


class PlaceNameExtractor:
    def __init__(self):
        self.place_to_papers = defaultdict(set)  # {place_name: {paper1, paper2, ...}}
        self.paper_to_places = defaultdict(dict)  # {paper: {case_study: [...], geographical: [...]}}
        self.place_coordinates = {}  # {place_name: (lat, lon)}
        self.all_papers = []

    def parse_coordinates(self, text: str) -> Optional[Tuple[float, float]]:
        """
        Parse coordinates from various formats:
        - 40.7128°N, 74.0060°W
        - 40.7128, -74.0060
        - (40.7128, -74.0060)
        - Lat: 40.7128, Lon: -74.0060
        - 40°42'46"N 74°00'21"W
        """
        if not text or not isinstance(text, str):
            return None

        # Pattern 1: Decimal degrees with optional N/S/E/W
        # Examples: "40.7128°N, 74.0060°W" or "40.7128, -74.0060"
        pattern1 = r'(-?\d+\.?\d*)[°\s]*([NS])?\s*,?\s*(-?\d+\.?\d*)[°\s]*([EW])?'
        match = re.search(pattern1, text, re.IGNORECASE)
        if match:
            lat = float(match.group(1))
            lat_dir = match.group(2)
            lon = float(match.group(3))
            lon_dir = match.group(4)

            # Apply direction
            if lat_dir and lat_dir.upper() == 'S':
                lat = -abs(lat)
            if lon_dir and lon_dir.upper() == 'W':
                lon = -abs(lon)

            # Validate ranges
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return (lat, lon)

        # Pattern 2: Labeled format
        # Examples: "Lat: 40.7128, Lon: -74.0060" or "Latitude: 40.7128, Longitude: -74.0060"
        pattern2 = r'lat(?:itude)?[:\s]+(-?\d+\.?\d*)[°\s]*([NS])?\s*,?\s*lon(?:gitude)?[:\s]+(-?\d+\.?\d*)[°\s]*([EW])?'
        match = re.search(pattern2, text, re.IGNORECASE)
        if match:
            lat = float(match.group(1))
            lat_dir = match.group(2)
            lon = float(match.group(3))
            lon_dir = match.group(4)

            if lat_dir and lat_dir.upper() == 'S':
                lat = -abs(lat)
            if lon_dir and lon_dir.upper() == 'W':
                lon = -abs(lon)

            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return (lat, lon)

        # Pattern 3: Degrees, Minutes, Seconds (DMS)
        # Example: "40°42'46"N 74°00'21"W"
        pattern3 = r'(\d+)[°\s]+(\d+)[\'′\s]+(\d+(?:\.\d+)?)[\"″\s]*([NS])\s+(\d+)[°\s]+(\d+)[\'′\s]+(\d+(?:\.\d+)?)[\"″\s]*([EW])'
        match = re.search(pattern3, text, re.IGNORECASE)
        if match:
            lat_d, lat_m, lat_s, lat_dir = float(match.group(1)), float(match.group(2)), float(
                match.group(3)), match.group(4)
            lon_d, lon_m, lon_s, lon_dir = float(match.group(5)), float(match.group(6)), float(
                match.group(7)), match.group(8)

            # Convert to decimal degrees
            lat = lat_d + lat_m / 60 + lat_s / 3600
            lon = lon_d + lon_m / 60 + lon_s / 3600

            if lat_dir.upper() == 'S':
                lat = -lat
            if lon_dir.upper() == 'W':
                lon = -lon

            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return (lat, lon)

        return None

    def detect_coordinate_in_place_name(self, place_name: str) -> Tuple[str, Optional[Tuple[float, float]]]:
        """
        Detect if a place name contains coordinates and separate them.
        Returns: (cleaned_place_name, coordinates_tuple or None)
        """
        coords = self.parse_coordinates(place_name)

        if coords:
            # Try to extract a cleaner place name by removing coordinate patterns
            cleaned = re.sub(r'[-\d\.°′″\'\",NSEW\s]+$', '', place_name).strip()
            cleaned = re.sub(r'\([^)]*[-\d\.°′″\'\",NSEW\s]+[^)]*\)', '', cleaned).strip()

            # If nothing left, use "Coordinates: lat, lon"
            if not cleaned or len(cleaned) < 3:
                cleaned = f"Coordinates: {coords[0]:.4f}, {coords[1]:.4f}"

            return (cleaned, coords)

        return (place_name, None)

    def extract_places_from_json(self, json_file: str) -> Dict[str, List[str]]:
        """Extract place names from a single JSON file."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Get paper identifier (title or filename)
            paper_id = data.get('metadata', {}).get('title', '')
            if not paper_id:
                paper_id = data.get('metadata', {}).get('pdf_filename', os.path.basename(json_file))

            # Extract place names from both fields
            research_context = data.get('research_context', {})
            case_study_places = research_context.get('case_study_locations', [])
            geographical_places = research_context.get('geographical_regions', [])

            return {
                'paper_id': paper_id,
                'case_study_locations': case_study_places,
                'geographical_regions': geographical_places
            }

        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            return None

    def process_directory(self, directory: str, pattern: str = "*.txt") -> None:
        """Process all JSON/TXT files in a directory."""
        path = Path(directory)
        files = list(path.glob(pattern))

        if not files:
            print(f"No files matching '{pattern}' found in {directory}")
            return

        print(f"Found {len(files)} files to process\n")

        for i, file in enumerate(files, 1):
            print(f"[{i}/{len(files)}] Processing: {file.name}")

            places_data = self.extract_places_from_json(str(file))

            if places_data:
                paper_id = places_data['paper_id']
                self.all_papers.append(paper_id)

                # Process places and detect coordinates
                processed_case_study = []
                processed_geographical = []

                # Process case study locations
                for place in places_data['case_study_locations']:
                    if place:
                        cleaned_place, coords = self.detect_coordinate_in_place_name(place)
                        processed_case_study.append(cleaned_place)

                        if coords:
                            self.place_coordinates[cleaned_place] = coords

                        self.place_to_papers[cleaned_place].add(paper_id)

                # Process geographical regions
                for place in places_data['geographical_regions']:
                    if place:
                        cleaned_place, coords = self.detect_coordinate_in_place_name(place)
                        processed_geographical.append(cleaned_place)

                        if coords:
                            self.place_coordinates[cleaned_place] = coords

                        self.place_to_papers[cleaned_place].add(paper_id)

                # Store paper -> places mapping
                self.paper_to_places[paper_id] = {
                    'case_study_locations': processed_case_study,
                    'geographical_regions': processed_geographical
                }

        print(f"\n✓ Processed {len(self.all_papers)} papers")
        print(f"✓ Found {len(self.place_to_papers)} unique places")
        print(f"✓ Detected coordinates for {len(self.place_coordinates)} places")

    def process_file_list(self, file_list: List[str]) -> None:
        """Process a specific list of JSON files."""
        print(f"Processing {len(file_list)} files\n")

        for i, file in enumerate(file_list, 1):
            print(f"[{i}/{len(file_list)}] Processing: {os.path.basename(file)}")

            places_data = self.extract_places_from_json(file)

            if places_data:
                paper_id = places_data['paper_id']
                self.all_papers.append(paper_id)

                # Process places and detect coordinates
                processed_case_study = []
                processed_geographical = []

                # Process case study locations
                for place in places_data['case_study_locations']:
                    if place:
                        cleaned_place, coords = self.detect_coordinate_in_place_name(place)
                        processed_case_study.append(cleaned_place)

                        if coords:
                            self.place_coordinates[cleaned_place] = coords

                        self.place_to_papers[cleaned_place].add(paper_id)

                # Process geographical regions
                for place in places_data['geographical_regions']:
                    if place:
                        cleaned_place, coords = self.detect_coordinate_in_place_name(place)
                        processed_geographical.append(cleaned_place)

                        if coords:
                            self.place_coordinates[cleaned_place] = coords

                        self.place_to_papers[cleaned_place].add(paper_id)

                self.paper_to_places[paper_id] = {
                    'case_study_locations': processed_case_study,
                    'geographical_regions': processed_geographical
                }

        print(f"\n✓ Processed {len(self.all_papers)} papers")
        print(f"✓ Found {len(self.place_to_papers)} unique places")
        print(f"✓ Detected coordinates for {len(self.place_coordinates)} places")

    def save_place_summary(self, output_file: str) -> None:
        """Save a summary CSV with places and which papers mention them."""
        rows = []

        for place, papers in sorted(self.place_to_papers.items()):
            # Determine which field(s) the place appears in
            field_types = set()
            for paper in papers:
                if place in self.paper_to_places[paper]['case_study_locations']:
                    field_types.add('case_study')
                if place in self.paper_to_places[paper]['geographical_regions']:
                    field_types.add('geographical')

            # Get coordinates if available
            coords = self.place_coordinates.get(place)
            lat_str = f"{coords[0]:.6f}" if coords else ""
            lon_str = f"{coords[1]:.6f}" if coords else ""

            rows.append({
                'place_name': place,
                'num_papers': len(papers),
                'field_type': ', '.join(sorted(field_types)),
                'has_coordinates': 'yes' if coords else 'no',
                'latitude': lat_str,
                'longitude': lon_str,
                'papers': ' | '.join(sorted(papers))
            })

        # Sort by number of papers (descending), then alphabetically
        rows.sort(key=lambda x: (-x['num_papers'], x['place_name']))

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['place_name', 'num_papers', 'field_type', 'has_coordinates',
                          'latitude', 'longitude', 'papers']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n✓ Place summary saved to: {output_file}")

    def save_paper_summary(self, output_file: str) -> None:
        """Save a summary CSV showing all places per paper."""
        rows = []

        for paper in sorted(self.all_papers):
            case_study = self.paper_to_places[paper]['case_study_locations']
            geographical = self.paper_to_places[paper]['geographical_regions']

            all_places = set(case_study + geographical)

            rows.append({
                'paper': paper,
                'num_case_study_places': len(case_study),
                'num_geographical_places': len(geographical),
                'total_unique_places': len(all_places),
                'case_study_locations': ' | '.join(case_study) if case_study else '',
                'geographical_regions': ' | '.join(geographical) if geographical else ''
            })

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['paper', 'num_case_study_places', 'num_geographical_places',
                          'total_unique_places', 'case_study_locations', 'geographical_regions']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"✓ Paper summary saved to: {output_file}")

    def save_all_places_list(self, output_file: str) -> None:
        """Save a simple text file with all unique places (for geo mapping)."""
        all_places = sorted(self.place_to_papers.keys())

        with open(output_file, 'w', encoding='utf-8') as f:
            for place in all_places:
                f.write(f"{place}\n")

        print(f"✓ All places list saved to: {output_file}")
        print(f"  (This file can be used with simple_geo_mapper.py)")

    def save_places_with_coordinates(self, output_file: str) -> None:
        """Save places that have coordinates with their lat/lon."""
        if not self.place_coordinates:
            print("✓ No places with coordinates found to save")
            return

        rows = []
        for place in sorted(self.place_coordinates.keys()):
            coords = self.place_coordinates[place]
            papers = self.place_to_papers[place]

            rows.append({
                'place_name': place,
                'latitude': coords[0],
                'longitude': coords[1],
                'num_papers': len(papers),
                'papers': ' | '.join(sorted(papers))
            })

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['place_name', 'latitude', 'longitude', 'num_papers', 'papers']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"✓ Places with coordinates saved to: {output_file}")
        print(f"  ({len(rows)} places with embedded coordinates)")

    def save_detailed_mapping(self, output_file: str) -> None:
        """Save a detailed CSV showing place -> paper -> field type."""
        rows = []

        for place in sorted(self.place_to_papers.keys()):
            for paper in sorted(self.place_to_papers[place]):
                in_case_study = place in self.paper_to_places[paper]['case_study_locations']
                in_geographical = place in self.paper_to_places[paper]['geographical_regions']

                field_type = []
                if in_case_study:
                    field_type.append('case_study_locations')
                if in_geographical:
                    field_type.append('geographical_regions')

                rows.append({
                    'place_name': place,
                    'paper': paper,
                    'field_type': ', '.join(field_type)
                })

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['place_name', 'paper', 'field_type']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"✓ Detailed mapping saved to: {output_file}")

    def print_statistics(self) -> None:
        """Print summary statistics."""
        print("\n" + "=" * 70)
        print("PLACE NAME EXTRACTION SUMMARY")
        print("=" * 70)

        print(f"\nTotal papers processed: {len(self.all_papers)}")
        print(f"Total unique places: {len(self.place_to_papers)}")
        print(f"Places with embedded coordinates: {len(self.place_coordinates)}")

        # Count places by field type
        case_study_only = 0
        geographical_only = 0
        both_fields = 0

        for place in self.place_to_papers.keys():
            in_case_study = False
            in_geographical = False

            for paper in self.place_to_papers[place]:
                if place in self.paper_to_places[paper]['case_study_locations']:
                    in_case_study = True
                if place in self.paper_to_places[paper]['geographical_regions']:
                    in_geographical = True

            if in_case_study and in_geographical:
                both_fields += 1
            elif in_case_study:
                case_study_only += 1
            elif in_geographical:
                geographical_only += 1

        print(f"\nPlaces by field type:")
        print(f"  - Only in case_study_locations: {case_study_only}")
        print(f"  - Only in geographical_regions: {geographical_only}")
        print(f"  - In both fields: {both_fields}")

        # Coordinate statistics
        if self.place_coordinates:
            print(f"\nCoordinate formats detected:")
            print(f"  - Total places with coordinates: {len(self.place_coordinates)}")

            # Show some examples
            print(f"\nExample coordinates found:")
            for i, (place, coords) in enumerate(list(self.place_coordinates.items())[:3], 1):
                print(f"  {i}. {place}: ({coords[0]:.6f}, {coords[1]:.6f})")

        # Most frequently mentioned places
        top_places = sorted(self.place_to_papers.items(),
                            key=lambda x: len(x[1]), reverse=True)[:10]

        if top_places:
            print(f"\nTop 10 most frequently mentioned places:")
            for i, (place, papers) in enumerate(top_places, 1):
                coord_marker = " [has coords]" if place in self.place_coordinates else ""
                print(f"  {i}. {place} ({len(papers)} papers){coord_marker}")

        # Papers with most places
        paper_place_counts = [(p, len(set(self.paper_to_places[p]['case_study_locations'] +
                                          self.paper_to_places[p]['geographical_regions'])))
                              for p in self.all_papers]
        paper_place_counts.sort(key=lambda x: x[1], reverse=True)

        if paper_place_counts:
            print(f"\nTop 5 papers with most places mentioned:")
            for i, (paper, count) in enumerate(paper_place_counts[:5], 1):
                # Truncate long paper titles
                paper_display = paper[:60] + "..." if len(paper) > 60 else paper
                print(f"  {i}. {paper_display} ({count} places)")

        print("\n" + "=" * 70)


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python place_extractor.py <directory_or_file> [file2] [file3] ...")
        print("\nExamples:")
        print("  # Process all .txt files in a directory:")
        print("  python place_extractor.py ./extractions/")
        print()
        print("  # Process specific files:")
        print("  python place_extractor.py paper1.txt paper2.txt paper3.txt")
        print()
        print("Outputs:")
        print("  - place_summary.csv: Places with papers that mention them")
        print("  - paper_summary.csv: Papers with all their places")
        print("  - all_places.txt: Simple list for geo mapping")
        print("  - detailed_mapping.csv: Complete place-paper-field mapping")
        sys.exit(1)

    extractor = PlaceNameExtractor()

    # Check if first argument is a directory or file
    first_arg = sys.argv[1]

    if os.path.isdir(first_arg):
        # Process directory
        print(f"Processing directory: {first_arg}\n")
        extractor.process_directory(first_arg)
    else:
        # Process file list
        files = sys.argv[1:]
        print(f"Processing {len(files)} files\n")
        extractor.process_file_list(files)

    # Save all output formats
    extractor.save_place_summary('place_summary.csv')
    extractor.save_paper_summary('paper_summary.csv')
    extractor.save_all_places_list('all_places.txt')
    extractor.save_places_with_coordinates('places_with_coordinates.csv')
    extractor.save_detailed_mapping('detailed_mapping.csv')

    # Print statistics
    extractor.print_statistics()

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\nFor places WITHOUT coordinates:")
    print("  python simple_geo_mapper.py all_places.txt places_with_geo.csv")
    print("\nFor places WITH embedded coordinates:")
    print("  python reverse_geocoder.py places_with_coordinates.csv coords_geocoded.csv")
    print("\nOr to process individual papers with full JSON integration:")
    print("  python geo_identifier.py <input.txt> <output.txt>")
    print("=" * 70)


if __name__ == '__main__':
    main()