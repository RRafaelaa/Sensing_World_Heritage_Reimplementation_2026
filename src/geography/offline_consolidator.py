#!/usr/bin/env python3
"""
Offline Paper Geography Consolidator
Works ENTIRELY from existing CSV files - no internet connection needed!
Perfect when geocoding APIs are timing out or unavailable.
"""

import csv
import os
from typing import Dict, List, Set
from collections import defaultdict


class OfflinePaperGeographyConsolidator:
    def __init__(self):
        self.place_to_geo = {}  # {place_name: {city, country, ...}}
        self.paper_to_places = defaultdict(set)  # {paper: {place1, place2, ...}}
        self.stats = {
            'total_places': 0,
            'geocoded_places': 0,
            'places_from_coords': 0,
            'places_from_names': 0
        }

    def load_geocoded_data(self, geocoded_csv: str, source_type: str = 'auto') -> None:
        """
        Load geocoded data from CSV files.

        Source types:
        - 'auto': Auto-detect format
        - 'forward': From simple_geo_mapper.py (place_name → coordinates)
        - 'reverse': From reverse_geocoder.py (coordinates → place_name)
        """
        if not os.path.exists(geocoded_csv):
            print(f"  ⚠ {geocoded_csv} not found, skipping")
            return

        print(f"  Loading {geocoded_csv}...")

        count = 0
        with open(geocoded_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Try different column name variations
                place_name = (row.get('place_name') or
                              row.get('original_place_name') or
                              row.get('Place Name') or '')

                city = (row.get('city') or
                        row.get('nearest_city') or
                        row.get('City') or '')

                state = (row.get('state_province') or
                         row.get('state') or
                         row.get('State') or '')

                country = (row.get('country') or
                           row.get('Country') or '')

                lat = row.get('latitude', '')
                lon = row.get('longitude', '')

                # Skip if this is a failure entry (don't store failures)
                if country.split(':')[0] in ['NOT FOUND', 'TIMEOUT', 'ERROR', '']:
                    continue

                if place_name and country:
                    self.place_to_geo[place_name] = {
                        'city': city,
                        'state': state,
                        'country': country,
                        'latitude': lat,
                        'longitude': lon
                    }
                    count += 1

                    # Track source type
                    if lat and lon and not city:
                        self.stats['places_from_coords'] += 1
                    else:
                        self.stats['places_from_names'] += 1

        print(f"    ✓ Loaded {count} successfully geocoded places")
        if count == 0:
            print(f"    ⚠ No successful results found in this file")

    def load_place_summary(self, place_summary_csv: str) -> None:
        """
        Load place-to-papers mapping from place_summary.csv.
        This is the REQUIRED file that connects places to papers.
        """
        if not os.path.exists(place_summary_csv):
            print(f"\n❌ ERROR: {place_summary_csv} not found!")
            print("This file is REQUIRED. Please run place_extractor.py first:")
            print("  python place_extractor.py ./extractions/")
            return

        print(f"\n📥 Loading place-to-papers mapping from {place_summary_csv}...")

        with open(place_summary_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                place_name = row.get('place_name', '')
                papers_str = row.get('papers', '')
                has_coords = row.get('has_coordinates', 'no') == 'yes'
                latitude = row.get('latitude', '')
                longitude = row.get('longitude', '')

                if place_name and papers_str:
                    # Split papers by ' | '
                    papers = [p.strip() for p in papers_str.split('|')]

                    for paper in papers:
                        self.paper_to_places[paper].add(place_name)

                    self.stats['total_places'] += 1

                    # Store coordinate data if available and not yet geocoded
                    if has_coords and latitude and longitude and place_name not in self.place_to_geo:
                        self.place_to_geo[place_name] = {
                            'city': '',
                            'state': '',
                            'country': '',
                            'latitude': latitude,
                            'longitude': longitude,
                            'has_coords_only': True
                        }

        self.stats['geocoded_places'] = len(self.place_to_geo)

        print(f"  ✓ Loaded {len(self.paper_to_places)} papers")
        print(f"  ✓ Loaded {self.stats['total_places']} unique places")
        print(f"  ✓ {self.stats['geocoded_places']} places have geocoding data")

    def load_paper_summary(self, paper_summary_csv: str) -> None:
        """
        Load place-to-papers mapping from paper_summary.csv.
        Alternative input format where papers are rows with places as columns.
        """
        if not os.path.exists(paper_summary_csv):
            print(f"\n❌ ERROR: {paper_summary_csv} not found!")
            return

        print(f"\n📥 Loading paper-to-places mapping from {paper_summary_csv}...")

        total_rows = 0
        papers_with_places = 0
        papers_without_places = 0
        skipped_papers = []

        with open(paper_summary_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                total_rows += 1
                paper = row.get('paper', '').strip()

                if not paper:
                    skipped_papers.append(f"Row {total_rows}: No paper identifier")
                    continue

                has_places = False

                # Get places from case_study_locations column
                case_study_str = row.get('case_study_locations', '')
                if case_study_str:
                    case_study_places = [p.strip() for p in case_study_str.split('|') if p.strip()]
                    for place in case_study_places:
                        self.paper_to_places[paper].add(place)
                        has_places = True

                # Get places from geographical_regions column
                geographical_str = row.get('geographical_regions', '')
                if geographical_str:
                    geographical_places = [p.strip() for p in geographical_str.split('|') if p.strip()]
                    for place in geographical_places:
                        self.paper_to_places[paper].add(place)
                        has_places = True

                if has_places:
                    papers_with_places += 1
                else:
                    papers_without_places += 1
                    # Still add the paper even if it has no places
                    if paper not in self.paper_to_places:
                        self.paper_to_places[paper] = set()

        # Count unique places
        all_places = set()
        for places in self.paper_to_places.values():
            all_places.update(places)

        print(f"  ✓ Total rows in CSV: {total_rows}")
        print(f"  ✓ Papers loaded: {len(self.paper_to_places)}")
        print(f"  ✓ Papers with places: {papers_with_places}")
        print(f"  ✓ Papers without places: {papers_without_places}")
        print(f"  ✓ Found {len(all_places)} unique places")

        if skipped_papers:
            print(f"\n⚠️  Skipped {len(skipped_papers)} rows:")
            for skip in skipped_papers[:5]:  # Show first 5
                print(f"     {skip}")
            if len(skipped_papers) > 5:
                print(f"     ... and {len(skipped_papers) - 5} more")

        if total_rows != len(self.paper_to_places):
            missing = total_rows - len(self.paper_to_places)
            print(f"\n⚠️  WARNING: {missing} papers not loaded!")
            print(f"     This could be due to:")
            print(f"     - Empty paper identifier column")
            print(f"     - Duplicate paper names (only last one kept)")
            print(f"     - CSV formatting issues")

        self.stats['total_places'] = len(all_places)

    def format_location(self, place_name: str) -> str:
        """
        Format a place as 'City (Country)' using cached geocoding data.
        Returns empty string if no data available.
        """
        geo_data = self.place_to_geo.get(place_name)

        if not geo_data:
            return ""

        # Check if this place only has coordinates without reverse geocoding
        if geo_data.get('has_coords_only'):
            return ""

        city = geo_data.get('city', '')
        state = geo_data.get('state', '')
        country = geo_data.get('country', '')

        if not country:
            return ""

        if city:
            return f"{city} ({country})"
        elif state:
            return f"{state} ({country})"
        else:
            # Don't return standalone country names
            return ""

    def consolidate_paper_locations(self, paper: str) -> Dict:
        """
        Consolidate all locations for a single paper.
        Returns dict with consolidated strings and statistics.
        """
        places = self.paper_to_places.get(paper, set())

        if not places:
            return {
                'num_places': 0,
                'num_geocoded': 0,
                'num_cities': 0,
                'num_countries': 0,
                'consolidated_locations': '',
                'countries_only': '',
                'cities_only': '',
                'unmapped_places': ''
            }

        locations = []
        countries = set()
        cities = set()
        unmapped = []
        geocoded_count = 0

        for place in places:
            formatted = self.format_location(place)

            if formatted:
                geocoded_count += 1
                if formatted not in locations:  # Avoid duplicates
                    locations.append(formatted)

                # Extract city and country
                geo_data = self.place_to_geo.get(place, {})
                country = geo_data.get('country', '')
                # Filter out error messages from countries
                if country and country not in ['NOT FOUND', 'TIMEOUT', 'ERROR', '']:
                    countries.add(country)
                city = geo_data.get('city', '')
                if city:
                    cities.add(city)
            else:
                unmapped.append(place)

        return {
            'num_places': len(places),
            'num_geocoded': geocoded_count,
            'num_cities': len(cities),
            'num_countries': len(countries),
            'consolidated_locations': "; ".join(sorted(locations)),
            'countries_only': "; ".join(sorted(countries)),
            'cities_only': "; ".join(sorted(cities)),
            'unmapped_places': " | ".join(sorted(unmapped))
        }

    def generate_consolidated_csv(self, output_file: str) -> None:
        """Generate the final consolidated CSV."""
        print(f"\n🔄 Generating consolidated output...")

        results = []

        for paper in sorted(self.paper_to_places.keys()):
            consolidated = self.consolidate_paper_locations(paper)

            result = {
                'paper': paper,
                'num_places': consolidated['num_places'],
                'num_geocoded': consolidated['num_geocoded'],
                'num_cities': consolidated['num_cities'],
                'num_countries': consolidated['num_countries'],
                'consolidated_locations': consolidated['consolidated_locations'],
                'countries_only': consolidated['countries_only'],
                'cities_only': consolidated['cities_only'],
                'unmapped_places': consolidated['unmapped_places']
            }

            results.append(result)

        # Save to CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['paper', 'num_places', 'num_geocoded', 'num_cities', 'num_countries',
                          'consolidated_locations', 'countries_only', 'cities_only',
                          'unmapped_places']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"✅ Consolidated output saved to: {output_file}")

        # Print summary
        self.print_summary(results)

    def print_summary(self, results: list) -> None:
        """Print summary statistics."""
        print("\n" + "=" * 70)
        print("📊 CONSOLIDATION SUMMARY")
        print("=" * 70)

        total_papers = len(results)
        papers_with_locations = len([r for r in results if r['consolidated_locations']])
        total_unmapped = sum(len(r['unmapped_places'].split('|')) if r['unmapped_places'] else 0
                             for r in results)

        total_countries = set()
        total_cities = set()

        for result in results:
            if result['countries_only']:
                total_countries.update(result['countries_only'].split('; '))
            if result['cities_only']:
                total_cities.update(result['cities_only'].split('; '))

        print(f"\n📄 Total papers: {total_papers}")
        print(f"✓ Papers with mapped locations: {papers_with_locations}")
        print(f"⚠ Total unmapped places: {total_unmapped}")
        print(f"\n🌍 Total unique countries: {len(total_countries)}")
        print(f"🏙️  Total unique cities: {len(total_cities)}")

        geocoding_rate = (self.stats['geocoded_places'] / self.stats['total_places'] * 100
                          if self.stats['total_places'] > 0 else 0)
        print(
            f"\n📍 Geocoding coverage: {self.stats['geocoded_places']}/{self.stats['total_places']} ({geocoding_rate:.1f}%)")

        if total_unmapped > 0:
            print(f"\n⚠️  WARNING: {total_unmapped} places could not be mapped")
            print("   To improve coverage, run these geocoding commands:")
            print("   → python simple_geo_mapper.py all_places.txt places_with_geo.csv")
            print("   → python reverse_geocoder.py places_with_coordinates.csv coords_geocoded.csv")
            print("   Then re-run this script with the new CSV files")

        # Country distribution
        country_counts = defaultdict(int)
        for result in results:
            if result['countries_only']:
                for country in result['countries_only'].split('; '):
                    if country:
                        country_counts[country] += 1

        if country_counts:
            print(f"\n🏆 Top 10 countries across all papers:")
            for i, (country, count) in enumerate(sorted(country_counts.items(),
                                                        key=lambda x: x[1], reverse=True)[:10], 1):
                print(f"   {i:2d}. {country}: {count} paper(s)")

        # Sample results
        if results:
            print(f"\n📋 Sample consolidated locations (first 3 papers):")
            for i, result in enumerate(results[:3], 1):
                paper_display = result['paper'][:55] + "..." if len(result['paper']) > 55 else result['paper']
                locations = result['consolidated_locations'][:70] + "..." if len(
                    result['consolidated_locations']) > 70 else result['consolidated_locations']
                print(f"\n   {i}. {paper_display}")
                if locations:
                    print(f"      → {locations}")
                else:
                    print(f"      → (no mapped locations - see unmapped_places column)")

        print("\n" + "=" * 70)


def main():
    import sys

    print("=" * 70)
    print("🔌 OFFLINE PAPER GEOGRAPHY CONSOLIDATOR")
    print("Works entirely from existing CSV files - no internet required!")
    print("=" * 70)

    if len(sys.argv) < 2:
        print("\n📖 Usage: python offline_consolidator.py <summary.csv> [geocoded1.csv] [geocoded2.csv] ...")
        print("\n✅ Required (either one):")
        print("   place_summary.csv - From place_extractor.py (places → papers)")
        print("   paper_summary.csv - From place_extractor.py (papers → places)")
        print("\n📍 Optional (provide all available geocoded files):")
        print("   places_with_geo.csv - From simple_geo_mapper.py")
        print("   coords_geocoded.csv - From reverse_geocoder.py")
        print("\n💡 Example:")
        print("   python offline_consolidator.py place_summary.csv places_with_geo.csv coords_geocoded.csv")
        print("   python offline_consolidator.py paper_summary.csv places_with_geo.csv")
        print("\n📤 Output: paper_geography_consolidated.csv")
        print("=" * 70)
        sys.exit(1)

    # Initialize consolidator
    consolidator = OfflinePaperGeographyConsolidator()

    # First argument is the summary file
    summary_file = sys.argv[1]

    if not os.path.exists(summary_file):
        print(f"\n❌ ERROR: {summary_file} not found!")
        sys.exit(1)

    # Auto-detect file type by checking columns
    with open(summary_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        first_row = next(reader, None)
        if first_row is None:
            print(f"\n❌ ERROR: {summary_file} is empty!")
            sys.exit(1)

        columns = set(first_row.keys())

    # Determine file type
    if 'papers' in columns and 'place_name' in columns:
        # place_summary.csv format (places → papers)
        print(f"\n📋 Detected: place_summary.csv format")
        consolidator.load_place_summary(summary_file)
    elif 'case_study_locations' in columns or 'geographical_regions' in columns:
        # paper_summary.csv format (papers → places)
        print(f"\n📋 Detected: paper_summary.csv format")
        consolidator.load_paper_summary(summary_file)
    else:
        print(f"\n❌ ERROR: Cannot determine file type!")
        print(f"Expected columns:")
        print(f"  - For place_summary.csv: 'place_name', 'papers'")
        print(f"  - For paper_summary.csv: 'paper', 'case_study_locations', 'geographical_regions'")
        print(f"\nFound columns: {', '.join(columns)}")
        sys.exit(1)

    if not consolidator.paper_to_places:
        print("\n❌ No data loaded. Exiting.")
        sys.exit(1)

    # Load all geocoded data files provided
    if len(sys.argv) > 2:
        print(f"\n📥 Loading geocoded data files...")
        for geocoded_file in sys.argv[2:]:
            consolidator.load_geocoded_data(geocoded_file)
    else:
        print("\n⚠️  WARNING: No geocoded CSV files provided!")
        print("   Output will only include places with embedded coordinates from summary file")
        print("\n💡 To get better geocoding coverage, run:")
        print("   1. python simple_geo_mapper.py all_places.txt places_with_geo.csv")
        print("   2. python reverse_geocoder.py places_with_coordinates.csv coords_geocoded.csv")
        print("   3. Re-run this script with the new CSV files")

    # Generate consolidated output
    output_file = 'paper_geography_consolidated.csv'
    consolidator.generate_consolidated_csv(output_file)

    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Output saved to: {output_file}")
    print("\n📋 Columns in output:")
    print("   • paper - Paper identifier")
    print("   • consolidated_locations - 'Beijing (China); Tokyo (Japan)' format")
    print("   • countries_only - List of countries")
    print("   • cities_only - List of cities")
    print("   • unmapped_places - Places without geocoding data")
    print("=" * 70)


if __name__ == '__main__':
    main()