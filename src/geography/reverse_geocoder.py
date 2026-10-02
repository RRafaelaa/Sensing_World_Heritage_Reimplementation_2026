#!/usr/bin/env python3
"""
Reverse Geocoder
Converts coordinates (latitude, longitude) to place names, cities, and countries.
"""

import csv
import time
from typing import Dict, Optional, Tuple
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError


class ReverseGeocoder:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="heritage_reverse_geocoder")
        self.cache = {}

    def reverse_geocode(self, lat: float, lon: float, max_retries: int = 3) -> Optional[Dict]:
        """
        Reverse geocode coordinates to get location information.
        """
        cache_key = f"{lat:.6f},{lon:.6f}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        for attempt in range(max_retries):
            try:
                time.sleep(1)  # Rate limiting
                location = self.geolocator.reverse(f"{lat}, {lon}", language='en', exactly_one=True)

                if location:
                    address = location.raw.get('address', {})

                    result = {
                        'display_name': location.address,
                        'latitude': lat,
                        'longitude': lon,
                        'city': address.get('city', address.get('town', address.get('village', ''))),
                        'county': address.get('county', ''),
                        'state': address.get('state', address.get('province', address.get('region', ''))),
                        'country': address.get('country', ''),
                        'country_code': address.get('country_code', '').upper(),
                        'place_type': location.raw.get('type', ''),
                        'category': location.raw.get('class', '')
                    }

                    self.cache[cache_key] = result
                    return result
                else:
                    self.cache[cache_key] = None
                    return None

            except (GeocoderTimedOut, GeocoderServiceError) as e:
                if attempt == max_retries - 1:
                    print(f"Failed to reverse geocode ({lat}, {lon}) after {max_retries} attempts")
                    self.cache[cache_key] = None
                    return None
                time.sleep(2 ** attempt)

        return None

    def get_hierarchy(self, geo_data: Dict) -> str:
        """Create a hierarchical location string."""
        parts = []

        if geo_data.get('city'):
            parts.append(geo_data['city'])
        elif geo_data.get('county'):
            parts.append(geo_data['county'])

        if geo_data.get('state'):
            parts.append(geo_data['state'])

        if geo_data.get('country'):
            parts.append(geo_data['country'])

        return ' → '.join(parts) if parts else geo_data.get('display_name', '')

    def process_coordinates_csv(self, input_file: str, output_file: str) -> None:
        """
        Process a CSV file with coordinates and add location information.
        Expected columns: place_name, latitude, longitude, [other columns...]
        """
        print(f"Reading coordinates from {input_file}...\n")

        rows = []
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            print("No data found in input file")
            return

        print(f"Processing {len(rows)} locations...\n")

        results = []
        for i, row in enumerate(rows, 1):
            try:
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                place_name = row.get('place_name', '')

                print(f"[{i}/{len(rows)}] {place_name or f'({lat:.4f}, {lon:.4f})'} ... ", end="")

                geo_data = self.reverse_geocode(lat, lon)

                if geo_data:
                    hierarchy = self.get_hierarchy(geo_data)

                    result = {
                        'original_place_name': place_name,
                        'latitude': lat,
                        'longitude': lon,
                        'nearest_city': geo_data['city'],
                        'county': geo_data['county'],
                        'state_province': geo_data['state'],
                        'country': geo_data['country'],
                        'country_code': geo_data['country_code'],
                        'hierarchy': hierarchy,
                        'place_type': geo_data['place_type'],
                        'full_address': geo_data['display_name']
                    }

                    # Add any additional columns from original CSV
                    for key, value in row.items():
                        if key not in ['place_name', 'latitude', 'longitude']:
                            result[key] = value

                    print(f"✓ {geo_data['country']}")
                else:
                    result = {
                        'original_place_name': place_name,
                        'latitude': lat,
                        'longitude': lon,
                        'nearest_city': '',
                        'county': '',
                        'state_province': '',
                        'country': 'NOT FOUND',
                        'country_code': '',
                        'hierarchy': '',
                        'place_type': '',
                        'full_address': ''
                    }

                    # Add any additional columns
                    for key, value in row.items():
                        if key not in ['place_name', 'latitude', 'longitude']:
                            result[key] = value

                    print("✗ Not found")

                results.append(result)

            except (ValueError, KeyError) as e:
                print(f"✗ Error: {e}")
                continue

        # Save results
        if results:
            print(f"\nSaving results to {output_file}...")

            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

            print(f"✓ Results saved to {output_file}")

            # Print summary
            self.print_summary(results)

    def print_summary(self, results: list) -> None:
        """Print summary statistics."""
        print("\n" + "=" * 70)
        print("REVERSE GEOCODING SUMMARY")
        print("=" * 70)

        total = len(results)
        successful = len([r for r in results if r['country'] and r['country'] != 'NOT FOUND'])

        countries = set(r['country'] for r in results if r['country'] and r['country'] != 'NOT FOUND')

        print(f"\nTotal locations processed: {total}")
        print(f"Successfully reverse geocoded: {successful}")
        print(f"Countries identified: {len(countries)}")

        if countries:
            print(f"\nCountries found:")
            for country in sorted(countries):
                count = len([r for r in results if r['country'] == country])
                print(f"  - {country}: {count} location(s)")

        print("\n" + "=" * 70)


def main():
    import sys

    if len(sys.argv) < 3:
        print("Usage: python reverse_geocoder.py <input_csv> <output_csv>")
        print("\nInput CSV must have columns: place_name, latitude, longitude")
        print("Additional columns will be preserved in the output.")
        print("\nExample:")
        print("  python reverse_geocoder.py places_with_coords.csv places_geocoded.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    geocoder = ReverseGeocoder()
    geocoder.process_coordinates_csv(input_file, output_file)


if __name__ == '__main__':
    main()