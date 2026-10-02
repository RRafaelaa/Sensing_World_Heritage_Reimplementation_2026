#!/usr/bin/env python3
"""
Robust Simple Place Name to Country Mapper
Improved version with:
- Better timeout handling
- Progress saving (resume from failures)
- Configurable timeouts
- Better error reporting
"""

import csv
import json
import os
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time


class RobustGeoMapper:
    def __init__(self, timeout=10):
        """
        Initialize with configurable timeout.

        Args:
            timeout: Seconds to wait for each geocoding request (default: 10)
        """
        self.geolocator = Nominatim(user_agent="heritage_mapper", timeout=timeout)
        self.cache_file = "geocoding_cache.json"
        self.cache = self.load_cache()
        self.failed_places = []
        self.timeout = timeout

    def load_cache(self):
        """Load cached geocoding results to avoid re-querying."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                print(f"📂 Loaded {len(cache)} cached results from {self.cache_file}")
                return cache
            except:
                return {}
        return {}

    def save_cache(self):
        """Save cache to disk."""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def geocode_place(self, place, max_retries=3, retry_delay=2):
        """
        Geocode a place with retry logic and timeout handling.

        Args:
            place: Place name to geocode
            max_retries: Number of retry attempts
            retry_delay: Seconds to wait between retries
        """
        place = place.strip()
        if not place:
            return None

        # Check cache first
        if place in self.cache and self.cache[place]['status']=='success':
            return self.cache[place]

        for attempt in range(max_retries):
            try:
                # Rate limiting - wait between requests
                time.sleep(1)

                location = self.geolocator.geocode(
                    place,
                    addressdetails=True,
                    language='en',
                    timeout=self.timeout
                )

                if location:
                    address = location.raw.get('address', {})

                    city = address.get('city', address.get('town', address.get('village', '')))
                    state = address.get('state', address.get('province', address.get('region', '')))
                    country = address.get('country', '')

                    # Build hierarchy
                    hierarchy_parts = [p for p in [city, state, country] if p]
                    hierarchy = ' → '.join(hierarchy_parts)

                    result = {
                        'place_name': place,
                        'city': city,
                        'state_province': state,
                        'country': country,
                        'hierarchy': hierarchy,
                        'latitude': location.latitude,
                        'longitude': location.longitude,
                        'status': 'success'
                    }

                    # Cache the result
                    self.cache[place] = result
                    return result
                else:
                    # Not found
                    result = {
                        'place_name': place,
                        'city': '',
                        'state_province': '',
                        'country': 'NOT FOUND',
                        'hierarchy': '',
                        'latitude': '',
                        'longitude': '',
                        'status': 'not_found'
                    }
                    self.cache[place] = result
                    return result

            except GeocoderTimedOut:
                if attempt < max_retries - 1:
                    print(f"      ⏱️  Timeout, retrying ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    print(f"      ❌ Timeout after {max_retries} attempts")
                    result = {
                        'place_name': place,
                        'city': '',
                        'state_province': '',
                        'country': 'TIMEOUT',
                        'hierarchy': '',
                        'latitude': '',
                        'longitude': '',
                        'status': 'timeout'
                    }
                    self.failed_places.append(place)
                    return result

            except GeocoderServiceError as e:
                print(f"      ❌ Service error: {str(e)[:150]}")
                result = {
                    'place_name': place,
                    'city': '',
                    'state_province': '',
                    'country': 'ERROR',
                    'hierarchy': '',
                    'latitude': '',
                    'longitude': '',
                    'status': 'error',
                    'error': str(e)
                }
                self.failed_places.append(place)
                return result

            except Exception as e:
                print(f"      ❌ Unexpected error: {str(e)[:50]}")
                result = {
                    'place_name': place,
                    'city': '',
                    'state_province': '',
                    'country': 'ERROR',
                    'hierarchy': '',
                    'latitude': '',
                    'longitude': '',
                    'status': 'error',
                    'error': str(e)
                }
                self.failed_places.append(place)
                return result

        # Should not reach here, but just in case
        return None

    def process_places(self, places_file, output_file, save_progress_every=10):
        """
        Process places from input file with progress tracking.

        Args:
            places_file: Input text file with place names (one per line)
            output_file: Output CSV file
            save_progress_every: Save cache and partial results every N places
        """
        # Read all places
        with open(places_file, 'r', encoding='utf-8') as f:
            places = [line.strip() for line in f if line.strip()]

        print(f"📍 Loaded {len(places)} place names from {places_file}")

        # Check how many are already cached
        cached_count = sum(1 for p in places if p in self.cache)
        if cached_count > 0:
            print(f"✓ {cached_count} places already in cache (will skip)")

        print(f"\n🌍 Starting geocoding...")
        print("=" * 70)

        results = []

        for i, place in enumerate(places, 1):
            if not place:
                continue

            # Check if already in results (from cache)
            if place in self.cache:
                cached_result = self.cache[place]
                status = cached_result.get('status', 'success')

                if status == 'success':
                    print(f"[{i}/{len(places)}] {place[:40]:<40} ✓ (cached) {cached_result.get('country', '')}")
                    results.append(cached_result)
                else:
                    print(f"[{i}/{len(places)}] {place[:40]:<40} ⚠ (cached failure) Retrying...")
                    result = self.geocode_place(place)
                    if result:
                        status = result.get('status', 'success')
                        if status == 'success':
                            print(f"✓ {result.get('country', '')}")
                        elif status == 'not_found':
                            print(f"✗ Not found")
                        elif status == 'timeout':
                            print()  # Already printed in geocode_place
                        else:
                            print()  # Already printed in geocode_place

                        results.append(result)

            else:
                # Geocode new place
                print(f"[{i}/{len(places)}] {place[:40]:<40} ", end="")

                result = self.geocode_place(place)

                if result:
                    status = result.get('status', 'success')
                    if status == 'success':
                        print(f"✓ {result.get('country', '')}")
                    elif status == 'not_found':
                        print(f"✗ Not found")
                    elif status == 'timeout':
                        print()  # Already printed in geocode_place
                    else:
                        print()  # Already printed in geocode_place

                    results.append(result)

            # Save progress periodically
            if i % save_progress_every == 0:
                self.save_cache()
                self.save_partial_results(results, output_file)
                print(f"    💾 Progress saved ({i}/{len(places)} completed)")

        # Final save
        self.save_cache()
        self.save_results(results, output_file)

        print("\n" + "=" * 70)
        print("✅ COMPLETE!")
        print("=" * 70)

        # Print summary
        self.print_summary(results)

        # Save failed places for retry
        if self.failed_places:
            self.save_failed_places()

    def save_partial_results(self, results, output_file):
        """Save partial results (for resume capability)."""
        temp_file = output_file.replace('.csv', '_partial.csv')
        with open(temp_file, 'w', newline='', encoding='utf-8') as f:
            if results:
                # Use only standard fieldnames, ignore extra fields
                fieldnames = ['place_name', 'city', 'state_province', 'country',
                              'hierarchy', 'latitude', 'longitude']
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(results)

    def save_results(self, results, output_file):
        """Save final results."""
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            if results:
                # Use only standard fieldnames, ignore extra fields like 'error', 'status'
                fieldnames = ['place_name', 'city', 'state_province', 'country',
                              'hierarchy', 'latitude', 'longitude']
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(results)

        print(f"\n✓ Results saved to: {output_file}")

        # Remove partial file if exists
        temp_file = output_file.replace('.csv', '_partial.csv')
        if os.path.exists(temp_file):
            os.remove(temp_file)

    def save_failed_places(self):
        """Save list of places that failed to geocode."""
        failed_file = "failed_places.txt"
        with open(failed_file, 'w', encoding='utf-8') as f:
            for place in self.failed_places:
                f.write(f"{place}\n")

        print(f"\n⚠️  {len(self.failed_places)} places failed to geocode")
        print(f"   Failed places saved to: {failed_file}")
        print(f"   You can retry these later with better network or manually check them")

    def print_summary(self, results):
        """Print summary statistics."""
        total = len(results)
        successful = len([r for r in results if r.get('country').split(':')[0] not in ['NOT FOUND', 'TIMEOUT', 'ERROR', '']])
        not_found = len([r for r in results if r.get('country').split(':')[0] == 'NOT FOUND'])
        timeout = len([r for r in results if r.get('country').split(':')[0] == 'TIMEOUT'])
        error = len([r for r in results if r.get('country').split(':')[0] == 'ERROR'])

        countries = set(r['country'] for r in results
                        if r.get('country') and r['country'].split(':')[0] not in ['NOT FOUND', 'TIMEOUT', 'ERROR'])

        print(f"\n📊 SUMMARY")
        print("=" * 70)
        print(f"Total places processed: {total}")
        print(f"✓ Successfully geocoded: {successful} ({successful / total * 100:.1f}%)")
        print(f"✗ Not found: {not_found}")
        print(f"⏱️  Timeout: {timeout}")
        print(f"❌ Error: {error}")
        print(f"\n🌍 Countries identified: {len(countries)}")

        if countries:
            print(f"\nCountries found:")
            for country in sorted(countries):
                count = len([r for r in results if r.get('country') == country])
                print(f"  • {country}: {count} place(s)")

        print("=" * 70)


def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='Robust geocoding tool with timeout handling and resume capability'
    )
    parser.add_argument('input_file', help='Input text file with place names (one per line)')
    parser.add_argument('output_file', nargs='?', default='places_with_geo.csv',
                        help='Output CSV file (default: places_with_geo.csv)')
    parser.add_argument('--timeout', type=int, default=10,
                        help='Timeout in seconds for each request (default: 10)')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear existing cache and start fresh')

    args = parser.parse_args()

    print("=" * 70)
    print("🗺️  ROBUST PLACE NAME GEOCODER")
    print("=" * 70)

    if args.clear_cache and os.path.exists('geocoding_cache.json'):
        os.remove('geocoding_cache.json')
        print("🗑️  Cache cleared")

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"\n❌ ERROR: Input file '{args.input_file}' not found!")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} <input_file> [output_file] [--timeout SECONDS]")
        print("\nExample:")
        print(f"  python {sys.argv[0]} all_places.txt places_with_geo.csv --timeout 15")
        sys.exit(1)

    # Create mapper with specified timeout
    mapper = RobustGeoMapper(timeout=args.timeout)

    print(f"\n⚙️  Settings:")
    print(f"   • Timeout: {args.timeout} seconds per request")
    print(f"   • Rate limit: 1 request per second")
    print(f"   • Max retries: 3 attempts per place")
    print(f"   • Progress saved every 10 places")

    # Process places
    mapper.process_places(args.input_file, args.output_file)

    print("\n💡 Tips:")
    print("   • Results are cached in 'geocoding_cache.json'")
    print("   • Re-running this script will skip already geocoded places")
    print("   • If many timeouts occur, try increasing --timeout value")
    print("   • If API is down, retry later - your progress is saved!")
    print("=" * 70)


if __name__ == '__main__':
    main()