#!/usr/bin/env python3
"""
Geocoding Results Importer and Retry Tool

This tool:
1. Imports your existing geocoding results into the cache
2. Identifies places that failed (NOT FOUND, TIMEOUT, ERROR)
3. Retries only the failed places
4. Merges results into a complete output file
"""

import csv
import json
import os
import sys
from collections import defaultdict


def import_existing_results(csv_file):
    """
    Import existing geocoding results from CSV into cache format.
    Returns: (cache_dict, failed_places_list)
    """
    print(f"📂 Reading existing results from {csv_file}...")

    cache = {}
    failed_places = []
    stats = defaultdict(int)

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            place_name = row.get('place_name', '')
            country = row.get('country', '')

            # Determine status
            if country.split(':')[0] in ['NOT FOUND', 'TIMEOUT', 'ERROR', '']:
                status = 'failed'
                failed_places.append(place_name)
                stats['failed'] += 1
            else:
                status = 'success'
                stats['success'] += 1

            # Create cache entry
            cache_entry = {
                'place_name': place_name,
                'city': row.get('city', ''),
                'state_province': row.get('state_province', row.get('state', '')),
                'country': country,
                'hierarchy': row.get('hierarchy', ''),
                'latitude': row.get('latitude', ''),
                'longitude': row.get('longitude', ''),
                'status': status
            }

            cache[place_name] = cache_entry
            stats['total'] += 1

    print(f"  ✓ Imported {stats['total']} places")
    print(f"  ✓ {stats['success']} successfully geocoded")
    print(f"  ✗ {stats['failed']} failed (will retry these)")

    return cache, failed_places


def save_cache(cache, cache_file='geocoding_cache.json'):
    """Save cache to JSON file."""
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"  💾 Cache saved to {cache_file}")


def create_retry_list(failed_places, output_file='places_to_retry.txt'):
    """Create a text file with failed places for retry."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for place in failed_places:
            f.write(f"{place}\n")
    print(f"  📝 Failed places list saved to {output_file}")
    return output_file


def merge_results(original_csv, new_csv, output_csv):
    """
    Merge original results with newly geocoded results.
    New results override old ones for the same place.
    """
    print(f"\n🔄 Merging results...")

    # Load original results
    results = {}
    with open(original_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            place_name = row.get('place_name', '')
            results[place_name] = row

    original_count = len(results)

    # Override with new results
    if os.path.exists(new_csv):
        with open(new_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            updated_count = 0
            for row in reader:
                place_name = row.get('place_name', '')
                # Only override if the new result is better
                old_country = results.get(place_name, {}).get('country', '')
                new_country = row.get('country', '')

                if new_country and new_country not in ['NOT FOUND', 'TIMEOUT', 'ERROR']:
                    results[place_name] = row
                    if old_country in ['NOT FOUND', 'TIMEOUT', 'ERROR', '']:
                        updated_count += 1

        print(f"  ✓ Updated {updated_count} places with new results")

    # Save merged results
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['place_name', 'city', 'state_province', 'country',
                      'hierarchy', 'latitude', 'longitude']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for place_name in sorted(results.keys()):
            writer.writerow(results[place_name])

    print(f"  ✓ Merged results saved to {output_csv}")

    # Print final stats
    success_count = sum(1 for r in results.values()
                        if r.get('country') and r['country'] not in ['NOT FOUND', 'TIMEOUT', 'ERROR'])
    failed_count = len(results) - success_count

    print(f"\n📊 Final Statistics:")
    print(f"  Total places: {len(results)}")
    print(f"  ✓ Successfully geocoded: {success_count} ({success_count / len(results) * 100:.1f}%)")
    print(f"  ✗ Still failed: {failed_count} ({failed_count / len(results) * 100:.1f}%)")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Import existing geocoding results and retry failed places'
    )
    parser.add_argument('existing_csv',
                        help='Your existing CSV from simple_geo_mapper.py')
    parser.add_argument('--retry-now', action='store_true',
                        help='Automatically retry failed places now (requires robust_geo_mapper.py)')
    parser.add_argument('--timeout', type=int, default=15,
                        help='Timeout for retry attempts (default: 15 seconds)')

    args = parser.parse_args()

    print("=" * 70)
    print("🔄 GEOCODING RESULTS IMPORTER & RETRY TOOL")
    print("=" * 70)

    if not os.path.exists(args.existing_csv):
        print(f"\n❌ ERROR: File '{args.existing_csv}' not found!")
        sys.exit(1)

    # Step 1: Import existing results
    print("\n📥 STEP 1: Importing existing results...")
    cache, failed_places = import_existing_results(args.existing_csv)

    # Step 2: Save to cache
    print("\n💾 STEP 2: Creating cache file...")
    save_cache(cache)

    # Step 3: Create retry list
    if failed_places:
        print(f"\n📝 STEP 3: Creating retry list...")
        retry_file = create_retry_list(failed_places)

        print("\n" + "=" * 70)
        print("✅ IMPORT COMPLETE!")
        print("=" * 70)

        print(f"\n📋 Next Steps:")
        print(f"\n1️⃣  Retry the {len(failed_places)} failed places:")
        print(f"   python robust_geo_mapper.py {retry_file} places_retry_results.csv --timeout {args.timeout}")

        print(f"\n2️⃣  Merge the retry results with your original file:")
        print(f"   python merge_geocoding_results.py {args.existing_csv} places_retry_results.csv places_complete.csv")

        print(f"\n💡 OR run with --retry-now to do both automatically:")
        print(f"   python {sys.argv[0]} {args.existing_csv} --retry-now")

        # Auto retry if requested
        if args.retry_now:
            print("\n" + "=" * 70)
            print("🔄 AUTO-RETRY MODE: Starting retry now...")
            print("=" * 70)

            # Check if robust_geo_mapper.py exists
            if not os.path.exists('robust_geo_mapper.py'):
                print("\n❌ ERROR: robust_geo_mapper.py not found in current directory!")
                print("Please ensure robust_geo_mapper.py is in the same folder.")
                sys.exit(1)

            # Run retry
            import subprocess
            retry_output = 'places_retry_results.csv'
            cmd = ['python', 'robust_geo_mapper.py', retry_file, retry_output,
                   '--timeout', str(args.timeout)]

            print(f"\n🚀 Running: {' '.join(cmd)}\n")
            result = subprocess.run(cmd)

            if result.returncode == 0:
                # Merge results
                print("\n" + "=" * 70)
                print("🔄 Merging results...")
                print("=" * 70)
                final_output = 'places_with_geo_complete.csv'
                merge_results(args.existing_csv, retry_output, final_output)

                print("\n" + "=" * 70)
                print("✅ ALL DONE!")
                print("=" * 70)
                print(f"\n📁 Final complete results: {final_output}")
            else:
                print("\n⚠️  Retry failed. You can run it manually:")
                print(f"   python robust_geo_mapper.py {retry_file} places_retry_results.csv")
    else:
        print("\n" + "=" * 70)
        print("✅ ALL PLACES ALREADY GEOCODED!")
        print("=" * 70)
        print("\nNo failed places found. Your geocoding is complete!")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()