#!/usr/bin/env python3
"""
Merge Geocoding Results

Merges two geocoding CSV files, with the second file overriding the first
for any places that were successfully geocoded.
"""

import csv
import sys


def merge_geocoding_results(original_csv, new_csv, output_csv):
    """
    Merge original and new geocoding results.
    New results override old ones if they're successful.
    """
    print("=" * 70)
    print("🔄 MERGING GEOCODING RESULTS")
    print("=" * 70)

    print(f"\n📂 Reading original results from: {original_csv}")

    # Load original results
    results = {}
    with open(original_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            place_name = row.get('place_name', '')
            if place_name:
                # Store only standard fields
                results[place_name] = {
                    'place_name': place_name,
                    'city': row.get('city', ''),
                    'state_province': row.get('state_province', row.get('state', '')),
                    'country': row.get('country', ''),
                    'hierarchy': row.get('hierarchy', ''),
                    'latitude': row.get('latitude', ''),
                    'longitude': row.get('longitude', '')
                }

    original_count = len(results)
    original_success = sum(1 for r in results.values()
                           if r['country'] and r['country'] not in ['NOT FOUND', 'TIMEOUT', 'ERROR'])

    print(f"  ✓ Loaded {original_count} places")
    print(f"  ✓ {original_success} were successfully geocoded")

    # Load and merge new results
    print(f"\n📂 Reading new results from: {new_csv}")

    updated_count = 0
    new_success_count = 0

    with open(new_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            place_name = row.get('place_name', '')
            new_country = row.get('country', '')

            if not place_name:
                continue

            # Count successful new results
            if new_country and new_country not in ['NOT FOUND', 'TIMEOUT', 'ERROR']:
                new_success_count += 1

                # Check if this is an improvement over old result
                if place_name in results:
                    old_country = results[place_name]['country']
                    if old_country in ['NOT FOUND', 'TIMEOUT', 'ERROR', '']:
                        updated_count += 1

                # Update with new result (only standard fields)
                results[place_name] = {
                    'place_name': place_name,
                    'city': row.get('city', ''),
                    'state_province': row.get('state_province', row.get('state', '')),
                    'country': new_country,
                    'hierarchy': row.get('hierarchy', ''),
                    'latitude': row.get('latitude', ''),
                    'longitude': row.get('longitude', '')
                }

    print(f"  ✓ Found {new_success_count} successfully geocoded places")
    print(f"  ✓ Updated {updated_count} previously failed places")

    # Save merged results
    print(f"\n💾 Saving merged results to: {output_csv}")

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['place_name', 'city', 'state_province', 'country',
                      'hierarchy', 'latitude', 'longitude']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for place_name in sorted(results.keys()):
            writer.writerow(results[place_name])

    # Final statistics
    final_success = sum(1 for r in results.values()
                        if r['country'] and r['country'] not in ['NOT FOUND', 'TIMEOUT', 'ERROR'])
    final_failed = len(results) - final_success
    improvement = final_success - original_success

    print("\n" + "=" * 70)
    print("📊 MERGE SUMMARY")
    print("=" * 70)

    print(f"\nTotal places: {len(results)}")
    print(f"\nBefore merge:")
    print(f"  ✓ Success: {original_success}")
    print(f"  ✗ Failed: {original_count - original_success}")

    print(f"\nAfter merge:")
    print(f"  ✓ Success: {final_success} (+{improvement} improvement)")
    print(f"  ✗ Failed: {final_failed}")

    print(f"\nSuccess rate: {final_success / len(results) * 100:.1f}%")

    if final_failed > 0:
        print(f"\n⚠️  {final_failed} places still need geocoding")
        print("You can create a list of failed places and retry again:")
        print("  python import_and_retry.py places_with_geo_complete.csv --retry-now")
    else:
        print(f"\n✅ All places successfully geocoded!")

    print("\n" + "=" * 70)
    print(f"✅ Merged results saved to: {output_csv}")
    print("=" * 70)


def main():
    if len(sys.argv) < 4:
        print("Usage: python merge_geocoding_results.py <original.csv> <new.csv> <output.csv>")
        print("\nExample:")
        print("  python merge_geocoding_results.py places_with_geo.csv places_retry_results.csv places_complete.csv")
        sys.exit(1)

    original_csv = sys.argv[1]
    new_csv = sys.argv[2]
    output_csv = sys.argv[3]

    try:
        merge_geocoding_results(original_csv, new_csv, output_csv)
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()