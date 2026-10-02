#!/usr/bin/env python3
"""
Validation script for heritage paper extractions
Checks completeness and generates statistics
"""

import json
import os
from pathlib import Path
from collections import Counter, defaultdict

def validate_json_structure(data, schema):
    """Check if JSON matches schema structure"""
    errors = []
    
    # Check top-level blocks
    required_blocks = list(schema.keys())
    for block in required_blocks:
        if block not in data:
            errors.append(f"Missing block: {block}")
    
    return errors

def count_filled_fields(data, prefix=""):
    """Recursively count filled vs empty fields"""
    filled = 0
    empty = 0
    
    if isinstance(data, dict):
        for key, value in data.items():
            field_name = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (dict, list)):
                f, e = count_filled_fields(value, field_name)
                filled += f
                empty += e
            elif value in ["", [], {}]:
                empty += 1
            else:
                filled += 1
    elif isinstance(data, list):
        if len(data) == 0:
            empty += 1
        else:
            filled += 1
    
    return filled, empty

def analyze_extraction(json_file):
    """Analyze a single extraction"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    filled, empty = count_filled_fields(data)
    total = filled + empty
    completeness = (filled / total * 100) if total > 0 else 0
    
    return {
        'filename': json_file.name,
        'filled_fields': filled,
        'empty_fields': empty,
        'total_fields': total,
        'completeness_pct': completeness,
        'title': data.get('metadata', {}).get('title', 'N/A'),
        'year': data.get('metadata', {}).get('year', 'N/A'),
        'platforms': data.get('social_media_data', {}).get('platforms_detected', []),
        'models': data.get('ai_methodology', {}).get('specific_models_mentioned', [])
    }

def generate_report(output_dir):
    """Generate validation report for all extractions"""
    output_path = Path(output_dir)
    json_files = list(output_path.glob('*.txt'))
    
    print("="*70)
    print(f"VALIDATION REPORT - Heritage Paper Extractions")
    print("="*70)
    print(f"\nTotal files: {len(json_files)}")
    
    if len(json_files) == 0:
        print("No JSON files found!")
        return
    
    # Analyze all files
    results = []
    errors = []
    
    for json_file in json_files:
        try:
            result = analyze_extraction(json_file)
            results.append(result)
        except Exception as e:
            errors.append(f"{json_file.name}: {str(e)}")
    
    print(f"Successfully analyzed: {len(results)}")
    print(f"Errors: {len(errors)}")
    
    if errors:
        print("\nError Details:")
        for error in errors[:10]:  # Show first 10
            print(f"  - {error}")
    
    # Calculate statistics
    completeness_scores = [r['completeness_pct'] for r in results]
    avg_completeness = sum(completeness_scores) / len(completeness_scores)
    
    print(f"\n{'='*70}")
    print("COMPLETENESS STATISTICS")
    print(f"{'='*70}")
    print(f"Average completeness: {avg_completeness:.1f}%")
    print(f"Highest completeness: {max(completeness_scores):.1f}%")
    print(f"Lowest completeness: {min(completeness_scores):.1f}%")
    
    # Distribution
    ranges = [(0, 25), (25, 50), (50, 75), (75, 100)]
    print("\nCompleteness Distribution:")
    for low, high in ranges:
        count = sum(1 for s in completeness_scores if low <= s < high)
        print(f"  {low}-{high}%: {count} papers")
    
    # Most complete papers
    results_sorted = sorted(results, key=lambda x: x['completeness_pct'], reverse=True)
    
    print(f"\n{'='*70}")
    print("TOP 10 MOST COMPLETE EXTRACTIONS")
    print(f"{'='*70}")
    for i, r in enumerate(results_sorted[:10], 1):
        print(f"{i}. {r['filename']}")
        print(f"   Completeness: {r['completeness_pct']:.1f}%")
        print(f"   Title: {r['title'][:60]}...")
        print(f"   Year: {r['year']}")
    
    # Field analysis
    print(f"\n{'='*70}")
    print("FIELD COVERAGE ANALYSIS")
    print(f"{'='*70}")
    
    # Count filled fields across papers
    all_years = [r['year'] for r in results if r['year']]
    all_platforms = []
    all_models = []
    
    for r in results:
        all_platforms.extend(r['platforms'])
        all_models.extend(r['models'])
    
    print(f"\nMetadata Coverage:")
    print(f"  Papers with year: {len(all_years)}/{len(results)} ({len(all_years)/len(results)*100:.1f}%)")
    print(f"  Papers with platforms: {sum(1 for r in results if r['platforms'])}/{len(results)}")
    print(f"  Papers with models: {sum(1 for r in results if r['models'])}/{len(results)}")
    
    # Most common values
    year_counts = Counter(all_years).most_common(10)
    platform_counts = Counter(all_platforms).most_common(10)
    model_counts = Counter(all_models).most_common(10)
    
    print(f"\n{'='*70}")
    print("MOST COMMON VALUES")
    print(f"{'='*70}")
    
    print("\nTop Years:")
    for year, count in year_counts:
        print(f"  {year}: {count} papers")
    
    print("\nTop Platforms:")
    for platform, count in platform_counts:
        print(f"  {platform}: {count} papers")
    
    print("\nTop Models:")
    for model, count in model_counts:
        print(f"  {model}: {count} papers")
    
    # Save detailed report to CSV
    import csv
    report_file = output_path / 'validation_report.csv'
    
    with open(report_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'title', 'year', 'completeness_pct', 
                     'filled_fields', 'empty_fields', 'platforms', 'models']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for r in results_sorted:
            writer.writerow({
                'filename': r['filename'],
                'title': r['title'],
                'year': r['year'],
                'completeness_pct': f"{r['completeness_pct']:.1f}",
                'filled_fields': r['filled_fields'],
                'empty_fields': r['empty_fields'],
                'platforms': ', '.join(r['platforms']),
                'models': ', '.join(r['models'])
            })
    
    print(f"\n{'='*70}")
    print(f"Detailed report saved to: {report_file}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate heritage paper extractions')
    parser.add_argument('--output', required=True, help='Output directory with JSON files')
    
    args = parser.parse_args()
    
    generate_report(args.output)
