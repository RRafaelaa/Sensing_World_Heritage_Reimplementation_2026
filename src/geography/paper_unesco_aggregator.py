#!/usr/bin/env python3
"""
Paper-Level UNESCO Status Aggregator
Aggregates UNESCO and Tentative List matches from place-level to paper-level.
Works with outputs from match_unesco.py and offline_consolidator.py

Usage:
    python paper_unesco_aggregator.py place_summary.csv places_unesco.csv places_tentative.csv output.csv
"""

import csv
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def load_place_unesco_status(
        unesco_csv: str,
        tentative_csv: str
) -> Dict[str, Tuple[str, str, str]]:
    """
    Load UNESCO status for each place from the matched CSV files.

    Returns:
        {place_name: (status, match_type, matched_site_name)}
        where status is 'UNESCO', 'Tentative', or 'Not Listed'
    """
    place_status = {}

    print(f"\n📥 Loading UNESCO World Heritage matches from: {unesco_csv}")
    unesco_count = 0
    with open(unesco_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            place_name = row.get('place_name', '').strip()
            unesco_site = row.get('unesco_original_name', '').strip()
            is_exact = row.get('is_exact_match', 'FALSE').strip().upper() == 'TRUE'
            same_words = place_name == unesco_site

            if place_name and unesco_site:
                match_type = 'Exact' if same_words else ('Partial' if is_exact  else 'Alternative')
                place_status[place_name] = ('UNESCO', match_type, unesco_site)
                unesco_count += 1

    print(f"  -> Found {unesco_count} UNESCO World Heritage matches")

    print(f"\n📥 Loading Tentative List matches from: {tentative_csv}")
    tentative_count = 0
    with open(tentative_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            place_name = row.get('place_name', '').strip()
            tentative_site = row.get('unesco_original_name', '').strip()
            is_exact = row.get('is_exact_match', 'FALSE').strip().upper() == 'TRUE'
            same_words = place_name == tentative_site

            # Only add if not already matched to UNESCO (UNESCO takes priority)
            if place_name and tentative_site and place_name not in place_status:
                match_type = 'Exact' if same_words else ('Partial' if is_exact  else 'Alternative')
                place_status[place_name] = ('Tentative', match_type, tentative_site)
                tentative_count += 1

    print(f"  -> Found {tentative_count} Tentative List matches")
    print(f"  -> Total unique places with UNESCO/Tentative status: {len(place_status)}")

    return place_status


def detect_file_format(summary_csv: str) -> str:
    """
    Detect whether the file is place_summary.csv or paper_summary.csv format.

    Returns:
        'place_summary' or 'paper_summary'
    """
    with open(summary_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        first_row = next(reader, None)
        if not first_row:
            raise ValueError(f"File {summary_csv} is empty")

        columns = set(first_row.keys())

    # place_summary.csv has: place_name, papers
    # paper_summary.csv has: paper, case_study_locations, geographical_regions

    if 'place_name' in columns and 'papers' in columns:
        return 'place_summary'
    elif 'paper' in columns and ('case_study_locations' in columns or 'geographical_regions' in columns):
        return 'paper_summary'
    else:
        raise ValueError(f"Cannot determine file format. Found columns: {', '.join(columns)}")


def load_place_to_papers_from_place_summary(place_summary_csv: str) -> Dict[str, List[str]]:
    """
    Load mapping of places to papers from place_summary.csv format.
    Format: place_name, papers (papers separated by |)

    Returns:
        {place_name: [paper1, paper2, ...]}
    """
    print(f"  Format: place_summary.csv (places → papers)")

    place_to_papers = defaultdict(list)

    with open(place_summary_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            place_name = row.get('place_name', '').strip()
            papers_str = row.get('papers', '').strip()

            if place_name and papers_str:
                # Split papers by '|'
                papers = [p.strip() for p in papers_str.split('|') if p.strip()]
                place_to_papers[place_name] = papers

    return dict(place_to_papers)


def load_place_to_papers_from_paper_summary(paper_summary_csv: str) -> Dict[str, List[str]]:
    """
    Load mapping of places to papers from paper_summary.csv format.
    Format: paper, case_study_locations, geographical_regions (places separated by |)

    Returns:
        {place_name: [paper1, paper2, ...]}
    """
    print(f"  Format: paper_summary.csv (papers → places)")

    place_to_papers = defaultdict(list)

    with open(paper_summary_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            paper = row.get('paper', '').strip()

            if not paper:
                continue

            # Get places from case_study_locations column
            case_study_str = row.get('case_study_locations', '')
            if case_study_str:
                case_study_places = [p.strip() for p in case_study_str.split('|') if p.strip()]
                for place in case_study_places:
                    if place not in place_to_papers:
                        place_to_papers[place] = []
                    if paper not in place_to_papers[place]:
                        place_to_papers[place].append(paper)

            # Get places from geographical_regions column
            geographical_str = row.get('geographical_regions', '')
            if geographical_str:
                geographical_places = [p.strip() for p in geographical_str.split('|') if p.strip()]
                for place in geographical_places:
                    if place not in place_to_papers:
                        place_to_papers[place] = []
                    if paper not in place_to_papers[place]:
                        place_to_papers[place].append(paper)

    return dict(place_to_papers)

def load_paper_summary(paper_summary_csv: str) -> None:
        """
        Load place-to-papers mapping from paper_summary.csv.
        Alternative input format where papers are rows with places as columns.
        """
        print(f"\n📥 Loading paper-to-places mapping from {paper_summary_csv}...")

        total_rows = 0
        papers_with_places = 0
        papers_without_places = 0
        skipped_papers = []
        paper_to_places = defaultdict(set)

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
                        paper_to_places[paper].add(place)
                        has_places = True

                # Get places from geographical_regions column
                geographical_str = row.get('geographical_regions', '')
                if geographical_str:
                    geographical_places = [p.strip() for p in geographical_str.split('|') if p.strip()]
                    for place in geographical_places:
                        paper_to_places[paper].add(place)
                        has_places = True

                if has_places:
                    papers_with_places += 1
                else:
                    papers_without_places += 1
                    # Still add the paper even if it has no places
                    if paper not in paper_to_places:
                        paper_to_places[paper] = set()

        # Count unique places
        all_places = set()
        for places in paper_to_places.values():
            all_places.update(places)

        print(f"  ✓ Total rows in CSV: {total_rows}")
        print(f"  ✓ Papers loaded: {len(paper_to_places)}")
        print(f"  ✓ Papers with places: {papers_with_places}")
        print(f"  ✓ Papers without places: {papers_without_places}")
        print(f"  ✓ Found {len(all_places)} unique places")

        if skipped_papers:
            print(f"\n⚠️  Skipped {len(skipped_papers)} rows:")
            for skip in skipped_papers[:5]:  # Show first 5
                print(f"     {skip}")
            if len(skipped_papers) > 5:
                print(f"     ... and {len(skipped_papers) - 5} more")

        if total_rows != len(paper_to_places):
            missing = total_rows - len(paper_to_places)
            print(f"\n⚠️  WARNING: {missing} papers not loaded!")
            print(f"     This could be due to:")
            print(f"     - Empty paper identifier column")
            print(f"     - Duplicate paper names (only last one kept)")
            print(f"     - CSV formatting issues")

        return dict(paper_to_places)

def load_place_to_papers(summary_csv: str) -> Dict[str, List[str]]:
    """
    Load mapping of places to papers from either place_summary.csv or paper_summary.csv.
    Auto-detects the format.

    Returns:
        {place_name: [paper1, paper2, ...]}
    """
    print(f"\n📥 Loading place-to-papers mapping from: {summary_csv}")

    # Detect format
    file_format = detect_file_format(summary_csv)

    # Load based on format
    if file_format == 'place_summary':
        place_to_papers = load_place_to_papers_from_place_summary(summary_csv)
    else:  # paper_summary
        place_to_papers = load_place_to_papers_from_paper_summary(summary_csv)

    print(f"  -> Loaded {len(place_to_papers)} places")
    total_papers = len(set(p for papers in place_to_papers.values() for p in papers))
    print(f"  -> Linked to {total_papers} unique papers")

    return place_to_papers

def aggregate_paper_unesco_status(
        place_to_papers: Dict[str, List[str]],
        place_status: Dict[str, Tuple[str, str, str]]
) -> Dict[str, Dict]:
    """
    Aggregate UNESCO status from places to papers.

    Returns:
        {paper: {
            'total_places': int,
            'unesco_exact': int,
            'unesco_partial': int,
            'tentative_exact': int,
            'tentative_partial': int,
            'not_listed': int,
            'unesco_sites': [list of site names],
            'tentative_sites': [list of site names],
            'unesco_places': [list of place names],
            'tentative_places': [list of place names]
        }}
    """
    print(f"\n🔄 Aggregating UNESCO status by paper...")

    paper_stats = defaultdict(lambda: {
        'total_places': 0,
        'unesco_exact': 0,
        'unesco_partial': 0,
        'unesco_alternative': 0,
        'tentative_exact': 0,
        'tentative_partial': 0,
        'tentative_alternative': 0,
        'not_listed': 0,
        'unesco_sites': set(),
        'tentative_sites': set(),
        'unesco_places': [],
        'tentative_places': []
    })

    # For each place, update all papers it appears in
    for place_name, papers in place_to_papers.items():
        status_info = place_status.get(place_name)

        for paper in papers:
            paper_stats[paper]['total_places'] += 1

            if status_info:
                status, match_type, site_name = status_info

                if status == 'UNESCO':
                    if match_type == 'Exact':
                        paper_stats[paper]['unesco_exact'] += 1
                    elif match_type == 'Partial':
                        paper_stats[paper]['unesco_partial'] += 1
                    else:
                        paper_stats[paper]['unesco_alternative'] += 1
                    paper_stats[paper]['unesco_sites'].add(site_name)
                    paper_stats[paper]['unesco_places'].append(place_name)

                elif status == 'Tentative':
                    if match_type == 'Exact':
                        paper_stats[paper]['tentative_exact'] += 1
                    elif match_type == 'Partial':
                        paper_stats[paper]['tentative_partial'] += 1
                    else:
                        paper_stats[paper]['tentative_alternative'] += 1
                    paper_stats[paper]['tentative_sites'].add(site_name)
                    paper_stats[paper]['tentative_places'].append(place_name)
            else:
                paper_stats[paper]['not_listed'] += 1

    print(f"  -> Aggregated status for {len(paper_stats)} papers")

    return dict(paper_stats)


def write_paper_unesco_csv(
        paper_stats: Dict[str, Dict],
        all_papers: List[str],
        output_csv: str
) -> None:
    """
    Write paper-level UNESCO statistics to CSV.
    """
    print(f"\n💾 Writing results to: {output_csv}")

    results = []
    for paper in all_papers:
        if paper in paper_stats:
            stats = paper_stats[paper]

            # Convert sets to sorted lists
            unesco_sites_list = sorted(stats['unesco_sites'])
            tentative_sites_list = sorted(stats['tentative_sites'])

            # Calculate totals
            total_unesco = stats['unesco_exact'] + stats['unesco_partial'] + stats['unesco_alternative']
            total_tentative = stats['tentative_exact'] + stats['tentative_partial'] + stats['tentative_alternative']
            total_listed = total_unesco + total_tentative

            # Determine overall status
            if total_unesco > 0 and total_tentative > 0:
                overall_status = 'UNESCO & Tentative'
            elif total_unesco > 0:
                overall_status = 'UNESCO'
            elif total_tentative > 0:
                overall_status = 'Tentative'
            else:
                overall_status = 'Not Listed'

            result = {
                'paper': paper,
                'total_places': stats['total_places'],
                'overall_status': overall_status,

                # UNESCO (inscribed)
                'unesco_total': total_unesco,
                'unesco_exact': stats['unesco_exact'],
                'unesco_partial': stats['unesco_partial'],
                'unesco_alternative': stats['unesco_alternative'],
                'unesco_sites_count': len(unesco_sites_list),
                'unesco_site_names': '; '.join(unesco_sites_list),
                'unesco_places': ' | '.join(stats['unesco_places'][:10]) +
                                 (f' ... (+{len(stats["unesco_places"]) - 10} more)' if len(
                                     stats['unesco_places']) > 10 else ''),

                # Tentative List
                'tentative_total': total_tentative,
                'tentative_exact': stats['tentative_exact'],
                'tentative_partial': stats['tentative_partial'],
                'tentative_alternative': stats['tentative_alternative'],
                'tentative_sites_count': len(tentative_sites_list),
                'tentative_site_names': '; '.join(tentative_sites_list),
                'tentative_places': ' | '.join(stats['tentative_places'][:10]) +
                                    (f' ... (+{len(stats["tentative_places"]) - 10} more)' if len(
                                        stats['tentative_places']) > 10 else ''),

                # Not listed
                'not_listed': stats['not_listed'],

                # Percentages
                'unesco_percentage': f"{(total_unesco / stats['total_places'] * 100):.1f}%" if stats[
                                                                                                   'total_places'] > 0 else '0.0%',
                'tentative_percentage': f"{(total_tentative / stats['total_places'] * 100):.1f}%" if stats[
                                                                                                         'total_places'] > 0 else '0.0%',
                'listed_percentage': f"{(total_listed / stats['total_places'] * 100):.1f}%" if stats[
                                                                                                   'total_places'] > 0 else '0.0%',
            }
        else:
            result = {
                'paper': paper}

        results.append(result)

    # Sort by total UNESCO + Tentative matches (descending)
    #results.sort(key=lambda x: (x['unesco_total'] + x['tentative_total'], x['total_places']), reverse=True)
    results.sort(key = lambda x: x['paper'])

    # Write to CSV
    fieldnames = [
        'paper', 'total_places', 'overall_status',
        'unesco_total', 'unesco_exact', 'unesco_partial', 'unesco_alternative', 'unesco_sites_count', 'unesco_percentage',
        'tentative_total', 'tentative_exact', 'tentative_partial', 'tentative_alternative', 'tentative_sites_count', 'tentative_percentage',
        'not_listed', 'listed_percentage',
        'unesco_site_names', 'tentative_site_names',
        'unesco_places', 'tentative_places'
    ]

    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"  -> Successfully wrote {len(results)} papers")

    return results


def print_summary(results: List[Dict]) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("📊 PAPER-LEVEL UNESCO STATUS SUMMARY")
    print("=" * 80)

    total_papers = len(results)
    papers_with_unesco = len([r for r in results if len(r)>2 and r['unesco_total'] > 0])
    papers_with_tentative = len([r for r in results if len(r)>2 and r['tentative_total'] > 0])
    papers_with_both = len([r for r in results if len(r)>2 and r['unesco_total'] > 0 and r['tentative_total'] > 0])
    papers_with_any = len([r for r in results if len(r)>2 and (r['unesco_total'] > 0 or r['tentative_total'] > 0)])
    papers_not_listed = total_papers - papers_with_any

    total_places = sum(r['total_places'] for r in results if len(r)>2)
    total_unesco_places = sum(r['unesco_total'] for r in results if len(r)>2)
    total_tentative_places = sum(r['tentative_total'] for r in results if len(r)>2)
    total_not_listed = sum(r['not_listed'] for r in results if len(r)>2)

    print(f"\n📄 Total papers: {total_papers:,}")
    print(f"\n🏛️  Papers with UNESCO World Heritage sites:")
    print(f"   - With UNESCO sites:        {papers_with_unesco:>6,} ({papers_with_unesco / total_papers * 100:.1f}%)")
    print(
        f"   - With Tentative sites:     {papers_with_tentative:>6,} ({papers_with_tentative / total_papers * 100:.1f}%)")
    print(f"   - With both:                {papers_with_both:>6,} ({papers_with_both / total_papers * 100:.1f}%)")
    print(f"   - With any UNESCO/Tentative:{papers_with_any:>6,} ({papers_with_any / total_papers * 100:.1f}%)")
    print(f"   - No UNESCO sites:          {papers_not_listed:>6,} ({papers_not_listed / total_papers * 100:.1f}%)")

    print(f"\n📍 Place-level statistics:")
    print(f"   - Total places:             {total_places:>6,}")
    print(f"   - UNESCO matches:           {total_unesco_places:>6,} ({total_unesco_places / total_places * 100:.1f}%)")
    print(
        f"   - Tentative matches:        {total_tentative_places:>6,} ({total_tentative_places / total_places * 100:.1f}%)")
    print(f"   - Not listed:               {total_not_listed:>6,} ({total_not_listed / total_places * 100:.1f}%)")

    # Count exact vs partial
    total_unesco_exact = sum(r['unesco_exact'] for r in results if len(r)>2)
    total_unesco_partial = sum(r['unesco_partial'] for r in results if len(r)>2)
    total_unesco_alternative = sum(r['unesco_alternative'] for r in results if len(r)>2)
    total_tentative_exact = sum(r['tentative_exact'] for r in results if len(r)>2)
    total_tentative_partial = sum(r['tentative_partial'] for r in results if len(r)>2)
    total_tentative_alternative = sum(r['tentative_alternative'] for r in results if len(r)>2)

    print(f"\n🎯 Match quality:")
    print(f"   UNESCO:")
    print(
        f"     - Exact matches:   {total_unesco_exact:>6,} ({total_unesco_exact / total_unesco_places * 100:.1f}% of UNESCO)")
    print(
        f"     - Partial matches: {total_unesco_partial:>6,} ({total_unesco_partial / total_unesco_places * 100:.1f}% of UNESCO)")
    print(
        f"     - Alternative matches: {total_unesco_alternative:>6,} ({total_unesco_alternative / total_unesco_places * 100:.1f}% of UNESCO)")
    print(f"   Tentative:")
    print(
        f"     - Exact matches:   {total_tentative_exact:>6,} ({total_tentative_exact / total_tentative_places * 100:.1f}% of Tentative)")
    print(
        f"     - Partial matches: {total_tentative_partial:>6,} ({total_tentative_partial / total_tentative_places * 100:.1f}% of Tentative)")
    print(
        f"     - Alternative matches: {total_tentative_alternative:>6,} ({total_tentative_alternative / total_tentative_places * 100:.1f}% of UNESCO)")

    # Top papers by UNESCO site count
    print(f"\n🏆 Top 10 papers by UNESCO World Heritage sites:")
    top_unesco = sorted([r for r in results if len(r)>2], key=lambda x: x['unesco_total'], reverse=True)[:10]
    for i, paper in enumerate(top_unesco, 1):
        if paper['unesco_total'] > 0:
            paper_name = paper['paper'][:50] + "..." if len(paper['paper']) > 50 else paper['paper']
            print(f"   {i:2d}. {paper_name}")
            print(
                f"       UNESCO: {paper['unesco_total']} places ({paper['unesco_exact']} exact, {paper['unesco_partial']} partial), {paper['unesco_alternative']} alternative)")
            print(f"       Sites: {paper['unesco_sites_count']} unique")

    # Top papers by Tentative site count
    print(f"\n📋 Top 10 papers by Tentative List sites:")
    top_tentative = sorted([r for r in results if len(r)>2], key=lambda x: x['tentative_total'], reverse=True)[:10]
    for i, paper in enumerate(top_tentative, 1):
        if paper['tentative_total'] > 0:
            paper_name = paper['paper'][:50] + "..." if len(paper['paper']) > 50 else paper['paper']
            print(f"   {i:2d}. {paper_name}")
            print(
                f"       Tentative: {paper['tentative_total']} places ({paper['tentative_exact']} exact, {paper['tentative_partial']} partial), {paper['tentative_alternative']} alternative)")
            print(f"       Sites: {paper['tentative_sites_count']} unique")

    print("\n" + "=" * 80)


def main():
    print("=" * 80)
    print("📊 PAPER-LEVEL UNESCO STATUS AGGREGATOR")
    print("Aggregates place-level UNESCO matches to paper-level statistics")
    print("=" * 80)

    if len(sys.argv) != 5:
        print("\n📖 Usage:")
        print(
            "   python paper_unesco_aggregator.py <place_summary.csv> <places_unesco.csv> <places_tentative.csv> <output.csv>")
        print("\n📄 Required files:")
        print("   place_summary.csv          - From place_extractor.py (place → papers mapping)")
        print("   places_unesco.csv          - From match_unesco.py with UNESCO dict (place-level matches)")
        print("   places_tentative.csv       - From match_unesco.py with Tentative dict (place-level matches)")
        print("   output.csv                 - Where to save paper-level results")
        print("\n💡 Example:")
        print("   python paper_unesco_aggregator.py \\")
        print("          place_summary.csv \\")
        print("          places_complete_with_unesco.csv \\")
        print("          places_complete_with_unesco_tentative.csv \\")
        print("          papers_unesco_status.csv")
        print("\n📤 Output columns:")
        print("   - paper, total_places, overall_status")
        print("   - unesco_total, unesco_exact, unesco_partial, unesco_sites_count, unesco_percentage")
        print("   - tentative_total, tentative_exact, tentative_partial, tentative_sites_count, tentative_percentage")
        print("   - not_listed, listed_percentage")
        print("   - unesco_site_names, tentative_site_names")
        print("   - unesco_places, tentative_places")
        print("=" * 80)
        sys.exit(1)

    place_summary_csv = sys.argv[1]
    places_unesco_csv = sys.argv[2]
    places_tentative_csv = sys.argv[3]
    output_csv = sys.argv[4]

    # Load data
    place_status = load_place_unesco_status(places_unesco_csv, places_tentative_csv)
    place_to_papers = load_place_to_papers(place_summary_csv)

    file_format = detect_file_format(place_summary_csv)

    # Aggregate to paper level
    paper_stats = aggregate_paper_unesco_status(place_to_papers, place_status)

    # Load based on format
    if file_format == 'place_summary':
        all_papers = list(paper_stats)
    else:  # paper_summary
        all_papers = list(load_paper_summary(place_summary_csv))

    # Write output
    results = write_paper_unesco_csv(paper_stats, all_papers, output_csv)

    # Print summary
    print_summary(results)

    print("\n" + "=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Output saved to: {output_csv}")
    print("\n📊 Each paper now has:")
    print("   • Count of UNESCO World Heritage places (exact & partial)")
    print("   • Count of Tentative List places (exact & partial)")
    print("   • List of unique UNESCO/Tentative sites mentioned")
    print("   • Percentage of places that are UNESCO/Tentative listed")
    print("=" * 80)


if __name__ == '__main__':
    main()