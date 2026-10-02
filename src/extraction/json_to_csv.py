#!/usr/bin/env python3
"""
JSON to CSV Converter for Heritage Papers
Converts extracted JSON data into multiple CSV files for analysis

Creates multiple specialized CSV files:
1. papers_overview.csv - High-level metadata
2. heritage_analysis.csv - Heritage domain focus
3. social_media_analysis.csv - Social media data
4. ai_methodology.csv - AI methods and models
5. graph_analysis.csv - Graph/network data
6. master_dataset.csv - Combined key fields
7. master_dataset.xlsx - Excel file with multiple sheets

Usage:
    python json_to_csv_converter.py --input data/outputs --output data/csv
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

try:
    import pandas as pd
except ImportError:
    print("WARNING: pandas not installed. Excel export will be disabled.")
    print("Install with: pip install pandas openpyxl")
    pd = None


class JSONtoCSVConverter:
    """Convert JSON heritage paper data to CSV format"""

    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize converter

        Args:
            input_dir: Directory containing JSON files
            output_dir: Directory for output CSV files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for all paper data
        self.papers = []
        self.filenames = []  # Store actual JSON filenames

    def load_json_files(self):
        """Load all JSON files from input directory (supports .json and .txt extensions)"""
        print("=" * 70)
        print("Loading JSON files...")
        print("=" * 70)

        # Support both .json and .txt extensions
        json_files = list(self.input_dir.glob("*.json"))
        txt_files = list(self.input_dir.glob("*.txt"))

        all_files = json_files + txt_files

        if not all_files:
            print(f"ERROR: No JSON or TXT files found in {self.input_dir}")
            sys.exit(1)

        print(f"Found {len(json_files)} .json files")
        print(f"Found {len(txt_files)} .txt files")
        print(f"Total: {len(all_files)} files")

        loaded = 0
        failed = 0

        for file_path in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.papers.append(data)
                    self.filenames.append(file_path.name)  # Store actual filename
                    loaded += 1
            except json.JSONDecodeError as e:
                print(f"⚠️  Failed to parse JSON in {file_path.name}: {e}")
                failed += 1
            except Exception as e:
                print(f"⚠️  Failed to load {file_path.name}: {e}")
                failed += 1

        print(f"\n✅ Loaded: {loaded} papers")
        if failed > 0:
            print(f"❌ Failed: {failed} papers")
        print()

        return loaded

    def clean_text(self, text: str) -> str:
        """
        Clean text for CSV output by replacing newlines with spaces.
        This prevents embedded newlines from breaking CSV structure.

        Args:
            text: Input text that may contain newlines

        Returns:
            Cleaned text with newlines replaced by spaces
        """
        if not isinstance(text, str):
            return str(text) if text is not None else ""

        # Replace various newline characters with a single space
        text = text.replace('\r\n', ' ')  # Windows line endings
        text = text.replace('\n', ' ')  # Unix line endings
        text = text.replace('\r', ' ')  # Old Mac line endings

        # Replace multiple spaces with single space
        while '  ' in text:
            text = text.replace('  ', ' ')

        return text.strip()

    def flatten_array(self, arr: List, separator: str = "; ") -> str:
        """
        Convert array to semicolon-separated string with cleaned text

        Args:
            arr: List of items to flatten
            separator: String to use between items

        Returns:
            Flattened string with cleaned text
        """
        if not arr:
            return ""
        # Filter out empty strings and clean each item
        arr = [self.clean_text(str(x)) for x in arr if x]
        return separator.join(arr)

    def safe_get(self, data: Dict, *keys, default="") -> Any:
        """Safely get nested dictionary value"""
        result = data
        for key in keys:
            if isinstance(result, dict):
                result = result.get(key, default)
            else:
                return default
        return result if result is not None else default

    def generate_overview_csv(self):
        """Generate papers_overview.csv with high-level metadata"""
        print("Generating papers_overview.csv...")

        rows = []
        for paper, filename in zip(self.papers, self.filenames):
            row = {
                'filename': filename,  # Use actual JSON filename
                'title': self.clean_text(self.safe_get(paper, 'metadata', 'title')),
                'authors': self.flatten_array(self.safe_get(paper, 'metadata', 'authors', default=[])),
                'year': self.safe_get(paper, 'metadata', 'year'),
                'publication_type': self.clean_text(self.safe_get(paper, 'metadata', 'publication_type')),
                'venue_name': self.clean_text(self.safe_get(paper, 'metadata', 'venue_name')),
                'publisher': self.clean_text(self.safe_get(paper, 'metadata', 'publisher')),
                'doi': self.clean_text(self.safe_get(paper, 'metadata', 'doi')),
                'language': self.clean_text(self.safe_get(paper, 'metadata', 'language')),
                'page_count': self.safe_get(paper, 'metadata', 'page_count'),
                'case_study_locations': self.flatten_array(
                    self.safe_get(paper, 'research_context', 'case_study_locations', default=[])
                ),
                'geographical_regions': self.flatten_array(
                    self.safe_get(paper, 'research_context', 'geographical_regions', default=[])
                ),
                'historical_time_period': self.clean_text(
                    self.safe_get(paper, 'research_context', 'historical_time_period')
                ),
                'pdf_filename': self.clean_text(self.safe_get(paper, 'metadata', 'pdf_filename')),
            }
            rows.append(row)

        self._write_csv('papers_overview.csv', rows)
        return rows

    def generate_heritage_csv(self):
        """Generate heritage_analysis.csv focused on heritage data"""
        print("Generating heritage_analysis.csv...")

        rows = []
        for paper, filename in zip(self.papers, self.filenames):
            row = {
                'filename': filename,  # Use actual JSON filename
                'title': self.clean_text(self.safe_get(paper, 'metadata', 'title')),
                'cultural_heritage_domain': self.flatten_array(
                    self.safe_get(paper, 'research_context', 'cultural_heritage_domain', default=[])
                ),
                'heritage_type': self.flatten_array(
                    self.safe_get(paper, 'research_context', 'heritage_type', default=[])
                ),
                'cultural_entities_analyzed': self.flatten_array(
                    self.safe_get(paper, 'research_context', 'cultural_entities_or_themes_analyzed', default=[])
                ),
                'heritage_terms_detected': self.flatten_array(
                    self.safe_get(paper, 'research_context', 'heritage_terms_detected', default=[])
                ),
                'heritage_values_used': self.flatten_array(
                    self.safe_get(paper, 'research_context', 'heritage_values_used', default=[])
                ),
                'heritage_values_discussed': self.flatten_array(
                    self.safe_get(paper, 'research_context', 'heritage_values_discussed', default=[])
                ),
                'heritage_management_actions': self.flatten_array(
                    self.safe_get(paper, 'research_context', 'heritage_management_actions', default=[])
                ),
                'case_study_locations': self.flatten_array(
                    self.safe_get(paper, 'research_context', 'case_study_locations', default=[])
                ),
                'specific_event': self.clean_text(
                    self.safe_get(paper, 'research_context', 'specific_event_or_incident')
                ),
            }
            rows.append(row)

        self._write_csv('heritage_analysis.csv', rows)
        return rows

    def generate_social_media_csv(self):
        """Generate social_media_analysis.csv with social media data"""
        print("Generating social_media_analysis.csv...")

        rows = []
        for paper, filename in zip(self.papers, self.filenames):
            sm_data = self.safe_get(paper, 'social_media_data', default={})
            dataset = self.safe_get(sm_data, 'dataset_description', default={})
            modality_counts = self.safe_get(dataset, 'modality_counts', default={})

            row = {
                'filename': filename,  # Use actual JSON filename
                'title': self.clean_text(self.safe_get(paper, 'metadata', 'title')),
                'platforms_detected': self.flatten_array(
                    self.safe_get(sm_data, 'platforms_detected', default=[])
                ),
                'data_modalities': self.flatten_array(
                    self.safe_get(sm_data, 'data_modalities_detected', default=[])
                ),
                'has_geolocation': 'Yes' if self.safe_get(sm_data, 'geolocation_data_used', default=[]) else 'No',
                'total_posts': self.safe_get(dataset, 'total_posts_or_items'),
                'total_users': self.safe_get(dataset, 'total_users'),
                'text_posts': modality_counts.get('text_posts'),
                'images': modality_counts.get('images'),
                'videos': modality_counts.get('videos'),
                'collection_method': self.clean_text(
                    self.safe_get(sm_data, 'data_collection', 'collection_method_or_tool')
                ),
                'api_used': self.flatten_array(
                    self.safe_get(sm_data, 'data_collection', 'api_or_tool_used', default=[])
                ),
                'time_coverage': self.clean_text(
                    self.safe_get(dataset, 'time_coverage_or_dates')
                ),
                'spatial_coverage': self.clean_text(
                    self.safe_get(dataset, 'spatial_coverage_or_location')
                ),
            }
            rows.append(row)

        self._write_csv('social_media_analysis.csv', rows)
        return rows

    def generate_ai_methodology_csv(self):
        """Generate ai_methodology.csv with AI methods and models"""
        print("Generating ai_methodology.csv...")

        rows = []
        for paper, filename in zip(self.papers, self.filenames):
            ai_data = self.safe_get(paper, 'ai_methodology', default={})

            row = {
                'filename': filename,  # Use actual JSON filename
                'title': self.clean_text(self.safe_get(paper, 'metadata', 'title')),
                'task_types': self.flatten_array(
                    self.safe_get(ai_data, 'task_types', default=[])
                ),
                'model_families': self.flatten_array(
                    self.safe_get(ai_data, 'model_families', default=[])
                ),
                'specific_models': self.flatten_array(
                    self.safe_get(ai_data, 'specific_models_mentioned', default=[])
                ),
                'pretrained_or_finetuned': self.clean_text(
                    self.safe_get(ai_data, 'pretrained_or_finetuned')
                ),
                'training_data_size': self.safe_get(ai_data, 'training_details', 'train_val_test_split'),
                'loss_functions': self.flatten_array(
                    self.safe_get(ai_data, 'training_details', 'loss_functions', default=[])
                ),
                'optimizer': self.clean_text(
                    self.safe_get(ai_data, 'training_details', 'optimizer')
                ),
                'learning_rate': self.safe_get(ai_data, 'training_details', 'learning_rate'),
                'batch_size': self.safe_get(ai_data, 'training_details', 'batch_size'),
                'epochs': self.safe_get(ai_data, 'training_details', 'epochs'),
            }
            rows.append(row)

        self._write_csv('ai_methodology.csv', rows)
        return rows

    def generate_graph_analysis_csv(self):
        """Generate graph_analysis.csv for papers with graph/network analysis"""
        print("Generating graph_analysis.csv...")

        rows = []
        for paper, filename in zip(self.papers, self.filenames):
            graph_data = self.safe_get(paper, 'graph_or_network_analysis', default={})
            graph_constructed = self.safe_get(graph_data, 'graph_constructed')

            # Only include papers that have graph analysis
            if graph_constructed == 'yes':
                row = {
                    'filename': filename,  # Use actual JSON filename
                    'title': self.clean_text(self.safe_get(paper, 'metadata', 'title')),
                    'graph_type': self.clean_text(self.safe_get(graph_data, 'graph_type')),
                    'node_definition': self.clean_text(self.safe_get(graph_data, 'node_definition')),
                    'edge_definition': self.clean_text(self.safe_get(graph_data, 'edge_definition')),
                    'number_of_nodes': self.safe_get(graph_data, 'number_of_nodes'),
                    'number_of_edges': self.safe_get(graph_data, 'number_of_edges'),
                    'graph_properties': self.flatten_array(
                        self.safe_get(graph_data, 'graph_properties_reported', default=[])
                    ),
                    'construction_method': self.clean_text(
                        self.safe_get(graph_data, 'graph_construction_method')
                    ),
                    'algorithms_used': self.flatten_array(
                        self.safe_get(graph_data, 'graph_algorithms_or_models_used', default=[])
                    ),
                }
                rows.append(row)

        self._write_csv('graph_analysis.csv', rows)
        return rows

    def generate_evaluation_csv(self):
        """Generate evaluation.csv with evaluation metrics and results"""
        print("Generating evaluation.csv...")

        rows = []
        for paper, filename in zip(self.papers, self.filenames):
            eval_data = self.safe_get(paper, 'evaluation', default={})

            row = {
                'filename': filename,  # Use actual JSON filename
                'title': self.clean_text(self.safe_get(paper, 'metadata', 'title')),
                'evaluation_metrics': self.flatten_array(
                    self.safe_get(eval_data, 'evaluation_metrics_used', default=[])
                ),
                'overall_results': self.clean_text(
                    self.safe_get(eval_data, 'overall_results_or_findings')
                ),
                'statistical_tests': self.flatten_array(
                    self.safe_get(eval_data, 'statistical_tests_or_significance', default=[])
                ),
                'qualitative_summary': self.clean_text(
                    self.safe_get(eval_data, 'qualitative_summary_of_results')
                ),
                'error_summary': self.clean_text(
                    self.safe_get(eval_data, 'error_analysis', 'error_summary')
                ),
                'ablation_studies': self.clean_text(
                    self.safe_get(eval_data, 'ablation_studies_or_component_analysis')
                ),
                'baseline_comparison': self.clean_text(
                    self.safe_get(eval_data, 'proposed_vs_baseline_comparison')
                ),
            }
            rows.append(row)

        self._write_csv('evaluation.csv', rows)
        return rows

    def generate_findings_csv(self):
        """Generate findings.csv with key findings and contributions"""
        print("Generating findings.csv...")

        rows = []
        for paper, filename in zip(self.papers, self.filenames):
            findings = self.safe_get(paper, 'findings_and_contributions', default={})
            #contributions = self.safe_get(paper, 'contributions', default={})

            row = {
                'filename': filename,  # Use actual JSON filename
                'title': self.clean_text(self.safe_get(paper, 'metadata', 'title')),
                'cultural_patterns': self.clean_text(
                    self.safe_get(findings, 'cultural_patterns_identified')
                ),
                'social_media_insights': self.clean_text(
                    self.safe_get(findings, 'social_media_behavior_insights')
                ),
                'heritage_findings': self.clean_text(
                    self.safe_get(findings, 'heritage_significance_findings')
                ),
                'spatial_temporal_patterns': self.clean_text(
                    self.safe_get(findings, 'spatial_or_temporal_patterns')
                ),
                'applications': self.clean_text(
                    self.safe_get(findings, 'applications_and_use_cases')
                ),
                'methodological_contributions': self.clean_text(
                    self.safe_get(findings, 'methodological_contributions')
                ),
                'empirical_contributions': self.clean_text(
                    self.safe_get(findings, 'empirical_contributions')
                ),
                'conceptual_contributions': self.clean_text(
                    self.safe_get(findings, 'conceptual_or_theoretical_contributions')
                ),
            }
            rows.append(row)

        self._write_csv('findings.csv', rows)
        return rows

    def generate_master_csv(self):
        """Generate master_dataset.csv with selected key fields"""
        print("Generating master_dataset.csv...")

        rows = []
        for paper, filename in zip(self.papers, self.filenames):
            sm_data = self.safe_get(paper, 'social_media_data', default={})
            ai_data = self.safe_get(paper, 'ai_methodology', default={})
            eval_data = self.safe_get(paper, 'evaluation', default={})

            row = {
                # Metadata
                'filename': filename,  # Use actual JSON filename
                'title': self.clean_text(self.safe_get(paper, 'metadata', 'title')),
                'year': self.safe_get(paper, 'metadata', 'year'),
                'publication_type': self.clean_text(self.safe_get(paper, 'metadata', 'publication_type')),

                # Heritage
                'heritage_domain': self.flatten_array(
                    self.safe_get(paper, 'research_context', 'cultural_heritage_domain', default=[])
                ),
                'case_study_locations': self.flatten_array(
                    self.safe_get(paper, 'research_context', 'case_study_locations', default=[])
                ),

                # Social Media
                'platforms': self.flatten_array(
                    self.safe_get(sm_data, 'platforms_detected', default=[])
                ),
                'data_modalities': self.flatten_array(
                    self.safe_get(sm_data, 'data_modalities_detected', default=[])
                ),
                'total_posts': self.safe_get(sm_data, 'dataset_description', 'total_posts_or_items'),

                # AI
                'task_types': self.flatten_array(
                    self.safe_get(ai_data, 'task_types', default=[])
                ),
                'models_used': self.flatten_array(
                    self.safe_get(ai_data, 'specific_models_mentioned', default=[])
                ),

                # Results
                'evaluation_metrics': self.flatten_array(
                    self.safe_get(eval_data, 'evaluation_metrics_used', default=[])
                ),
                'key_findings': self.clean_text(
                    self.safe_get(eval_data, 'overall_results_or_findings')
                ),
            }
            rows.append(row)

        self._write_csv('master_dataset.csv', rows)
        return rows

    def generate_complete_csv(self):
        """Generate complete_dataset.csv with ALL fields from the JSON"""
        print("Generating complete_dataset.csv (all fields)...")

        rows = []
        for paper, filename in zip(self.papers, self.filenames):
            # Get all major sections
            metadata = self.safe_get(paper, 'metadata', default={})
            research_context = self.safe_get(paper, 'research_context', default={})
            sm_data = self.safe_get(paper, 'social_media_data', default={})
            dataset = self.safe_get(sm_data, 'dataset_description', default={})
            modality_counts = self.safe_get(dataset, 'modality_counts', default={})
            collection = self.safe_get(sm_data, 'data_collection', default={})
            preprocessing = self.safe_get(sm_data, 'annotation_and_labels', default={})
            ai_data = self.safe_get(paper, 'ai_methodology', default={})
            training = self.safe_get(ai_data, 'training_details', default={})
            resources = self.safe_get(ai_data, 'computational_resources', default={})
            graph_data = self.safe_get(paper, 'graph_or_network_analysis', default={})
            eval_data = self.safe_get(paper, 'evaluation', default={})
            quan_results = self.safe_get(eval_data, 'quantitative_results', default={})
            qual_results = self.safe_get(eval_data, 'qualitative_results', default={})
            error_analysis = self.safe_get(eval_data, 'error_analysis', default={})
            benchmark = self.safe_get(eval_data, 'benchmark_comparisons', default={})
            findings = self.safe_get(paper, 'findings_and_contributions', default={})
            #contributions = self.safe_get(paper, 'contributions', default={})
            limitations = self.safe_get(paper, 'ethics_and_reproducibility', default={})
            reproducibility = self.safe_get(limitations, 'reproducibility', default={})
            patterns = self.safe_get(paper, 'pattern_hits_and_notes', default={})

            row = {
                # === METADATA BLOCK ===
                'filename': filename,  # Use actual JSON filename
                'title': self.clean_text(self.safe_get(metadata, 'title')),
                'authors': self.flatten_array(self.safe_get(metadata, 'authors', default=[])),
                'affiliations': self.flatten_array(self.safe_get(metadata, 'affiliations', default=[])),
                'year': self.safe_get(metadata, 'year'),
                'publication_type': self.clean_text(self.safe_get(metadata, 'publication_type')),
                'venue_name': self.clean_text(self.safe_get(metadata, 'venue_name')),
                'publisher': self.clean_text(self.safe_get(metadata, 'publisher')),
                'doi': self.clean_text(self.safe_get(metadata, 'doi')),
                'url': self.clean_text(self.safe_get(metadata, 'url')),
                'language': self.clean_text(self.safe_get(metadata, 'language')),
                'page_count': self.safe_get(metadata, 'page_count'),

                # === RESEARCH CONTEXT BLOCK ===
                'cultural_heritage_domain': self.flatten_array(
                    self.safe_get(research_context, 'cultural_heritage_domain', default=[])),
                'heritage_type': self.flatten_array(self.safe_get(research_context, 'heritage_type', default=[])),
                'cultural_entities_analyzed': self.flatten_array(
                    self.safe_get(research_context, 'cultural_entities_or_themes_analyzed', default=[])),
                'heritage_terms_detected': self.flatten_array(
                    self.safe_get(research_context, 'heritage_terms_detected', default=[])),
                'heritage_values_used': self.flatten_array(
                    self.safe_get(research_context, 'heritage_values_used', default=[])),
                'heritage_values_discussed': self.flatten_array(
                    self.safe_get(research_context, 'heritage_values_discussed', default=[])),
                'heritage_management_actions': self.flatten_array(
                    self.safe_get(research_context, 'heritage_management_actions', default=[])),
                'case_study_locations': self.flatten_array(
                    self.safe_get(research_context, 'case_study_locations', default=[])),
                'geographical_regions': self.flatten_array(
                    self.safe_get(research_context, 'geographical_regions', default=[])),
                'historical_time_period': self.clean_text(
                    self.safe_get(research_context, 'historical_time_period')),
                'specific_event_or_incident': self.clean_text(
                    self.safe_get(research_context, 'specific_event_or_incident')),
                'research_problem_statement': self.clean_text(
                    self.safe_get(research_context, 'research_problem_statement')),
                'research_objectives': self.clean_text(
                    self.safe_get(research_context, 'research_objectives')),

                # === SOCIAL MEDIA DATA BLOCK ===
                'platforms_detected': self.flatten_array(self.safe_get(sm_data, 'platforms_detected', default=[])),
                'data_modalities_detected': self.flatten_array(
                    self.safe_get(sm_data, 'data_modalities_detected', default=[])),
                'geolocation_data_used': self.flatten_array(
                    self.safe_get(sm_data, 'geolocation_data_used', default=[])),
                'spatial_data_characteristics': self.flatten_array(
                    self.safe_get(sm_data, 'spatial_data_characteristics', default=[])),
                'temporal_data_characteristics': self.flatten_array(
                    self.safe_get(sm_data, 'temporal_data_characteristics', default=[])),
                'engagement_metrics_detected': self.flatten_array(
                    self.safe_get(sm_data, 'engagement_metrics_detected', default=[])),
                'metadata_fields_detected': self.flatten_array(
                    self.safe_get(sm_data, 'metadata_fields_detected', default=[])),

                # Dataset description
                'dataset_total_posts': self.safe_get(dataset, 'total_posts'),
                'dataset_total_users': self.safe_get(dataset, 'total_users'),
                'dataset_text_posts': modality_counts.get('text_posts'),
                'dataset_images': modality_counts.get('images'),
                'dataset_videos': modality_counts.get('videos'),
                'dataset_3d_objects': modality_counts.get('3d_objects'),
                'dataset_geotagged_entries': modality_counts.get('geotagged_entries'),
                'dataset_time_coverage': self.clean_text(self.safe_get(dataset, 'time_coverage')),
                'dataset_spatial_coverage': self.clean_text(self.safe_get(dataset, 'spatial_coverage')),

                # Data collection
                'collection_method': self.clean_text(self.safe_get(collection, 'collection_method')),
                'api_or_tool_used': self.clean_text(self.safe_get(collection, 'api_or_tool_used')),
                'sampling_strategy': self.clean_text(self.safe_get(collection, 'sampling_strategy')),

                # Data preprocessing
                'data_cleaning_preprocessing': self.clean_text(
                    self.safe_get(collection, 'data_cleaning_preprocessing')),
                'filtering_criteria': self.clean_text(self.safe_get(collection, 'filtering_criteria')),
                'annotation_method': self.clean_text(self.safe_get(preprocessing, 'annotation_method')),
                'annotators': self.clean_text(self.safe_get(preprocessing, 'annotators')),
                'label_schema': self.clean_text(self.safe_get(preprocessing, 'label_schema')),
                'inter_annotator_agreement': self.clean_text(
                    self.safe_get(preprocessing, 'inter_annotator_agreement')),

                # === AI METHODOLOGY BLOCK ===
                'task_types': self.flatten_array(self.safe_get(ai_data, 'task_types', default=[])),
                'model_families': self.flatten_array(self.safe_get(ai_data, 'model_families', default=[])),
                'specific_models_mentioned': self.flatten_array(
                    self.safe_get(ai_data, 'specific_models_mentioned', default=[])),
                'pretrained_or_finetuned': self.clean_text(self.safe_get(ai_data, 'pretrained_or_finetuned')),
                'embeddings_or_representations': self.flatten_array(
                    self.safe_get(ai_data, 'embeddings_or_representations', default=[])),
                'feature_extraction_methods': self.flatten_array(
                    self.safe_get(ai_data, 'feature_extraction_methods', default=[])),
                'input_features_used': self.flatten_array(self.safe_get(ai_data, 'input_features_used', default=[])),
                'multimodal_fusion_method': self.clean_text(self.safe_get(ai_data, 'multimodal_fusion_method')),

                # Training details
                'train_val_test_split': self.clean_text(self.safe_get(training, 'train_val_test_split')),
                'loss_functions': self.flatten_array(self.safe_get(training, 'loss_functions', default=[])),
                'optimizer': self.clean_text(self.safe_get(training, 'optimizer')),
                'learning_rate': self.safe_get(training, 'learning_rate'),
                'batch_size': self.safe_get(training, 'batch_size'),
                'epochs': self.safe_get(training, 'epochs'),
                'regularization': self.flatten_array(self.safe_get(training, 'regularization', default=[])),
                'augmentation_methods': self.flatten_array(
                    self.safe_get(training, 'augmentation_methods', default=[])),

                'hyperparameters': self.clean_text(self.safe_get(ai_data, 'hyperparameters', default=[])),
                'baseline_models': self.flatten_array(
                    self.safe_get(ai_data, 'baseline_models', default=[])),

                'hardware': self.clean_text(self.safe_get(resources, 'hardware')),
                'gpu_models': self.flatten_array(self.safe_get(resources, 'gpu_models', default=[])),
                'num_gpus': self.safe_get(resources, 'num_gpus'),
                'memory': self.clean_text(self.safe_get(resources, 'memory')),
                'runtime': self.clean_text(self.safe_get(resources, 'runtime')),

                'model_architecture_details': self.clean_text(
                    self.safe_get(ai_data, 'model_architecture_details')),
                'training_procedure_summary': self.clean_text(
                    self.safe_get(ai_data, 'training_procedure_summary')),

                # === GRAPH/NETWORK ANALYSIS BLOCK ===
                'graph_constructed': self.clean_text(self.safe_get(graph_data, 'graph_constructed')),
                'graph_type': self.clean_text(self.safe_get(graph_data, 'graph_type')),
                'node_definition': self.clean_text(self.safe_get(graph_data, 'node_definition')),
                'edge_definition': self.clean_text(self.safe_get(graph_data, 'edge_definition')),
                'number_of_nodes': self.safe_get(graph_data, 'number_of_nodes'),
                'number_of_edges': self.safe_get(graph_data, 'number_of_edges'),
                'graph_properties_reported': self.flatten_array(
                    self.safe_get(graph_data, 'graph_properties_reported', default=[])),
                'graph_construction_method': self.clean_text(self.safe_get(graph_data, 'graph_construction_method')),
                'graph_algorithms_or_models_used': self.flatten_array(
                    self.safe_get(graph_data, 'graph_algorithms_or_models_used', default=[])),
                'graph_visualization_methods': self.flatten_array(
                    self.safe_get(graph_data, 'graph_visualization_methods', default=[])),
                'graph_purpose_in_study': self.clean_text(self.safe_get(graph_data, 'graph_purpose_in_study')),

                # === EVALUATION BLOCK ===
                'evaluation_metrics_used': self.flatten_array(
                    self.safe_get(eval_data, 'evaluation_metrics_used', default=[])),

                'overall_results': self.clean_text(self.safe_get(quan_results, 'overall_results')),
                'per_class_results': self.clean_text(self.safe_get(quan_results, 'per_class_or_category_results')),
                'per_model_results': self.clean_text(self.safe_get(quan_results, 'per_model_results')),
                'statistical_tests': self.flatten_array(
                    self.safe_get(quan_results, 'statistical_tests', default=[])),

                'qualitative_summary': self.clean_text(self.safe_get(qual_results, 'summary')),
                'example_outputs': self.clean_text(self.safe_get(qual_results, 'example_outputs')),
                'qualitative_comparisons': self.clean_text(
                    self.safe_get(qual_results, 'qualitative_comparisons')),

                'error_summary': self.clean_text(self.safe_get(error_analysis, 'error_summary')),
                'error_examples': self.clean_text(self.safe_get(error_analysis, 'error_examples')),
                'error_causes': self.clean_text(self.safe_get(error_analysis, 'identified_error_causes')),

                'ablation_studies': self.clean_text(
                    self.safe_get(eval_data, 'ablation_studies')),
                'cross_validation_strategy': self.clean_text(
                    self.safe_get(eval_data, 'cross_validation_or_splitting_strategy')),

                'baseline_models_eval': self.flatten_array(
                    self.safe_get(benchmark, 'baseline_models', default=[])),
                'proposed_vs_baseline': self.clean_text(
                    self.safe_get(benchmark, 'proposed_vs_baseline_results')),

                'visualization_types_used': self.flatten_array(
                    self.safe_get(eval_data, 'visualization_types_used', default=[])),

                # === FINDINGS & INSIGHTS BLOCK ===
                'cultural_patterns_identified': self.clean_text(
                    self.safe_get(findings, 'cultural_patterns_identified')),
                'social_media_behavior_insights': self.clean_text(
                    self.safe_get(findings, 'social_media_behavior_insights')),
                'heritage_significance_findings': self.clean_text(
                    self.safe_get(findings, 'heritage_significance_findings')),
                'perceptual_or_sentiment_insights': self.clean_text(
                    self.safe_get(findings, 'perceptual_or_sentiment_insights')),
                'community_user_differences': self.clean_text(
                    self.safe_get(findings, 'community_or_user_group_differences')),
                'spatial_or_temporal_patterns': self.clean_text(
                    self.safe_get(findings, 'spatial_or_temporal_patterns')),
                'applications_and_use_cases': self.clean_text(
                    self.safe_get(findings, 'applications_and_use_cases')),
                'implications_for_heritage_management': self.clean_text(
                    self.safe_get(findings, 'implications_for_heritage_management')),

                # === CONTRIBUTIONS BLOCK ===
                'methodological_contributions': self.clean_text(
                    self.safe_get(findings, 'methodological_contributions')),
                'empirical_contributions': self.clean_text(self.safe_get(findings, 'empirical_contributions')),
                'conceptual_contributions': self.clean_text(
                    self.safe_get(findings, 'conceptual_contributions')),

                # === LIMITATIONS & ETHICS BLOCK ===
                'limitations_or_challenges': self.clean_text(
                    self.safe_get(findings, 'limitations_or_challenges')),

                'privacy_and_data_protection': self.clean_text(
                    self.safe_get(limitations, 'privacy_and_data_protection')),
                'bias_fairness_representativeness': self.clean_text(
                    self.safe_get(limitations, 'bias_fairness_and_representativeness')),
                'ethical_frameworks_or_guidelines': self.clean_text(
                    self.safe_get(limitations, 'ethical_frameworks_or_guidelines')),
                'cultural_community_considerations': self.clean_text(
                    self.safe_get(limitations, 'cultural_and_community_considerations')),
                'legal_or_policy_constraints': self.clean_text(
                    self.safe_get(limitations, 'legal_or_policy_constraints')),
                'risks_to_people_or_heritage': self.clean_text(
                    self.safe_get(limitations, 'risks_to_people_or_heritage')),
                'copyright_or_licensing_issues': self.clean_text(
                    self.safe_get(limitations, 'copyright_or_licensing_issues')),
                'limitations_acknowledged': self.clean_text(
                    self.safe_get(limitations, 'limitations_acknowledged')),

                # Reproducibility
                'code_availability': self.clean_text(self.safe_get(reproducibility, 'code_availability')),
                'data_availability': self.clean_text(self.safe_get(reproducibility, 'data_availability')),
                'model_checkpoint_availability': self.clean_text(
                    self.safe_get(reproducibility, 'model_checkpoint_availability')),
                'dataset_identifier_or_link': self.clean_text(
                    self.safe_get(reproducibility, 'dataset_identifier_or_link')),
                'computational_environment': self.clean_text(
                    self.safe_get(reproducibility, 'computational_environment')),
                'open_source_libraries_used': self.flatten_array(
                    self.safe_get(reproducibility, 'open_source_libraries_used', default=[])),

                'pipeline_transparency': self.clean_text(
                    self.safe_get(limitations, 'pipeline_transparency_or_completeness')),

                # === PATTERN HITS & NOTES BLOCK ===
                'raw_terms_detected': self.flatten_array(self.safe_get(patterns, 'raw_terms_detected', default=[])),
                'ambiguous_or_uncertain_cases': self.flatten_array(
                    self.safe_get(patterns, 'ambiguous_or_uncertain_cases', default=[])),
                'extraction_notes': self.clean_text(self.safe_get(patterns, 'extraction_notes')),
                'additional_relevant_information': self.clean_text(
                    self.safe_get(patterns, 'additional_relevant_information')),
            }
            rows.append(row)

        self._write_csv('complete_dataset.csv', rows)
        print(f"   Total columns: {len(rows[0]) if rows else 0}")
        return rows

    def _write_csv(self, filename: str, rows: List[Dict]):
        """Write rows to CSV file with proper handling of embedded newlines"""
        if not rows:
            print(f"⚠️  No data for {filename}")
            return

        output_path = self.output_dir / filename

        # Get all unique keys across all rows
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        # Clean all data before writing
        cleaned_rows = []
        for row in rows:
            cleaned_row = {}
            for key in fieldnames:
                value = row.get(key, "")
                # Apply cleaning to all string values
                if isinstance(value, str):
                    cleaned_row[key] = self.clean_text(value)
                else:
                    cleaned_row[key] = value
            cleaned_rows.append(cleaned_row)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cleaned_rows)

        print(f"✅ Created: {output_path} ({len(rows)} rows)")

    def generate_excel(self, csv_data: Dict[str, List[Dict]]):
        """Generate Excel file with multiple sheets"""
        if pd is None:
            print("\n⚠️  Skipping Excel generation (pandas not installed)")
            return

        print("\nGenerating master_dataset.xlsx...")

        output_path = self.output_dir / 'master_dataset.xlsx'

        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for sheet_name, rows in csv_data.items():
                    if rows:
                        df = pd.DataFrame(rows)
                        # Truncate sheet name to 31 chars (Excel limit)
                        sheet_name_clean = sheet_name[:31]
                        df.to_excel(writer, sheet_name=sheet_name_clean, index=False)

            print(f"✅ Created: {output_path}")
            print(f"   Sheets: {len(csv_data)}")
        except Exception as e:
            print(f"❌ Excel generation failed: {e}")

    def generate_statistics(self):
        """Generate statistics about the dataset"""
        print("\n" + "=" * 70)
        print("DATASET STATISTICS")
        print("=" * 70)

        print(f"\nTotal papers: {len(self.papers)}")

        # Publication years
        years = [self.safe_get(p, 'metadata', 'year') for p in self.papers]
        years = [y for y in years if y]
        if years:
            print(f"Year range: {min(years)} - {max(years)}")

        # Publication types
        types = defaultdict(int)
        for p in self.papers:
            pub_type = self.safe_get(p, 'metadata', 'publication_type')
            if pub_type:
                types[pub_type] += 1

        if types:
            print("\nPublication types:")
            for pub_type, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {pub_type}: {count}")

        # Platforms
        platforms = defaultdict(int)
        for p in self.papers:
            plats = self.safe_get(p, 'social_media_data', 'platforms_detected', default=[])
            for plat in plats:
                if plat:
                    platforms[plat] += 1

        if platforms:
            print("\nTop social media platforms:")
            for plat, count in sorted(platforms.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {plat}: {count}")

        # AI models
        models = defaultdict(int)
        for p in self.papers:
            mods = self.safe_get(p, 'ai_methodology', 'specific_models_mentioned', default=[])
            for mod in mods:
                if mod:
                    models[mod] += 1

        if models:
            print("\nTop AI models:")
            for mod, count in sorted(models.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {mod}: {count}")

        # Graph analysis
        graph_count = sum(1 for p in self.papers
                          if self.safe_get(p, 'graph_or_network_analysis', 'graph_constructed') == 'yes')
        print(f"\nPapers with graph analysis: {graph_count}")

        print()

    def convert(self):
        """Run full conversion process"""
        # Load JSON files
        num_papers = self.load_json_files()

        if num_papers == 0:
            print("No papers loaded. Exiting.")
            sys.exit(1)

        print("=" * 70)
        print("Generating CSV files...")
        print("=" * 70)
        print()

        # Generate all CSV files
        csv_data = {
            'Overview': self.generate_overview_csv(),
            'Heritage': self.generate_heritage_csv(),
            'Social Media': self.generate_social_media_csv(),
            'AI Methodology': self.generate_ai_methodology_csv(),
            'Graph Analysis': self.generate_graph_analysis_csv(),
            'Evaluation': self.generate_evaluation_csv(),
            'Findings': self.generate_findings_csv(),
            'Master': self.generate_master_csv(),
            'Complete': self.generate_complete_csv(),  # NEW: All fields
        }

        # Generate Excel file
        self.generate_excel(csv_data)

        # Print statistics
        self.generate_statistics()

        print("=" * 70)
        print("CONVERSION COMPLETE!")
        print("=" * 70)
        print(f"\nOutput directory: {self.output_dir}")
        print("\nGenerated files:")
        for csv_file in sorted(self.output_dir.glob("*.csv")):
            print(f"  ✅ {csv_file.name}")
        if (self.output_dir / 'master_dataset.xlsx').exists():
            print(f"  ✅ master_dataset.xlsx")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Convert heritage paper JSON data to CSV format'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input directory containing JSON files'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for CSV files'
    )

    args = parser.parse_args()

    # Check input directory exists
    if not Path(args.input).exists():
        print(f"ERROR: Input directory does not exist: {args.input}")
        sys.exit(1)

    # Run conversion
    converter = JSONtoCSVConverter(args.input, args.output)
    converter.convert()


if __name__ == '__main__':
    main()