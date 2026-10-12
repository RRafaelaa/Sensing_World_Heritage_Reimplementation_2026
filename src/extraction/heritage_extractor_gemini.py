#!/usr/bin/env python3
"""
Heritage Paper Extractor - Google Gemini Version
High-quality extraction using Google's Gemini API

Improvements over original:
- Better error handling and retry logic
- Progress tracking with statistics
- Resume capability (skip completed files)
- Cost estimation and tracking
- Validation and logging
- Configurable via command line
- Better prompt engineering
- JSON cleaning and validation
- Rate limiting support
"""

import os
import sys
import json
import re
import time
import argparse
from pathlib import Path
from string import Template
from datetime import datetime
from typing import Optional, Dict

PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "config" / "extraction" / "prompt_template.txt"

try:
    import google.generativeai as genai
    from google.api_core import retry, exceptions
except ImportError:
    print("ERROR: google-generativeai package not installed")
    print("Install with: pip install google-generativeai")
    sys.exit(1)


class GeminiExtractor:
    """Extract structured information using Google Gemini API"""

    def __init__(self, api_key: str, schema_path: str, instructions_path: str,
                 model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize with API key and extraction guidelines

        Args:
            api_key: Google API key
            schema_path: Path to schema.txt
            instructions_path: Path to instructions.txt
            model_name: Gemini model to use (default: gemini-2.0-flash-exp)
        """
        # Configure API
        genai.configure(api_key=api_key)

        # Load the shared prompt template, schema, and instructions
        self.schema = self.load_file(schema_path)
        self.instructions = self.load_file(instructions_path)
        self.prompt_template = Template(self.load_file(str(PROMPT_TEMPLATE_PATH)))

        # Model configuration
        self.generation_config = {
            "temperature": 0.1,  # Low for factual extraction
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        }

        # Safety settings - very permissive for academic/research content
        # Using BLOCK_NONE for academic research papers to avoid false positives
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]

        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
        )

        self.model_name = model_name

        # Statistics
        self.papers_processed = 0
        self.papers_failed = 0
        self.total_tokens_used = 0
        self.total_cost = 0.0

        print(f"✅ Initialized Gemini Extractor")
        print(f"   Model: {model_name}")
        print(f"   Max output: {self.generation_config['max_output_tokens']} tokens")

    def load_file(self, path: str) -> str:
        """Load file content"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"❌ Error reading {path}: {e}")
            sys.exit(1)

    def create_extraction_prompt(self, paper_text: str, filename: str) -> str:
        """Create comprehensive extraction prompt"""
        return self.prompt_template.substitute(
            filename=filename,
            instructions=self.instructions,
            schema=self.schema,
            paper_text=paper_text,
        )

    def clean_json_response(self, response_text: str) -> str:
        """Clean common formatting issues in API responses"""
        clean_text = response_text.strip()

        # Remove markdown code blocks
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]

        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]

        clean_text = clean_text.strip()

        # Fix common JSON errors
        # Fix trailing commas before closing braces/brackets
        import re
        clean_text = re.sub(r',(\s*[}\]])', r'\1', clean_text)

        # Fix missing commas between array elements (common error)
        # This is tricky - only do basic fixes
        clean_text = re.sub(r'"\s*\n\s*"', '",\n"', clean_text)

        # Fix control characters that break JSON
        clean_text = clean_text.replace('\t', '  ')
        clean_text = clean_text.replace('\r\n', '\n')

        return clean_text

    def extract_paper(self, text_content: str, filename: str) -> Optional[Dict]:
        """
        Extract information from a single paper using Gemini API

        Returns:
            Populated schema dictionary or None if failed
        """

        print(f"\n{'=' * 70}")
        print(f"Processing: {filename}")
        print(f"{'=' * 70}")
        print(f"Paper length: {len(text_content):,} characters")

        # Create prompt
        prompt = self.create_extraction_prompt(text_content, filename)
        print(f"Prompt length: {len(prompt):,} characters")
        print("Sending to Gemini API...")

        try:
            # Call API with retry logic
            response = self.model.generate_content(
                prompt,
                request_options={
                    'retry': retry.Retry(
                        predicate=retry.if_transient_error,
                        initial=1.0,
                        maximum=10.0,
                        multiplier=2.0,
                        timeout=300.0
                    )
                }
            )

            # Check if response was blocked by safety filters
            if not response.candidates:
                print("❌ Response blocked by safety filters (no candidates returned)")
                print("   This paper may contain sensitive content that triggered Gemini's safety filters.")
                print("   Skipping this paper...")
                self.papers_failed += 1
                return None

            candidate = response.candidates[0]

            # Check finish reason
            if candidate.finish_reason == 2:  # SAFETY
                print("❌ Response blocked by safety filters")
                if hasattr(candidate, 'safety_ratings'):
                    print(f"   Safety ratings: {candidate.safety_ratings}")
                print("   This is a false positive on legitimate academic content.")
                print("   Consider using Claude API which handles academic content better.")
                print("   Skipping this paper...")
                self.papers_failed += 1
                return None
            elif candidate.finish_reason == 3:  # RECITATION
                print("❌ Response blocked due to recitation concerns")
                print("   Skipping this paper...")
                self.papers_failed += 1
                return None
            elif candidate.finish_reason not in [0, 1]:  # 0=UNSPECIFIED, 1=STOP (normal)
                print(f"⚠️  Unexpected finish_reason: {candidate.finish_reason}")

            # Extract response text
            response_text = response.text

            # Track usage (Gemini provides token counts)
            try:
                # Usage metadata varies by model version
                if hasattr(response, 'usage_metadata'):
                    usage = response.usage_metadata
                    input_tokens = getattr(usage, 'prompt_token_count', 0)
                    output_tokens = getattr(usage, 'candidates_token_count', 0)
                    total_tokens = getattr(usage, 'total_token_count', input_tokens + output_tokens)
                else:
                    # Estimate if metadata not available
                    input_tokens = len(prompt) // 4  # Rough estimate
                    output_tokens = len(response_text) // 4
                    total_tokens = input_tokens + output_tokens

                self.total_tokens_used += total_tokens

                # Calculate cost based on model
                # Gemini Flash: $0.075/1M input, $0.30/1M output
                # Gemini Pro 2.0: $1.25/1M input, $5.00/1M output
                # Gemini 3 Pro: $1.50/1M input, $7.50/1M output (estimated)
                if "flash" in self.model_name.lower():
                    cost = (input_tokens / 1_000_000 * 0.075) + (output_tokens / 1_000_000 * 0.30)
                elif "gemini-3" in self.model_name.lower():
                    # Gemini 3 Pro pricing (may need adjustment when official pricing released)
                    cost = (input_tokens / 1_000_000 * 1.50) + (output_tokens / 1_000_000 * 7.50)
                else:  # Pro 2.0 and other models
                    cost = (input_tokens / 1_000_000 * 1.25) + (output_tokens / 1_000_000 * 5.00)

                self.total_cost += cost

                print(f"\n📊 Token Usage:")
                print(f"  Input: {input_tokens:,} tokens")
                print(f"  Output: {output_tokens:,} tokens")
                print(f"  Total: {total_tokens:,} tokens")
                print(f"  Cost: ${cost:.4f}")

            except Exception as e:
                print(f"⚠️  Could not extract usage metadata: {e}")

            # Clean and parse JSON
            try:
                clean_response = self.clean_json_response(response_text)
                result = json.loads(clean_response)

                print(f"✅ Successfully extracted and parsed JSON")
                self.papers_processed += 1

                return result

            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing error: {e}")
                print(f"Response preview: {response_text[:500]}...")

                # Try aggressive fixing
                print("🔧 Attempting to fix JSON...")
                try:
                    import re
                    fixed = clean_response

                    # Remove any text before first {
                    match = re.search(r'\{', fixed)
                    if match:
                        fixed = fixed[match.start():]

                    # Remove any text after last }
                    last_brace = fixed.rfind('}')
                    if last_brace != -1:
                        fixed = fixed[:last_brace + 1]

                    # Fix common issues
                    # Remove trailing commas before } or ]
                    fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)

                    # Try to parse fixed version
                    result = json.loads(fixed)
                    print("✅ JSON fixed and parsed successfully!")
                    self.papers_processed += 1
                    return result

                except Exception as fix_error:
                    print(f"❌ Could not auto-fix JSON: {fix_error}")

                # Save problematic response for manual inspection
                error_file = f"error_response_{filename}"
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(response_text)
                print(f"💾 Saved raw response to: {error_file}")
                print("   You can manually inspect and fix this JSON later.")

                self.papers_failed += 1
                return None

        except exceptions.GoogleAPIError as e:
            print(f"❌ Google API Error: {e}")
            self.papers_failed += 1
            return None

        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            self.papers_failed += 1
            return None

    def process_directory(self, input_dir: str, output_dir: str,
                          max_papers: Optional[int] = None,
                          start_from: int = 0,
                          delay: float = 1.0):
        """
        Process all text files in directory

        Args:
            input_dir: Directory with .txt files
            output_dir: Directory for JSON outputs
            max_papers: Maximum number of papers to process
            start_from: Skip first N papers
            delay: Seconds to wait between API calls
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all .txt files
        txt_files = sorted(list(input_path.glob('*.txt')))

        print(f"\n{'=' * 70}")
        print(f"GEMINI EXTRACTOR - Starting")
        print(f"{'=' * 70}")
        print(f"Model: {self.model_name}")
        print(f"Total files found: {len(txt_files)}")
        print(f"Start from: {start_from}")
        print(f"Max papers: {max_papers if max_papers else 'ALL'}")

        # Apply filters
        txt_files = txt_files[start_from:]
        if max_papers:
            txt_files = txt_files[:max_papers]

        print(f"Will process: {len(txt_files)} papers")

        # Estimate cost
        avg_tokens_per_paper = 13000
        estimated_tokens = len(txt_files) * avg_tokens_per_paper

        if "flash" in self.model_name.lower():
            estimated_cost = (estimated_tokens / 1_000_000) * 0.075 + \
                             (estimated_tokens / 1_000_000) * 0.30 * 0.5
        elif "gemini-3" in self.model_name.lower():
            estimated_cost = (estimated_tokens / 1_000_000) * 1.50 + \
                             (estimated_tokens / 1_000_000) * 7.50 * 0.5
        else:
            estimated_cost = (estimated_tokens / 1_000_000) * 1.25 + \
                             (estimated_tokens / 1_000_000) * 5.00 * 0.5

        print(f"\n📊 Estimated Usage:")
        print(f"  Tokens: ~{estimated_tokens:,}")
        print(f"  Cost: ~${estimated_cost:.2f}")

        # Confirm
        if len(txt_files) > 10:
            confirm = input(f"\nProceed with processing {len(txt_files)} papers? (y/n): ")
            if confirm.lower() != 'y':
                print("Cancelled by user")
                return

        print(f"\n{'=' * 70}")
        print("Starting extraction...")
        print(f"{'=' * 70}")

        start_time = time.time()

        for i, txt_file in enumerate(txt_files, 1):
            print(f"\n[{i}/{len(txt_files)}] {txt_file.name}")

            try:
                # Read content
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    text_content = f.read()

                # Check if already processed
                output_file = output_path / txt_file.name
                if output_file.exists():
                    print(f"⏭️  Already processed, skipping...")
                    continue

                # Extract
                result = self.extract_paper(text_content, txt_file.name)

                if result:
                    # Save output
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)

                    print(f"💾 Saved: {output_file.name}")
                else:
                    print(f"⚠️  Failed to extract")

                # Progress update
                elapsed = time.time() - start_time
                avg_time_per_paper = elapsed / i
                remaining_papers = len(txt_files) - i
                estimated_remaining = avg_time_per_paper * remaining_papers

                print(f"\n⏱️  Progress: {i}/{len(txt_files)} papers")
                print(f"   Elapsed: {elapsed / 60:.1f} minutes")
                print(f"   Est. remaining: {estimated_remaining / 60:.1f} minutes")
                print(f"   Total cost so far: ${self.total_cost:.2f}")

                # Rate limiting
                if i < len(txt_files):
                    print(f"   Waiting {delay:.1f} seconds before next paper...")
                    time.sleep(delay)

            except Exception as e:
                print(f"❌ Error processing {txt_file.name}: {e}")
                self.papers_failed += 1
                continue

        # Final summary
        total_time = time.time() - start_time

        print(f"\n{'=' * 70}")
        print(f"EXTRACTION COMPLETE")
        print(f"{'=' * 70}")
        print(f"✅ Successfully processed: {self.papers_processed}")
        print(f"❌ Failed: {self.papers_failed}")
        print(f"📊 Total tokens used: {self.total_tokens_used:,}")
        print(f"💰 Total cost: ${self.total_cost:.2f}")
        print(f"⏱️  Total time: {total_time / 60:.1f} minutes")
        print(f"📁 Output directory: {output_path}")
        print(f"{'=' * 70}\n")

        # Save log
        log_file = output_path / f"extraction_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "papers_processed": self.papers_processed,
            "papers_failed": self.papers_failed,
            "total_tokens": self.total_tokens_used,
            "total_cost": self.total_cost,
            "total_time_minutes": total_time / 60,
            "input_directory": str(input_dir),
            "output_directory": str(output_dir)
        }

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"📋 Processing log saved to: {log_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Extract heritage paper information using Google Gemini',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all papers with Flash model (fast, cheap)
  python heritage_extractor_gemini.py \\
    --api-key YOUR_KEY \\
    --input papers_txt \\
    --output papers_json \\
    --schema schema.txt \\
    --instructions instructions.txt

  # Process with Pro model (higher quality)
  python heritage_extractor_gemini.py \\
    --api-key YOUR_KEY \\
    --model gemini-2.0-pro-exp \\
    --input papers_txt \\
    --output papers_json \\
    --schema schema.txt \\
    --instructions instructions.txt

  # Test with 5 papers
  python heritage_extractor_gemini.py \\
    --api-key YOUR_KEY \\
    --input papers_txt \\
    --output papers_json \\
    --schema schema.txt \\
    --instructions instructions.txt \\
    --max-papers 5

Environment Variable:
  Set GOOGLE_API_KEY environment variable instead of --api-key
  export GOOGLE_API_KEY=your_key_here
        """
    )

    parser.add_argument(
        '--api-key',
        help='Google API key (or set GOOGLE_API_KEY env var)'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input directory containing .txt files'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for JSON files'
    )
    parser.add_argument(
        '--schema',
        required=True,
        help='Path to schema.txt file'
    )
    parser.add_argument(
        '--instructions',
        required=True,
        help='Path to instructions.txt file'
    )
    parser.add_argument(
        '--model',
        default='gemini-2.0-flash-exp',
        choices=[
            'gemini-2.0-flash-exp',
            'gemini-2.0-flash-001',
            'gemini-2.5-flash',
            'gemini-2.5-pro',
            'gemini-2.0-pro-exp',
            'gemini-3-pro-preview',
            'gemini-1.5-pro',
            'gemini-1.5-flash',
            'gemini-exp-1206'
        ],
        help='Gemini model to use (default: gemini-2.0-flash-exp, try gemini-2.5-flash for latest)'
    )
    parser.add_argument(
        '--max-papers',
        type=int,
        help='Maximum number of papers to process'
    )
    parser.add_argument(
        '--start-from',
        type=int,
        default=0,
        help='Start from paper N (for resuming)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Seconds to wait between API calls (default: 1.0)'
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get('GOOGLE_API_KEY')

    if not api_key:
        print("ERROR: API key required")
        print("Either:")
        print("  1. Use --api-key YOUR_KEY")
        print("  2. Set environment variable: export GOOGLE_API_KEY=your_key")
        sys.exit(1)

    print("=" * 70)
    print("Heritage Paper Extraction - Google Gemini Version")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Schema: {args.schema}")
    print(f"Instructions: {args.instructions}")
    print("=" * 70)

    # Initialize extractor
    extractor = GeminiExtractor(
        api_key,
        args.schema,
        args.instructions,
        model_name=args.model
    )

    # Process files
    extractor.process_directory(
        args.input,
        args.output,
        max_papers=args.max_papers,
        start_from=args.start_from,
        delay=args.delay
    )


if __name__ == '__main__':
    main()
