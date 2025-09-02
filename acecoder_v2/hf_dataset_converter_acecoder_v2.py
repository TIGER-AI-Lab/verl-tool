#!/usr/bin/env python3
"""
🤗 Hugging Face Dataset Converter for AceCoderV2

This script converts AceCoderV2 dataset into a simplified format containing 
generated problem descriptions and test cases from synthesis_result.

Key Features:
- Extracts GENERATED complex algorithm problems (synthesis_result['problem'])
- Extracts GENERATED test cases (synthesis_result['tests'])
- Supports all original filtering and merging functionality
- Maintains backward compatibility with original data structure

Usage:
    python hf_dataset_converter.py --dataset_name siyiwu0330/acecoderv2 --output output.jsonl
    python hf_dataset_converter.py --local_file data.jsonl --output filtered.jsonl --round 3
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_from_huggingface(dataset_name: str, split: str = "train") -> List[Dict]:
    """Load dataset from Hugging Face Hub."""
    try:
        from datasets import load_dataset
        logger.info(f"📥 Loading dataset from Hugging Face: {dataset_name}")
        
        dataset = load_dataset(dataset_name, split=split)
        data = [dict(item) for item in dataset]
        
        logger.info(f"✅ Loaded {len(data)} items from Hugging Face")
        return data
        
    except ImportError:
        logger.error("❌ Please install the datasets library: pip install datasets")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error loading dataset from Hugging Face: {e}")
        sys.exit(1)

def load_from_local_file(file_path: str) -> List[Dict]:
    """Load dataset from local file."""
    try:
        logger.info(f"📥 Loading dataset from local file: {file_path}")
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            else:
                data = json.load(f)
        
        logger.info(f"✅ Loaded {len(data)} items from local file")
        return data
        
    except Exception as e:
        logger.error(f"❌ Error loading local file: {e}")
        sys.exit(1)

def extract_problem_and_tests(item: Dict[str, Any], target_round: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Extract problem description and test cases from a data item.
    
    Args:
        item: Raw data item from the dataset
        target_round: Specific round to extract (None for latest/best available)
    
    Returns:
        Dictionary with 'problem', 'tests', and metadata, or None if invalid
    """
    try:
        # Extract basic information
        problem_id = item.get('id')
        # Extract synthesis_result (contains generated problems and tests)
        synthesis_result = item.get('synthesis_result', {})
        if isinstance(synthesis_result, str):
            try:
                synthesis_result = json.loads(synthesis_result)
            except:
                synthesis_result = {}
        
        # Use generated problem from synthesis_result (transformed complex problem)
        # Fallback to original problem if synthesis_result doesn't have one
        generated_problem = synthesis_result.get('problem', '').strip()
        original_problem = item.get('problem', '').strip()
        problem_description = generated_problem if generated_problem else original_problem
        
        if not problem_description:
            logger.warning(f"⚠️  Skipping item {problem_id}: No problem description")
            return None
        
        # Extract generated test cases from synthesis_result
        # These are the test cases generated along with the transformed problem
        test_cases = synthesis_result.get('tests', [])
        
        if not test_cases or not isinstance(test_cases, list):
            logger.warning(f"⚠️  Skipping item {problem_id}: No valid test cases")
            return None
        
        # Extract metadata
        experiment_round = item.get('experiment_round', 'unknown')
        model_name = item.get('model_name', 'unknown')
        step_type = item.get('step_type', 'unknown')
        
        # Extract gen_result (always needed)
        gen_result = item.get('gen_result', {})
        if isinstance(gen_result, str):
            try:
                gen_result = json.loads(gen_result)
            except:
                gen_result = {}
        
        # Filter by round if specified
        if target_round is not None:
            # Try to extract round from filename or metadata
            source_file = item.get('source_file', '')
            if f'_round{target_round}' not in source_file and target_round != 0:
                return None  # Skip if not the target round
        
        # Create output item with generated content
        output_item = {
            'id': problem_id,
            'problem': problem_description,  # Generated complex problem from synthesis_result
            'tests': test_cases,             # Generated test cases from synthesis_result
            'metadata': {
                'experiment_round': experiment_round,
                'model_name': model_name,
                'step_type': step_type,
                'source_file': item.get('source_file', ''),
                'num_tests': len(test_cases),
                'content_type': 'generated' if generated_problem else 'original',  # Track content source
                'has_generated_problem': bool(generated_problem),
                'has_synthesis_result': bool(synthesis_result)
            }
        }
        
        # Add program information if available (optional)
        programs = item.get('program', [])
        if programs:
            if isinstance(programs, str):
                programs = [programs]
            output_item['metadata']['num_programs'] = len(programs)
        
        # Add evaluation statistics if available
        if gen_result and 'eval_results' in gen_result:
            eval_results = gen_result['eval_results']
            if isinstance(eval_results, list) and eval_results:
                pass_rates = [result.get('pass_rate', 0) for result in eval_results]
                output_item['metadata']['eval_stats'] = {
                    'num_programs_evaluated': len(eval_results),
                    'avg_pass_rate': sum(pass_rates) / len(pass_rates) if pass_rates else 0,
                    'max_pass_rate': max(pass_rates) if pass_rates else 0
                }
        
        return output_item
        
    except Exception as e:
        logger.warning(f"⚠️  Error processing item {item.get('id', 'unknown')}: {e}")
        return None

def filter_and_convert_dataset(
    data: List[Dict], 
    target_round: Optional[int] = None,
    min_tests: int = 1,
    deduplicate: bool = True
) -> List[Dict]:
    """
    Filter and convert dataset to simplified format.
    
    Args:
        data: Raw dataset items
        target_round: Specific round to extract (None for all)
        min_tests: Minimum number of test cases required
        deduplicate: Whether to merge duplicate problems or keep all separately
    
    Returns:
        List of converted items
    """
    logger.info(f"🔄 Converting dataset...")
    logger.info(f"   Target round: {'All' if target_round is None else target_round}")
    logger.info(f"   Min tests: {min_tests}")
    logger.info(f"   Deduplicate: {'Merge' if deduplicate else 'Keep All'}")
    
    # First pass: convert all items
    all_converted = []
    for i, item in enumerate(data):
        converted = extract_problem_and_tests(item, target_round)
        if converted is not None:
            all_converted.append(converted)
        
        if (i + 1) % 100 == 0:
            logger.info(f"   Processed {i + 1}/{len(data)} items")
    
    if not deduplicate:
        # No deduplication: filter by min_tests and return
        final_items = []
        for item in all_converted:
            if len(item['tests']) >= min_tests:
                final_items.append(item)
        
        logger.info(f"✅ Converted {len(final_items)} out of {len(data)} items (no merging)")
        return final_items
    
    # Deduplication with merging: group by problem content
    problem_groups = {}
    
    for item in all_converted:
        problem_content = item['problem'].strip()
        
        if problem_content not in problem_groups:
            problem_groups[problem_content] = []
        
        problem_groups[problem_content].append(item)
    
    # Merge items for each unique problem
    merged_items = []
    
    for problem_content, items in problem_groups.items():
        if len(items) == 1:
            # Single item, no merging needed
            merged_item = items[0]
        else:
            # Multiple items, merge them
            logger.info(f"🔗 Merging {len(items)} items for problem ID {items[0]['id']}")
            merged_item = merge_problem_items(items)
        
        # Check minimum test requirement after merging
        if len(merged_item['tests']) >= min_tests:
            merged_items.append(merged_item)
        else:
            logger.debug(f"Skipping merged item {merged_item['id']}: Only {len(merged_item['tests'])} tests (min: {min_tests})")
    
    logger.info(f"✅ Converted and merged {len(merged_items)} unique problems from {len(data)} items")
    return merged_items

def merge_problem_items(items: List[Dict]) -> Dict:
    """
    Merge multiple items of the same problem from different rounds.
    
    Args:
        items: List of converted items for the same problem
        
    Returns:
        Merged item with combined test cases and metadata
    """
    if not items:
        raise ValueError("Cannot merge empty items list")
    
    # Use the first item as base
    merged = items[0].copy()
    
    # Collect all unique test cases
    all_tests = set()
    rounds_info = []
    total_programs = 0
    all_eval_stats = []
    
    for item in items:
        # Collect tests (remove duplicates)
        for test in item['tests']:
            all_tests.add(test.strip())
        
        # Collect round information
        round_info = {
            'round': item['metadata'].get('source_file', '').split('round')[-1].split('.')[0] if 'round' in item['metadata'].get('source_file', '') else 'unknown',
            'num_tests': len(item['tests']),
            'num_programs': item['metadata'].get('num_programs', 0)
        }
        rounds_info.append(round_info)
        
        # Collect program counts
        total_programs += item['metadata'].get('num_programs', 0)
        
        # Collect evaluation stats
        if 'eval_stats' in item['metadata']:
            all_eval_stats.append(item['metadata']['eval_stats'])
    
    # Update merged item
    merged['tests'] = sorted(list(all_tests))  # Sort for consistency
    
    # Update metadata with merge information
    merged['metadata']['num_tests'] = len(merged['tests'])
    merged['metadata']['total_programs_across_rounds'] = total_programs
    merged['metadata']['rounds_merged'] = len(items)
    merged['metadata']['rounds_info'] = rounds_info
    
    # Average evaluation stats if available
    if all_eval_stats:
        avg_stats = {
            'num_programs_evaluated': sum(s['num_programs_evaluated'] for s in all_eval_stats) / len(all_eval_stats),
            'avg_pass_rate': sum(s['avg_pass_rate'] for s in all_eval_stats) / len(all_eval_stats),
            'max_pass_rate': max(s['max_pass_rate'] for s in all_eval_stats)
        }
        merged['metadata']['eval_stats'] = avg_stats
    
    return merged

def save_converted_dataset(data: List[Dict], output_path: str, format_type: str = "jsonl"):
    """Save converted dataset to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"💾 Saving {len(data)} items to {output_path}")
    
    if format_type == "jsonl":
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    elif format_type == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    logger.info(f"✅ Dataset saved successfully")

def print_dataset_stats(data: List[Dict]):
    """Print statistics about the converted dataset."""
    if not data:
        logger.warning("No data to analyze")
        return
    
    print("\n" + "="*50)
    print("📊 DATASET STATISTICS")
    print("="*50)
    
    # Basic stats
    print(f"Total items: {len(data)}")
    
    # Test case statistics
    test_counts = [len(item['tests']) for item in data]
    print(f"Test cases per problem:")
    print(f"  Average: {sum(test_counts) / len(test_counts):.1f}")
    print(f"  Min: {min(test_counts)}")
    print(f"  Max: {max(test_counts)}")
    
    # Round distribution
    rounds = [item['metadata']['experiment_round'] for item in data]
    round_counts = {}
    for round_name in rounds:
        round_counts[round_name] = round_counts.get(round_name, 0) + 1
    
    print(f"Distribution by experiment round:")
    for round_name, count in sorted(round_counts.items()):
        print(f"  {round_name}: {count} items")
    
    # Model distribution
    models = [item['metadata']['model_name'] for item in data]
    model_counts = {}
    for model in models:
        model_counts[model] = model_counts.get(model, 0) + 1
    
    print(f"Distribution by model:")
    for model, count in sorted(model_counts.items()):
        print(f"  {model}: {count} items")
    
    # Content type statistics
    content_types = [item['metadata']['content_type'] for item in data]
    content_counts = {}
    for content_type in content_types:
        content_counts[content_type] = content_counts.get(content_type, 0) + 1
    
    print(f"Content type distribution:")
    for content_type, count in sorted(content_counts.items()):
        print(f"  {content_type}: {count} items")
    
    # Generated content statistics
    generated_problems = sum(1 for item in data if item['metadata']['has_generated_problem'])
    print(f"Generated problems: {generated_problems}/{len(data)} ({generated_problems/len(data)*100:.1f}%)")
    
    # Sample item
    print(f"\nSample item:")
    sample = data[0]
    print(f"  ID: {sample['id']}")
    print(f"  Problem type: {sample['metadata']['content_type']}")
    print(f"  Problem: {sample['problem'][:100]}...")
    print(f"  Tests: {len(sample['tests'])} test cases")
    print(f"  First test: {sample['tests'][0][:80]}...")
    
    print("="*50)

def main():
    parser = argparse.ArgumentParser(
        description="Convert AceCoderV2 dataset to simplified format with problems and test cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load from Hugging Face and convert all rounds
  python hf_dataset_converter.py --dataset_name siyiwu0330/acecoderv2 --output converted.jsonl
  
  # Load from local file and filter specific round
  python hf_dataset_converter.py --local_file data.jsonl --output round3.jsonl --round 3
  
  # Convert with custom settings
  python hf_dataset_converter.py --dataset_name siyiwu0330/acecoderv2 --output filtered.jsonl --min_tests 5 --no_deduplicate
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--dataset_name", type=str, help="Hugging Face dataset name (e.g., siyiwu0330/acecoderv2)")
    input_group.add_argument("--local_file", type=str, help="Local file path (.json or .jsonl)")
    
    # Output options
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--format", type=str, choices=["json", "jsonl"], default="jsonl", help="Output format (default: jsonl)")
    
    # Filtering options
    parser.add_argument("--round", type=int, help="Extract specific round only (e.g., 0, 1, 2, ...)")
    parser.add_argument("--min_tests", type=int, default=1, help="Minimum number of test cases required (default: 1)")
    parser.add_argument("--no_deduplicate", action="store_true", help="Don't remove duplicate problems")
    
    # Other options
    parser.add_argument("--split", type=str, default="train", help="Dataset split for HF datasets (default: train)")
    parser.add_argument("--stats", action="store_true", help="Print detailed statistics")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load data
    if args.dataset_name:
        data = load_from_huggingface(args.dataset_name, args.split)
    else:
        data = load_from_local_file(args.local_file)
    
    # Convert data
    converted_data = filter_and_convert_dataset(
        data,
        target_round=args.round,
        min_tests=args.min_tests,
        deduplicate=not args.no_deduplicate
    )
    
    if not converted_data:
        logger.error("❌ No valid items found after conversion")
        sys.exit(1)
    
    # Save converted data
    save_converted_dataset(converted_data, args.output, args.format)
    
    # Print statistics
    if args.stats:
        print_dataset_stats(converted_data)
    
    print(f"\n🎉 Conversion completed successfully!")
    print(f"   Input: {len(data)} items")
    print(f"   Output: {len(converted_data)} items")
    print(f"   Saved to: {args.output}")

if __name__ == "__main__":
    main()
