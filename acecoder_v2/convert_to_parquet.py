# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Convert AceCoderV2 dataset from JSONL to parquet format
Based on hf_dataset_converter.py logic for data extraction and processing
"""
import fire
import os
from pathlib import Path
import random
import json
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a JSON Lines (.jsonl) file.

    Args:
        file_path (str): Path to the .jsonl file.

    Returns:
        list: A list of dicts, where each dict is one JSON object from a line.
    """
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                        continue
        logger.info(f"Successfully loaded {len(data)} records from {file_path}")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise
    
    return data

def extract_problem_and_tests(item: Dict[str, Any], target_round: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Extract problem description and test cases from a data item.
    This function mirrors the logic from hf_dataset_converter.py
    
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
    This function mirrors the logic from hf_dataset_converter.py
    
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
    This function mirrors the logic from hf_dataset_converter.py
    
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

def process_for_parquet(item: Dict[str, Any], instruction_template: str = None) -> Dict[str, Any]:
    """
    Process a converted item for parquet format.
    
    Args:
        item: Converted item from extract_problem_and_tests
        instruction_template: Custom instruction template
        
    Returns:
        Dictionary formatted for parquet output
    """
    # Use provided instruction or default
    if instruction_template is None:
        instruction_template = "Let's think step by step and generate the final program in a markdown code block like this: ```python\nyour code here\n```."
    
    # Create the prompt
    prompt_content = item['problem']
    if instruction_template:
        prompt_content += "\n\n" + instruction_template
    
    # Format for parquet (similar to original structure)
    parquet_item = {
        "data_source": "acecoderv2",
        "prompt": [
            {
                "role": "user",
                "content": prompt_content
            }
        ],
        "ability": "code",
        "reward_model": {
            "style": "rule",
            "ground_truth": ""
        },
        "extra_info": {
            'split': 'unknown',  # Will be set later
            'index': 0,          # Will be set later
            'id': str(item['id']),
            "question": item['problem'],
            "test_cases": item['tests'],
            "inputs_outputs": None,
            "metadata": item['metadata']  # Preserve all metadata
        }
    }
    
    return parquet_item

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

def main(
    jsonl_path: str = 'problems_merged.jsonl',
    local_dir: str = 'data/acecoder',
    test_size: int = 500,
    random_seed: int = 69,
    instruction_template: str = None,
    target_round: int = None,
    min_tests: int = 1,
    deduplicate: bool = True,
    stats: bool = True
):
    """
    Convert JSONL dataset to parquet format using hf_dataset_converter.py logic.
    
    Args:
        jsonl_path (str): Path to input JSONL file
        local_dir (str): Output directory for parquet files
        test_size (int): Number of examples for test set
        random_seed (int): Random seed for reproducibility
        instruction_template (str): Custom instruction template (optional)
        target_round (int): Specific round to extract (None for all)
        min_tests (int): Minimum number of test cases required
        deduplicate (bool): Whether to deduplicate problems
        stats (bool): Whether to print statistics
    """
    logger.info(f"Starting conversion from {jsonl_path} to parquet format")
    
    # Setup paths
    local_dir = Path(local_dir)
    local_dir_post_fix = ""
    local_dir = local_dir / (jsonl_path.split('.')[-2] + local_dir_post_fix)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = load_jsonl(jsonl_path)
    
    if not dataset:
        logger.error("No data loaded from JSONL file")
        return
    
    # Convert data using hf_dataset_converter logic
    converted_data = filter_and_convert_dataset(
        dataset,
        target_round=target_round,
        min_tests=min_tests,
        deduplicate=deduplicate
    )
    
    if not converted_data:
        logger.error("❌ No valid items found after conversion")
        return
    
    # Split dataset
    random.seed(random_seed)
    random.shuffle(converted_data)
    
    if len(converted_data) <= test_size:
        logger.warning(f"Dataset size ({len(converted_data)}) is smaller than test_size ({test_size})")
        test_data = converted_data
        train_data = []
    else:
        test_data = converted_data[:test_size]
        train_data = converted_data[test_size:]
    
    logger.info(f"Split: {len(train_data)} train, {len(test_data)} test")
    
    # Process datasets for parquet format
    logger.info("Processing training dataset...")
    train_processed = []
    for i, item in enumerate(train_data):
        try:
            processed = process_for_parquet(item, instruction_template)
            processed['extra_info']['split'] = 'train'
            processed['extra_info']['index'] = i
            train_processed.append(processed)
        except Exception as e:
            logger.error(f"Error processing train entry {i}: {e}")
            continue
    
    logger.info("Processing test dataset...")
    test_processed = []
    for i, item in enumerate(test_data):
        try:
            processed = process_for_parquet(item, instruction_template)
            processed['extra_info']['split'] = 'test'
            processed['extra_info']['index'] = i
            test_processed.append(processed)
        except Exception as e:
            logger.error(f"Error processing test entry {i}: {e}")
            continue
    
    # Convert to DataFrames
    logger.info("Converting to DataFrames...")
    train_df = pd.DataFrame(train_processed)
    test_df = pd.DataFrame(test_processed)
    
    # Save to parquet
    train_path = local_dir / 'train.parquet'
    test_path = local_dir / 'test.parquet'
    
    logger.info(f"Saving training data to {train_path}")
    train_df.to_parquet(train_path, index=False)
    
    logger.info(f"Saving test data to {test_path}")
    test_df.to_parquet(test_path, index=False)
    
    # Print summary
    print(f"\n✅ Conversion completed!")
    print(f"📊 Training samples: {len(train_df)}")
    print(f"📊 Test samples: {len(test_df)}")
    print(f"📁 Output directory: {local_dir}")
    print(f"📄 Train file: {train_path}")
    print(f"📄 Test file: {test_path}")
    
    # Show example
    if len(train_df) > 0:
        print(f"\n📝 Example training sample:")
        example = train_df.iloc[0]
        print(f"  ID: {example['extra_info']['id']}")
        print(f"  Question: {example['extra_info']['question'][:100]}...")
        print(f"  Test cases: {len(example['extra_info']['test_cases'])}")
    
    # Print statistics if requested
    if stats:
        print_dataset_stats(converted_data)
    
    # Save metadata
    metadata = {
        "source_file": jsonl_path,
        "total_samples": len(dataset),
        "converted_samples": len(converted_data),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "test_size": test_size,
        "random_seed": random_seed,
        "instruction_template": instruction_template,
        "target_round": target_round,
        "min_tests": min_tests,
        "deduplicate": deduplicate
    }
    
    metadata_path = local_dir / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Metadata saved to {metadata_path}")

if __name__ == '__main__':
    fire.Fire(main)