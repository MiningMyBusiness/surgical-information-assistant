import json
import argparse
from typing import Dict, List, Any
from collections import defaultdict, Counter
import statistics
from pathlib import Path

def count_words(text: str) -> int:
    """Count words in a text string"""
    if not isinstance(text, str):
        return 0
    return len(text.split())

def analyze_string_fields(data: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """Analyze word counts for all string fields in the dataset"""
    field_stats = defaultdict(list)
    
    # Collect word counts for each string field
    for item in data:
        for key, value in item.items():
            if isinstance(value, str):
                word_count = count_words(value)
                field_stats[key].append(word_count)
    
    # Calculate statistics for each field
    results = {}
    for field, word_counts in field_stats.items():
        if word_counts:  # Only process fields that have data
            results[field] = {
                'total_entries': len(word_counts),
                'total_words': sum(word_counts),
                'mean_words': statistics.mean(word_counts),
                'median_words': statistics.median(word_counts),
                'min_words': min(word_counts),
                'max_words': max(word_counts),
                'std_dev': statistics.stdev(word_counts) if len(word_counts) > 1 else 0,
                'word_count_distribution': Counter(word_counts)
            }
    
    return results

def print_summary_statistics(results: Dict[str, Dict[str, Any]], total_records: int):
    """Print formatted summary statistics"""
    print("=" * 80)
    print(f"JSON FILE WORD COUNT ANALYSIS")
    print("=" * 80)
    print(f"Total records in file: {total_records}")
    print(f"String fields found: {len(results)}")
    print()
    
    for field, stats in results.items():
        print(f"📝 FIELD: {field}")
        print("-" * 50)
        print(f"  Records with this field: {stats['total_entries']}")
        print(f"  Total words across all entries: {stats['total_words']:,}")
        print(f"  Average words per entry: {stats['mean_words']:.2f}")
        print(f"  Median words per entry: {stats['median_words']:.2f}")
        print(f"  Min words in an entry: {stats['min_words']}")
        print(f"  Max words in an entry: {stats['max_words']}")
        print(f"  Standard deviation: {stats['std_dev']:.2f}")
        
        # Show top 5 most common word counts
        top_counts = stats['word_count_distribution'].most_common(5)
        if top_counts:
            print(f"  Most common word counts:")
            for word_count, frequency in top_counts:
                print(f"    {word_count} words: {frequency} entries")
        print()

def save_detailed_report(results: Dict[str, Dict[str, Any]], output_file: str, total_records: int):
    """Save detailed analysis to a JSON file"""
    report = {
        'analysis_summary': {
            'total_records': total_records,
            'string_fields_analyzed': len(results),
            'analysis_timestamp': None  # Could add timestamp if needed
        },
        'field_statistics': {}
    }
    
    for field, stats in results.items():
        # Convert Counter to regular dict for JSON serialization
        word_count_dist = dict(stats['word_count_distribution'])
        
        report['field_statistics'][field] = {
            'total_entries': stats['total_entries'],
            'total_words': stats['total_words'],
            'mean_words': round(stats['mean_words'], 2),
            'median_words': stats['median_words'],
            'min_words': stats['min_words'],
            'max_words': stats['max_words'],
            'std_dev': round(stats['std_dev'], 2),
            'word_count_distribution': word_count_dist
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Detailed report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze word counts for string fields in a JSON file containing a list of dictionaries"
    )
    parser.add_argument(
        'input_file',
        help='Path to the input JSON file'
    )
    parser.add_argument(
        '--output',
        '-o',
        help='Path to save detailed analysis report (optional)',
        default=None
    )
    parser.add_argument(
        '--quiet',
        '-q',
        action='store_true',
        help="Only save report, don't print to console"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Error: File '{args.input_file}' not found")
        return
    
    # Load JSON data
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON file - {e}")
        return
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # Validate data structure
    if not isinstance(data, list):
        print("❌ Error: JSON file must contain a list of dictionaries")
        return
    
    if not data:
        print("❌ Error: JSON file contains an empty list")
        return
    
    if not all(isinstance(item, dict) for item in data):
        print("❌ Error: All items in the list must be dictionaries")
        return
    
    # Analyze the data
    print("🔍 Analyzing word counts...")
    results = analyze_string_fields(data)
    
    if not results:
        print("❌ No string fields found in the data")
        return
    
    # Print results to console (unless quiet mode)
    if not args.quiet:
        print_summary_statistics(results, len(data))
    
    # Save detailed report if requested
    if args.output:
        save_detailed_report(results, args.output, len(data))
    
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()