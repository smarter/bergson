#!/usr/bin/env python
"""
Validate experimental results against expected values from the paper.

Computes absolute and relative errors for all metrics reported in the paper.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple


# Expected values from the paper (Table 1 for 7B, Table 5/Appendix F for 1B)
EXPECTED_VALUES = {
    "7b": {
        "baseline": {
            "dolma_strict_acc": 0.999,
            "dolma_loose_acc": 1.000,
            "dolma_avg_lev": 0.002,
            "quotes_strict_acc": 0.999,
            "quotes_loose_acc": 1.000,
            "quotes_avg_lev": 0.001,
            "perplexity": 19.04,
        },
        "kfac": {
            "dolma_strict_acc": 0.034,
            "dolma_loose_acc": 0.088,
            "dolma_avg_lev": 0.704,
            "quotes_strict_acc": 0.161,
            "quotes_loose_acc": 0.238,
            "quotes_avg_lev": 0.625,
            "perplexity": 22.84,
            "ndcg": 0.91,
        },
    },
    "1b": {
        "baseline": {
            "dolma_strict_acc": 0.9846,
            "dolma_loose_acc": 0.9938,
            "dolma_avg_lev": 0.005,
            "quotes_strict_acc": 0.985,
            "quotes_loose_acc": 0.9895,
            "quotes_avg_lev": 0.006,
            "perplexity": 23.19,
        },
        "kfac": {
            "dolma_strict_acc": 0.028,
            "dolma_loose_acc": 0.072,
            "dolma_avg_lev": 0.761,
            "quotes_strict_acc": 0.277,
            "quotes_loose_acc": 0.399,
            "quotes_avg_lev": 0.470,
            "perplexity": 26.53,
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate experimental results against paper values"
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        required=True,
        help="Path to results JSON from eval_mem_kfac.py",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["1b", "7b"],
        required=True,
        help="Model size (1b or 7b)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for validation metrics JSON",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Relative error tolerance for flagging issues (default: 0.05 = 5%%)",
    )
    return parser.parse_args()


def extract_metrics_from_results(results: Dict[str, Any]) -> Dict[str, float]:
    """Extract relevant metrics from eval_mem_kfac.py output.

    Handles both nested JSON format and flattened JSONL format.
    """
    metrics = {}

    # Check if this is flattened JSONL format (with kfac_ prefix)
    if "kfac_mem_strict_acc" in results:
        # Flattened JSONL format
        metrics["dolma_strict_acc"] = results.get("kfac_mem_strict_acc", None)
        metrics["dolma_loose_acc"] = results.get("kfac_mem_loose_acc", None)
        metrics["dolma_avg_lev"] = results.get("kfac_mem_avg_levenshtein_norm", None)

        metrics["quotes_strict_acc"] = results.get("kfac_quotes_strict_acc", None)
        metrics["quotes_loose_acc"] = results.get("kfac_quotes_loose_acc", None)
        metrics["quotes_avg_lev"] = results.get("kfac_quotes_avg_levenshtein_norm", None)

        # Perplexity - prefer BSN-style if available
        if "kfac_perplexity_bsn_post" in results:
            metrics["perplexity"] = results["kfac_perplexity_bsn_post"]
        elif "kfac_perplexity" in results:
            metrics["perplexity"] = results["kfac_perplexity"]

        # nDCG@10
        if "kfac_ndcg@10" in results:
            metrics["ndcg"] = results["kfac_ndcg@10"]
    else:
        # Nested JSON format (legacy)
        # Memorization metrics (Dolma dataset)
        if "memorization" in results:
            mem = results["memorization"]
            metrics["dolma_strict_acc"] = mem.get("strict_acc", None)
            metrics["dolma_loose_acc"] = mem.get("loose_acc", None)
            metrics["dolma_avg_lev"] = mem.get("avg_levenshtein_norm", None)

        # Quotes metrics
        if "quotes" in results:
            quotes = results["quotes"]
            metrics["quotes_strict_acc"] = quotes.get("strict_acc", None)
            metrics["quotes_loose_acc"] = quotes.get("loose_acc", None)
            metrics["quotes_avg_lev"] = quotes.get("avg_levenshtein_norm", None)

        # Perplexity
        if "kfac_perplexity_bsn_post" in results:
            metrics["perplexity"] = results["kfac_perplexity_bsn_post"]
        elif "perplexity" in results:
            metrics["perplexity"] = results["perplexity"]

        # nDCG@10
        if "ndcg" in results:
            metrics["ndcg"] = results["ndcg"]

    return metrics


def compute_errors(
    actual: float, expected: float
) -> Tuple[float, float]:
    """Compute absolute and relative errors."""
    abs_error = actual - expected

    # Handle division by zero for relative error
    if abs(expected) < 1e-10:
        rel_error = float('inf') if abs(abs_error) > 1e-10 else 0.0
    else:
        rel_error = abs_error / expected

    return abs_error, rel_error


def validate_results(
    actual_metrics: Dict[str, float],
    expected_metrics: Dict[str, float],
    tolerance: float,
) -> Dict[str, Any]:
    """Compare actual vs expected metrics and compute errors."""
    validation = {
        "metrics": {},
        "summary": {
            "total_metrics": 0,
            "missing_metrics": 0,
            "within_tolerance": 0,
            "outside_tolerance": 0,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "avg_abs_error": 0.0,
            "avg_rel_error": 0.0,
        },
        "issues": [],
    }

    abs_errors = []
    rel_errors = []

    for metric_name, expected_value in expected_metrics.items():
        validation["summary"]["total_metrics"] += 1

        if metric_name not in actual_metrics or actual_metrics[metric_name] is None:
            validation["summary"]["missing_metrics"] += 1
            validation["issues"].append({
                "metric": metric_name,
                "issue": "missing",
                "expected": expected_value,
            })
            continue

        actual_value = actual_metrics[metric_name]
        abs_error, rel_error = compute_errors(actual_value, expected_value)

        abs_errors.append(abs(abs_error))
        rel_errors.append(abs(rel_error))

        metric_validation = {
            "expected": expected_value,
            "actual": actual_value,
            "absolute_error": abs_error,
            "relative_error": rel_error,
            "relative_error_pct": rel_error * 100,
            "within_tolerance": abs(rel_error) <= tolerance,
        }

        validation["metrics"][metric_name] = metric_validation

        if abs(rel_error) <= tolerance:
            validation["summary"]["within_tolerance"] += 1
        else:
            validation["summary"]["outside_tolerance"] += 1
            validation["issues"].append({
                "metric": metric_name,
                "issue": "outside_tolerance",
                "expected": expected_value,
                "actual": actual_value,
                "relative_error": rel_error,
                "relative_error_pct": rel_error * 100,
            })

    # Compute summary statistics
    if abs_errors:
        validation["summary"]["max_abs_error"] = max(abs_errors)
        validation["summary"]["avg_abs_error"] = sum(abs_errors) / len(abs_errors)

    if rel_errors:
        validation["summary"]["max_rel_error"] = max(rel_errors)
        validation["summary"]["avg_rel_error"] = sum(rel_errors) / len(rel_errors)
        validation["summary"]["avg_rel_error_pct"] = validation["summary"]["avg_rel_error"] * 100

    return validation


def print_validation_report(validation: Dict[str, Any], model_size: str):
    """Print human-readable validation report."""
    print("\n" + "=" * 70)
    print(f"VALIDATION REPORT: OLMo-2 {model_size.upper()} vs Paper Results")
    print("=" * 70)

    summary = validation["summary"]
    print(f"\nTotal metrics: {summary['total_metrics']}")
    print(f"Within tolerance: {summary['within_tolerance']} / {summary['total_metrics']}")
    print(f"Outside tolerance: {summary['outside_tolerance']} / {summary['total_metrics']}")
    print(f"Missing metrics: {summary['missing_metrics']} / {summary['total_metrics']}")

    print(f"\nError Statistics:")
    print(f"  Max absolute error: {summary['max_abs_error']:.6f}")
    print(f"  Avg absolute error: {summary['avg_abs_error']:.6f}")
    print(f"  Max relative error: {summary['max_rel_error'] * 100:.2f}%")
    print(f"  Avg relative error: {summary.get('avg_rel_error_pct', 0):.2f}%")

    # Print per-metric details
    print("\n" + "-" * 70)
    print("Per-Metric Comparison:")
    print("-" * 70)

    for metric_name, metric_data in validation["metrics"].items():
        status = "✓" if metric_data["within_tolerance"] else "✗"
        print(f"\n{status} {metric_name}:")
        print(f"    Expected: {metric_data['expected']:.6f}")
        print(f"    Actual:   {metric_data['actual']:.6f}")
        print(f"    Abs Error: {metric_data['absolute_error']:+.6f}")
        print(f"    Rel Error: {metric_data['relative_error_pct']:+.2f}%")

    # Print issues if any
    if validation["issues"]:
        print("\n" + "-" * 70)
        print("⚠ ISSUES FOUND:")
        print("-" * 70)
        for issue in validation["issues"]:
            if issue["issue"] == "missing":
                print(f"  • {issue['metric']}: MISSING (expected {issue['expected']})")
            else:
                print(f"  • {issue['metric']}: {issue['relative_error_pct']:+.2f}% error "
                      f"(expected {issue['expected']}, got {issue['actual']})")
    else:
        print("\n✓ All metrics within tolerance!")

    print("\n" + "=" * 70)


def main():
    args = parse_args()

    # Load results
    if not args.results_file.exists():
        print(f"Error: Results file not found: {args.results_file}", file=sys.stderr)
        sys.exit(1)

    # Handle both JSON and JSONL formats
    if args.results_file.suffix == ".jsonl":
        # Read last line from JSONL file
        with open(args.results_file, "r") as f:
            lines = [line for line in f if line.strip()]
            if not lines:
                print(f"Error: Results file is empty: {args.results_file}", file=sys.stderr)
                sys.exit(1)
            results = json.loads(lines[-1])
    else:
        # Read JSON file
        with open(args.results_file, "r") as f:
            results = json.load(f)

    # Determine whether this is baseline or K-FAC results
    # K-FAC results will have layer_config/layers, baseline won't
    is_kfac = ("layer_config" in results and results["layer_config"]) or \
              ("layers" in results and results["layers"])
    result_type = "kfac" if is_kfac else "baseline"

    print(f"Validating {result_type.upper()} results for {args.model_size.upper()} model")

    # Get expected values
    if result_type not in EXPECTED_VALUES[args.model_size]:
        print(f"Warning: No expected values for {result_type} in {args.model_size} model")
        expected_metrics = {}
    else:
        expected_metrics = EXPECTED_VALUES[args.model_size][result_type]

    # Extract actual metrics
    actual_metrics = extract_metrics_from_results(results)

    # Validate
    validation = validate_results(actual_metrics, expected_metrics, args.tolerance)

    # Add metadata
    validation["metadata"] = {
        "model_size": args.model_size,
        "result_type": result_type,
        "results_file": str(args.results_file),
        "tolerance": args.tolerance,
    }

    # Print report
    print_validation_report(validation, args.model_size)

    # Save validation results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(validation, f, indent=2)

    print(f"\n✓ Validation metrics saved to: {args.output}")

    # Exit with error code if issues found
    if validation["issues"]:
        print(f"\n⚠ Validation found {len(validation['issues'])} issue(s)")
        sys.exit(1)
    else:
        print("\n✓ All validations passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
