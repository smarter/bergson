#!/usr/bin/env python
"""Extract primary scores from olmes metrics.json to a simpler format for DVC.

Converts from:
    {"tasks": [{"alias": "gsm8k::olmes", "metrics": {"primary_score": 0.674}}]}

To:
    {"gsm8k::olmes": 0.674}

Usage:
    python extract_olmes_metrics.py input.json output.json
"""

import argparse
import json


def extract_primary_scores(input_path: str, output_path: str) -> None:
    with open(input_path) as f:
        data = json.load(f)

    scores = {task["alias"]: task["metrics"]["primary_score"] for task in data["tasks"]}

    with open(output_path, "w") as f:
        json.dump(scores, f)


def main():
    parser = argparse.ArgumentParser(description="Extract primary scores from olmes metrics")
    parser.add_argument("input", help="Path to olmes metrics.json")
    parser.add_argument("output", help="Path to output JSON file")
    args = parser.parse_args()

    extract_primary_scores(args.input, args.output)


if __name__ == "__main__":
    main()
