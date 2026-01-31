#!/usr/bin/env python
"""
Generate arithmetic dataset for fine-tuning.

Generates JSONL files with arithmetic expressions for training models
to recover math abilities after memorization removal.
"""
import argparse
import itertools
import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ArithmeticConfig:
    """Configuration for arithmetic dataset generation."""

    operations: list[str] = field(default_factory=lambda: ["+", "-", "*", "/"])
    num_operands: int = 2  # 2 = binary, 3 = chained with parens
    min_digits: int = 1
    max_digits: int = 3
    sample_percentage: float = 1.0
    total_samples: int | None = None
    output_format: str = "equation"  # "equation", "qa", "instruct"
    allow_negative: bool = True
    division_mode: str = "integer"  # "integer", "exact", "rounded"
    seed: int | None = None


def random_number(min_digits: int, max_digits: int, allow_negative: bool, rng: random.Random) -> int:
    """Generate a random number with the specified digit range."""
    digits = rng.randint(min_digits, max_digits)
    min_val = 10 ** (digits - 1) if digits > 1 else 0
    max_val = 10**digits - 1
    value = rng.randint(min_val, max_val)
    if allow_negative and rng.random() < 0.3:  # 30% chance of negative
        value = -value
    return value


def compute_result(expression: str, division_mode: str) -> int | float | None:
    """Compute the result of an expression, handling division modes."""
    try:
        result = eval(expression)  # noqa: S307
    except ZeroDivisionError:
        return None

    if "/" in expression:
        if division_mode == "integer":
            # Only allow clean integer divisions
            if result != int(result):
                return None
            return int(result)
        elif division_mode == "exact":
            # Allow exact decimal results
            return result
        elif division_mode == "rounded":
            # Round to 2 decimal places
            return round(result, 2)
    return int(result) if isinstance(result, float) and result == int(result) else result


def generate_binary_expression(
    operands: tuple[int, int],
    op: str,
    division_mode: str,
) -> tuple[str, int | float] | None:
    """Generate a binary (2-operand) expression."""
    a, b = operands

    # For division, ensure non-zero divisor
    if op == "/" and b == 0:
        return None

    expression = f"{a} {op} {b}"
    result = compute_result(expression, division_mode)
    if result is None:
        return None

    return expression, result


def generate_ternary_expression(
    operands: tuple[int, int, int],
    ops: tuple[str, str],
    use_parens: bool,
    paren_position: str,  # "left" or "right"
    division_mode: str,
) -> tuple[str, int | float] | None:
    """Generate a ternary (3-operand) expression with optional parentheses."""
    a, b, c = operands
    op1, op2 = ops

    # Check for division by zero
    if op1 == "/" and b == 0:
        return None
    if op2 == "/" and c == 0:
        return None

    if use_parens:
        if paren_position == "left":
            # (a op1 b) op2 c
            expression = f"({a} {op1} {b}) {op2} {c}"
        else:
            # a op1 (b op2 c)
            expression = f"{a} {op1} ({b} {op2} {c})"
            # Check inner division by zero
            if op2 == "/" and c == 0:
                return None
    else:
        # Standard precedence: a op1 b op2 c
        expression = f"{a} {op1} {b} {op2} {c}"

    result = compute_result(expression, division_mode)
    if result is None:
        return None

    return expression, result


def format_output(expression: str, result: int | float, output_format: str) -> str | list[dict[str, str]]:
    """Format the expression and result according to the output format."""
    # Format result nicely
    if isinstance(result, float):
        result_str = f"{result:.2f}".rstrip("0").rstrip(".")
    else:
        result_str = str(result)

    if output_format == "equation":
        return f"{expression} = {result_str}"
    elif output_format == "qa":
        return f"Q: What is {expression}?\nA: {result_str}"
    elif output_format == "instruct":
        return f"Calculate: {expression}\nAnswer: {result_str}"
    elif output_format == "chat":
        return [
            {"role": "user", "content": f"Calculate: {expression}"},
            {"role": "assistant", "content": result_str},
        ]
    else:
        raise ValueError(f"Unknown output format: {output_format}")


def generate_all_binary_expressions(config: ArithmeticConfig, rng: random.Random) -> list[str]:
    """Generate all possible binary expressions within the config constraints."""
    expressions = []

    # Generate all numbers in range
    all_numbers = []
    for digits in range(config.min_digits, config.max_digits + 1):
        min_val = 10 ** (digits - 1) if digits > 1 else 0
        max_val = 10**digits - 1
        for n in range(min_val, max_val + 1):
            all_numbers.append(n)
            if config.allow_negative and n != 0:
                all_numbers.append(-n)

    for a, b in itertools.product(all_numbers, repeat=2):
        for op in config.operations:
            result = generate_binary_expression((a, b), op, config.division_mode)
            if result is not None:
                expr, res = result
                expressions.append(format_output(expr, res, config.output_format))

    return expressions


def generate_sampled_expressions(config: ArithmeticConfig, rng: random.Random) -> list:
    """Generate sampled expressions using random sampling."""
    seen_expressions: set[str] = set()  # Track unique expressions for deduplication
    outputs: list = []  # Store formatted outputs (may be strings or lists for chat)
    assert config.total_samples is not None
    target_count = config.total_samples
    max_attempts = target_count * 100  # Prevent infinite loops
    attempts = 0

    while len(outputs) < target_count and attempts < max_attempts:
        attempts += 1

        if config.num_operands == 2:
            a = random_number(config.min_digits, config.max_digits, config.allow_negative, rng)
            b = random_number(config.min_digits, config.max_digits, config.allow_negative, rng)
            op = rng.choice(config.operations)

            result = generate_binary_expression((a, b), op, config.division_mode)
            if result is not None:
                expr, res = result
                if expr not in seen_expressions:
                    seen_expressions.add(expr)
                    outputs.append(format_output(expr, res, config.output_format))

        elif config.num_operands == 3:
            a = random_number(config.min_digits, config.max_digits, config.allow_negative, rng)
            b = random_number(config.min_digits, config.max_digits, config.allow_negative, rng)
            c = random_number(config.min_digits, config.max_digits, config.allow_negative, rng)
            op1 = rng.choice(config.operations)
            op2 = rng.choice(config.operations)
            use_parens = rng.random() < 0.5
            paren_position = rng.choice(["left", "right"])

            result = generate_ternary_expression((a, b, c), (op1, op2), use_parens, paren_position, config.division_mode)
            if result is not None:
                expr, res = result
                if expr not in seen_expressions:
                    seen_expressions.add(expr)
                    outputs.append(format_output(expr, res, config.output_format))

    return outputs


def generate_dataset(config: ArithmeticConfig) -> list:
    """Generate arithmetic dataset according to config."""
    rng = random.Random(config.seed)

    if config.total_samples is not None:
        # Sample-based generation
        return generate_sampled_expressions(config, rng)
    elif config.num_operands == 2 and config.sample_percentage == 1.0:
        # Full enumeration for binary expressions
        return generate_all_binary_expressions(config, rng)
    else:
        # For percentage-based sampling, estimate total and sample
        # This is an approximation
        estimated_total = 10000  # Conservative estimate
        config.total_samples = int(estimated_total * config.sample_percentage)
        return generate_sampled_expressions(config, rng)


def main():
    parser = argparse.ArgumentParser(
        description="Generate arithmetic dataset for fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--operations", type=str, default="+,-,*,/", help="Comma-separated operations")
    parser.add_argument("--num-operands", type=int, default=2, help="Number of operands (2 or 3)")
    parser.add_argument("--min-digits", type=int, default=1, help="Minimum number of digits")
    parser.add_argument("--max-digits", type=int, default=3, help="Maximum number of digits")
    parser.add_argument("--sample-percentage", type=float, default=1.0, help="Percentage of space to sample")
    parser.add_argument("--total-samples", type=int, default=None, help="Total number of samples to generate")
    parser.add_argument("--format", type=str, default="equation", choices=["equation", "qa", "instruct", "chat"])
    parser.add_argument("--division-mode", type=str, default="integer", choices=["integer", "exact", "rounded"])
    parser.add_argument("--allow-negative", action="store_true", default=True, help="Allow negative numbers")
    parser.add_argument("--no-negative", action="store_true", help="Disable negative numbers")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    config = ArithmeticConfig(
        operations=args.operations.split(","),
        num_operands=args.num_operands,
        min_digits=args.min_digits,
        max_digits=args.max_digits,
        sample_percentage=args.sample_percentage,
        total_samples=args.total_samples,
        output_format=args.format,
        allow_negative=not args.no_negative,
        division_mode=args.division_mode,
        seed=args.seed,
    )

    print(f"Generating arithmetic dataset with config:")
    print(f"  Operations: {config.operations}")
    print(f"  Operands: {config.num_operands}")
    print(f"  Digits: {config.min_digits}-{config.max_digits}")
    print(f"  Division mode: {config.division_mode}")
    print(f"  Format: {config.output_format}")
    if config.total_samples:
        print(f"  Target samples: {config.total_samples}")

    expressions = generate_dataset(config)
    random.Random(config.seed).shuffle(expressions)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for expr in expressions:
            if config.output_format == "chat":
                f.write(json.dumps({"messages": expr}) + "\n")
            else:
                f.write(json.dumps({"text": expr}) + "\n")

    print(f"Generated {len(expressions)} expressions to {args.output}")


if __name__ == "__main__":
    main()
