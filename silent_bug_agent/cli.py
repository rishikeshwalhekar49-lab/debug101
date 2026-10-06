"""Command Line Interface for silent_bug_agent.

Usage:
    python -m silent_bug_agent scan <path> [--semantic] [--json]
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .findings import Report, merge_reports
from .invariants.base import Severity
from .semantic import AnthropicClient, SemanticAnalyzer
from .static import StaticAnalyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="silent-bug-agent",
        description="Detect silent bugs in deep learning training code.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # scan subcommand
    scan_parser = subparsers.add_parser("scan", help="Scan a file or directory for silent bugs")
    scan_parser.add_argument(
        "path",
        type=str,
        help="Path to a Python file or directory containing training code",
    )
    scan_parser.add_argument(
        "--semantic",
        action="store_true",
        default=False,
        help="Run semantic LLM analysis in addition to AST static analysis (requires ANTHROPIC_API_KEY)",
    )
    scan_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output the final report as JSON",
    )
    scan_parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="Anthropic model to use when --semantic is enabled",
    )

    return parser


def run_scan(path_str: str, semantic: bool = False, json_output: bool = False, model: str = "claude-3-5-sonnet-20241022") -> int:
    target = Path(path_str).resolve()
    if not target.exists():
        print(f"Error: Target path '{path_str}' does not exist.", file=sys.stderr)
        return 2

    # 1. Run Static Analyzer
    static_analyzer = StaticAnalyzer()
    if target.is_file():
        report = static_analyzer.analyze_file(target)
    else:
        report = static_analyzer.analyze_directory(target)

    # 2. Run Semantic Analyzer if requested
    if semantic:
        try:
            client = AnthropicClient(model=model)
            semantic_analyzer = SemanticAnalyzer(client=client)
            if target.is_file():
                sem_report = semantic_analyzer.analyze_file(target)
            else:
                sem_report = semantic_analyzer.analyze_directory(target)
            # Deduplicate against static
            report = merge_reports(report, sem_report)
        except Exception as e:
            report.diagnostics.append(f"Semantic analysis failed: {e}")

    # 3. Output report
    if json_output:
        print(report.to_json(indent=2))
    else:
        print(report.format_text())

    # Exit code: return 1 if there are any WARNING or CRITICAL findings
    has_issues = any(f.severity in (Severity.WARNING, Severity.CRITICAL) for f in report.findings)
    return 1 if has_issues else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "scan":
        return run_scan(
            path_str=args.path,
            semantic=args.semantic,
            json_output=args.json,
            model=getattr(args, "model", "claude-3-5-sonnet-20241022"),
        )
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
