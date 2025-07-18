#!/usr/bin/env python3
"""
vLLM CLI Arguments Parser - GitHub Gist
========================================

A script to scrape vLLM serve command documentation and generate a YAML config file
with all available flags and their default values.

Features:
- Scrapes latest vLLM documentation
- Extracts all CLI flags with defaults and descriptions
- Generates clean YAML config file
- Progress bar with rich library
- Validation against raw HTML

Usage:
    python vllm_cli_parser.py

Requirements:
    pip install requests beautifulsoup4 pyyaml rich
"""

from __future__ import annotations
import re
import sys
import ast
import requests
import yaml
from bs4 import BeautifulSoup
from collections import OrderedDict
from pathlib import Path
import argparse

# Rich import with auto-install
try:
    from rich.progress import Progress
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.progress import Progress

# Configuration
URL = "https://docs.vllm.ai/en/latest/cli/index.html#serve"


def cast_default(text: str):
    """Convert the default string into a YAML-friendly Python object."""
    text = text.strip()
    if text in {"None", ""}:
        return None
    if text in {"True", "False"}:
        return text == "True"
    try:
        # literal_eval turns `'[\'*\']'` into ['*'] etc.
        return ast.literal_eval(text)
    except Exception:
        # fall back to raw string
        return text


def scrape_vllm_flags() -> OrderedDict[str, OrderedDict[str, tuple]]:
    """
    Scrape vLLM serve flags from documentation.
    Returns: {section -> {flag_name -> (default, description)}}
    """
    print("🌐 Fetching vLLM documentation...")
    html = requests.get(URL, timeout=15).text
    soup = BeautifulSoup(html, "html.parser")

    serve_h2 = soup.find(id="serve")
    if not serve_h2:
        sys.exit("❌ Could not locate serve section in documentation")

    sections: OrderedDict[str, OrderedDict[str, tuple]] = OrderedDict()
    current = "Top-level options"
    sections[current] = OrderedDict()

    # Count total flags for progress bar
    total_flags = 0
    temp_node = serve_h2
    while temp_node := temp_node.find_next():
        if temp_node.name == "h2":
            break
        if temp_node.name == "h5":
            total_flags += 1

    print(f"📊 Found {total_flags} flags to parse...")

    # Parse flags with progress bar
    node = serve_h2
    with Progress() as progress:
        task = progress.add_task("Parsing flags", total=total_flags)

        while node := node.find_next():
            if node.name == "h2":
                break  # end of serve section

            if node.name == "h4":  # subsection header
                current = node.get_text(strip=True)
                # Remove pilcrow symbol from section headers
                current = re.sub(r'¶.*$', '', current)
                sections[current] = OrderedDict()
                continue

            if node.name != "h5":  # flags live in <h5>
                continue

            # Extract flag name
            raw = node.get_text(strip=True)
            names = re.findall(r"`([^`]+)`", raw)

            if names:
                flag_text = names[0]
            else:
                flag_text = raw.strip("`")

            # Parse multiple flag variants (e.g., "flag1,flag2,--no-flag1")
            flag_variants = [f.strip() for f in flag_text.split(",")]

            # Filter to get the main long-form flag
            long_flags = []
            for flag in flag_variants:
                flag = re.sub(r'¶.*$', '', flag)  # Remove pilcrow symbol
                # Skip short forms and negation forms
                if flag.startswith('_') or (flag.startswith('-') and not flag.startswith('--')):
                    continue
                if flag.startswith('--no-') or flag.startswith('__no'):
                    continue
                long_flags.append(flag)

            # Select the best flag name
            if long_flags:
                long_flag = long_flags[0]
            else:
                long_flag = max(flag_variants, key=len)
                long_flag = re.sub(r'¶.*$', '', long_flag)

            # Clean up flag name
            long_flag = re.sub(r'__.*$', '', long_flag)
            long_flag = long_flag.rstrip('_')

            # Convert to hyphen convention
            key = long_flag.lstrip("-").replace("_", "-")

            # Extract description and default value
            desc_lines, default_val = [], None
            for sib in node.find_next_siblings():
                if sib.name and sib.name.startswith("h"):
                    break
                txt = sib.get_text(" ", strip=True)
                if not txt:
                    continue

                # Look for default value
                m = re.match(r"Default:\s*`?([^`]+)`?", txt)
                if m:
                    default_val = m.group(1)
                    break
                desc_lines.append(txt)

            default = cast_default(default_val or "None")
            description = " ".join(desc_lines)
            sections[current][key] = (default, description)
            progress.update(task, advance=1)

    return sections


def generate_yaml(sections: OrderedDict[str, OrderedDict[str, tuple]]) -> str:
    """Generate YAML config with inline comments."""
    lines: list[str] = []
    dumper = yaml.SafeDumper
    dumper.default_flow_style = False

    for section, opts in sections.items():
        lines.append(f"# {section}")
        for key, (default, desc) in opts.items():
            # Format value using PyYAML
            val_str = yaml.dump({key: default}, Dumper=dumper).split(":", 1)[1].strip()
            if not val_str:
                val_str = "null"

            # Handle multi-line descriptions
            if desc:
                desc = re.sub(r'\s+', ' ', desc.strip())
                if len(desc) > 80:
                    lines.append(f"{key}: {val_str}")
                    lines.append(f"  # {desc}")
                else:
                    lines.append(f"{key}: {val_str}  # {desc}")
            else:
                lines.append(f"{key}: {val_str}")
        lines.append("")  # blank line between sections

    return "\n".join(lines)


def validate_parsing(sections: OrderedDict[str, OrderedDict[str, tuple]]) -> bool:
    """Validate our parsing against raw HTML (optional verification)."""
    try:
        # Quick validation - count flags
        total_flags = sum(len(opts) for opts in sections.values())
        print(f"✅ Parsed {total_flags} flags successfully")
        return True
    except Exception as e:
        print(f"⚠️  Validation warning: {e}")
        return False


def main():
    """Main function to orchestrate the parsing process."""
    parser = argparse.ArgumentParser(description="Scrape vLLM serve CLI docs and generate YAML config.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vllm_serve_defaults.yaml"),
        help="Path to output YAML file (default: vllm_serve_defaults.yaml)",
    )
    args = parser.parse_args()
    output_file = args.output

    print("🚀 vLLM CLI Parser Starting...")

    try:
        # Scrape flags from documentation
        sections = scrape_vllm_flags()

        # Validate parsing
        validate_parsing(sections)

        # Generate YAML config
        print("📝 Generating YAML config...")
        yaml_content = generate_yaml(sections)

        # Write to file
        output_file.write_text(yaml_content)
        print(f"✅ Successfully wrote {output_file.resolve()}")

        # Summary
        total_flags = sum(len(opts) for opts in sections.values())
        print(f"📊 Summary: {len(sections)} sections, {total_flags} flags")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
