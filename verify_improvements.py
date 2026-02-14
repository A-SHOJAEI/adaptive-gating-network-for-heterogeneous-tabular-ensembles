#!/usr/bin/env python3
"""Verification script to check all mandatory improvements."""

import sys
from pathlib import Path
import subprocess


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a required file exists."""
    if Path(filepath).exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} NOT FOUND")
        return False


def check_file_not_exists(filepath: str, description: str) -> bool:
    """Check that an unwanted file does not exist."""
    if not Path(filepath).exists():
        print(f"✓ {description}: {filepath} removed")
        return True
    else:
        print(f"✗ {description}: {filepath} still exists")
        return False


def check_file_content(filepath: str, search_term: str, description: str) -> bool:
    """Check if file contains specific content."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            if search_term in content:
                print(f"✓ {description}")
                return True
            else:
                print(f"✗ {description}: '{search_term}' not found")
                return False
    except FileNotFoundError:
        print(f"✗ {description}: {filepath} not found")
        return False


def check_readme_length() -> bool:
    """Check that README is under 200 lines."""
    try:
        with open("README.md", 'r') as f:
            lines = len(f.readlines())
            if lines < 200:
                print(f"✓ README length: {lines} lines (< 200)")
                return True
            else:
                print(f"✗ README length: {lines} lines (>= 200)")
                return False
    except FileNotFoundError:
        print("✗ README.md not found")
        return False


def check_no_scientific_notation() -> bool:
    """Check YAML files don't use scientific notation."""
    yaml_files = list(Path("configs").glob("*.yaml"))
    all_good = True
    for yaml_file in yaml_files:
        with open(yaml_file, 'r') as f:
            content = f.read()
            if 'e-' in content.lower() or 'e+' in content.lower():
                # Check if it's actually scientific notation (not in comments/strings)
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if not line.strip().startswith('#'):
                        if 'e-' in line.lower() or 'e+' in line.lower():
                            print(f"✗ Scientific notation found in {yaml_file}:{i}")
                            all_good = False
    if all_good:
        print(f"✓ No scientific notation in YAML files")
    return all_good


def main():
    """Run all verification checks."""
    print("=" * 70)
    print("VERIFICATION OF MANDATORY IMPROVEMENTS")
    print("=" * 70)
    print()

    checks = []

    print("1. Checking for removed auto-generated documents...")
    checks.append(check_file_not_exists("COMPLETION_REPORT.md", "Removed COMPLETION_REPORT"))
    checks.append(check_file_not_exists("FIXES_SUMMARY.md", "Removed FIXES_SUMMARY"))
    checks.append(check_file_not_exists("IMPROVEMENTS_SUMMARY.md", "Removed IMPROVEMENTS_SUMMARY"))
    checks.append(check_file_not_exists("MANDATORY_FIXES_CHECKLIST.md", "Removed MANDATORY_FIXES_CHECKLIST"))
    checks.append(check_file_not_exists("PROJECT_SUMMARY.md", "Removed PROJECT_SUMMARY"))
    checks.append(check_file_not_exists("REQUIREMENTS_CHECKLIST.md", "Removed REQUIREMENTS_CHECKLIST"))
    print()

    print("2. Checking PyTorch GatingNetwork implementation...")
    checks.append(check_file_content(
        "src/adaptive_gating_network_for_heterogeneous_tabular_ensembles/models/components.py",
        "import torch",
        "PyTorch import present"
    ))
    checks.append(check_file_content(
        "src/adaptive_gating_network_for_heterogeneous_tabular_ensembles/models/components.py",
        "class GatingMLP(nn.Module)",
        "PyTorch GatingMLP class implemented"
    ))
    checks.append(check_file_content(
        "src/adaptive_gating_network_for_heterogeneous_tabular_ensembles/models/components.py",
        "loss_per_sample * batch_w",
        "Sample weight support in loss"
    ))
    print()

    print("3. Checking LICENSE...")
    checks.append(check_file_exists("LICENSE", "LICENSE file"))
    checks.append(check_file_content("LICENSE", "MIT License", "MIT License"))
    checks.append(check_file_content("LICENSE", "Copyright (c) 2026 Alireza Shojaei", "Copyright holder"))
    print()

    print("4. Checking README...")
    checks.append(check_file_exists("README.md", "README file"))
    checks.append(check_readme_length())
    checks.append(check_file_content("README.md", "PyTorch", "PyTorch mentioned"))
    print()

    print("5. Checking YAML configurations...")
    checks.append(check_file_exists("configs/default.yaml", "Default config"))
    checks.append(check_file_exists("configs/ablation.yaml", "Ablation config"))
    checks.append(check_no_scientific_notation())
    print()

    print("6. Checking requirements...")
    checks.append(check_file_content("requirements.txt", "torch", "PyTorch in requirements"))
    checks.append(check_file_content("requirements.txt", "numpy", "NumPy in requirements"))
    print()

    print("7. Checking type hints and docstrings...")
    checks.append(check_file_content(
        "src/adaptive_gating_network_for_heterogeneous_tabular_ensembles/models/components.py",
        "-> None:",
        "Type hints present"
    ))
    checks.append(check_file_content(
        "src/adaptive_gating_network_for_heterogeneous_tabular_ensembles/models/components.py",
        'Args:',
        "Google-style docstrings"
    ))
    print()

    print("8. Checking MLflow error handling...")
    checks.append(check_file_content(
        "src/adaptive_gating_network_for_heterogeneous_tabular_ensembles/training/trainer.py",
        "except Exception",
        "MLflow exception handling"
    ))
    print()

    print("9. Checking test structure...")
    checks.append(check_file_exists("tests/conftest.py", "Test fixtures"))
    checks.append(check_file_exists("tests/test_model.py", "Model tests"))
    print()

    print("10. Checking scripts...")
    checks.append(check_file_exists("scripts/train.py", "Training script"))
    print()

    # Summary
    print()
    print("=" * 70)
    passed = sum(checks)
    total = len(checks)
    percentage = (passed / total) * 100 if total > 0 else 0

    print(f"RESULTS: {passed}/{total} checks passed ({percentage:.1f}%)")
    print("=" * 70)

    if passed == total:
        print("✓ ALL MANDATORY IMPROVEMENTS VERIFIED")
        return 0
    else:
        print(f"✗ {total - passed} checks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
