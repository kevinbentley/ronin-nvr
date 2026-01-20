#!/usr/bin/env python
"""Validate Alembic migration chain integrity.

This script checks for:
1. Broken revision chains (down_revision references non-existent revision)
2. Multiple heads (branch conflicts)
3. Circular dependencies
4. Revision ID vs filename mismatches (warning)

Usage:
    python scripts/validate_migrations.py [--fix-suggestions]

Exit codes:
    0: All validations passed
    1: Validation errors found
    2: Script execution error
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple


class Migration(NamedTuple):
    """Represents a parsed migration file."""

    filepath: Path
    filename: str
    revision: str
    down_revision: str | None
    docstring_revises: str | None


def parse_migration(filepath: Path) -> Migration | None:
    """Parse a migration file and extract revision information."""
    content = filepath.read_text()

    # Extract revision ID
    rev_match = re.search(
        r"^revision(?:\s*:\s*str)?\s*=\s*['\"]([^'\"]+)['\"]", content, re.MULTILINE
    )
    if not rev_match:
        return None
    revision = rev_match.group(1)

    # Extract down_revision
    down_match = re.search(
        r"^down_revision(?:\s*:\s*Union\[str,\s*None\])?\s*="
        r"\s*(?:None|['\"]([^'\"]*)['\"])",
        content,
        re.MULTILINE,
    )
    down_revision = (
        down_match.group(1) if down_match and down_match.group(1) else None
    )

    # Extract Revises from docstring
    revises_match = re.search(r"^Revises:\s*(.+)$", content, re.MULTILINE)
    docstring_revises = revises_match.group(1).strip() if revises_match else None
    if docstring_revises == "":
        docstring_revises = None

    return Migration(
        filepath=filepath,
        filename=filepath.stem,
        revision=revision,
        down_revision=down_revision,
        docstring_revises=docstring_revises,
    )


def validate_migrations(
    versions_dir: Path, show_fix_suggestions: bool = False
) -> int:
    """Validate all migrations in the versions directory.

    Returns:
        Exit code (0 = success, 1 = errors found)
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Parse all migrations
    migrations: dict[str, Migration] = {}
    migration_files = list(versions_dir.glob("*.py"))

    for filepath in migration_files:
        if filepath.name.startswith("__"):
            continue
        migration = parse_migration(filepath)
        if migration:
            if migration.revision in migrations:
                errors.append(
                    f"DUPLICATE REVISION: '{migration.revision}' found in:\n"
                    f"  - {migrations[migration.revision].filepath}\n"
                    f"  - {migration.filepath}"
                )
            migrations[migration.revision] = migration

    if not migrations:
        print("No migrations found.")
        return 0

    # Find all revisions and build dependency graph
    all_revisions = set(migrations.keys())
    referenced_revisions: set[str] = set()

    for migration in migrations.values():
        if migration.down_revision:
            referenced_revisions.add(migration.down_revision)

    # Check 1: Broken chain (down_revision references non-existent revision)
    for migration in migrations.values():
        if migration.down_revision and migration.down_revision not in all_revisions:
            error_msg = (
                f"BROKEN CHAIN: {migration.filename}\n"
                f"  down_revision='{migration.down_revision}' does not exist"
            )

            # Try to find what they might have meant
            if show_fix_suggestions:
                possible_matches = [
                    rev
                    for rev, m in migrations.items()
                    if migration.down_revision in m.filename
                    or m.filename in migration.down_revision
                ]
                if possible_matches:
                    error_msg += f"\n  Possible match: '{possible_matches[0]}'"

            errors.append(error_msg)

    # Check 2: Multiple heads
    heads = all_revisions - referenced_revisions
    if len(heads) > 1:
        head_details = []
        for head in heads:
            m = migrations[head]
            head_details.append(f"    - {head} ({m.filename})")
        errors.append(
            f"MULTIPLE HEADS: Found {len(heads)} heads (should be 1):\n"
            + "\n".join(head_details)
        )

    # Check 3: Orphaned migrations (no path to base)
    def can_reach_base(rev: str, visited: set[str]) -> bool:
        if rev in visited:
            return False  # Circular
        visited.add(rev)
        if rev not in migrations:
            return False
        migration = migrations[rev]
        if migration.down_revision is None:
            return True  # Base migration
        return can_reach_base(migration.down_revision, visited)

    for revision, migration in migrations.items():
        if not can_reach_base(revision, set()):
            errors.append(
                f"UNREACHABLE: {migration.filename} cannot reach base migration"
            )

    # Check 4: Docstring mismatch (warning only)
    for migration in migrations.values():
        if (
            migration.docstring_revises
            and migration.docstring_revises != migration.down_revision
        ):
            if migration.down_revision is None and migration.docstring_revises == "":
                continue
            warnings.append(
                f"DOCSTRING MISMATCH: {migration.filename}\n"
                f"  Docstring says: Revises: {migration.docstring_revises}\n"
                f"  Actual down_revision: {migration.down_revision}"
            )

    # Check 5: Filename vs revision ID mismatch (informational)
    for migration in migrations.values():
        # Extract date prefix from filename if present
        filename_base = migration.filename.split("_", 1)[0]

        # Check if filename looks like it should match revision
        if filename_base.isdigit() and len(filename_base) == 8:  # YYYYMMDD format
            if not migration.revision.startswith(filename_base):
                warnings.append(
                    f"NAMING MISMATCH: {migration.filename}\n"
                    f"  Filename suggests date: {filename_base}\n"
                    f"  Actual revision: {migration.revision}"
                )

    # Print results
    print(f"\nValidated {len(migrations)} migrations\n")

    if warnings:
        print("=" * 60)
        print("WARNINGS:")
        print("=" * 60)
        for warning in warnings:
            print(f"\n{warning}")

    if errors:
        print("\n" + "=" * 60)
        print("ERRORS:")
        print("=" * 60)
        for error in errors:
            print(f"\n{error}")
        print(f"\n{len(errors)} error(s) found.")
        return 1

    print("All validations passed!")
    if heads:
        print(f"Current head: {list(heads)[0]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Alembic migration chain integrity"
    )
    parser.add_argument(
        "--fix-suggestions",
        action="store_true",
        help="Show suggestions for fixing broken chains",
    )
    parser.add_argument(
        "--versions-dir",
        type=Path,
        default=None,
        help="Path to alembic versions directory",
    )
    args = parser.parse_args()

    # Find versions directory
    if args.versions_dir:
        versions_dir = args.versions_dir
    else:
        # Try common locations
        script_dir = Path(__file__).parent
        candidates = [
            script_dir.parent / "alembic" / "versions",
            Path.cwd() / "alembic" / "versions",
            Path.cwd() / "backend" / "alembic" / "versions",
        ]
        versions_dir = None
        for candidate in candidates:
            if candidate.exists():
                versions_dir = candidate
                break

        if not versions_dir:
            print(
                "Error: Could not find alembic/versions directory", file=sys.stderr
            )
            return 2

    if not versions_dir.exists():
        print(f"Error: Directory not found: {versions_dir}", file=sys.stderr)
        return 2

    return validate_migrations(versions_dir, args.fix_suggestions)


if __name__ == "__main__":
    sys.exit(main())
