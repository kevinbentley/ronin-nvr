# Database Migrations Guide

This project uses Alembic for database schema migrations.

## Revision ID Convention

This project uses a **date-based revision ID convention**:

```
YYYYMMDD_description
```

For example: `20260120_add_object_event_clips`

The revision ID is automatically generated to match this pattern when you create a new migration.

## Creating New Migrations

```bash
# Autogenerate from model changes
cd backend
alembic revision --autogenerate -m "add_user_preferences"

# Create empty migration for manual SQL
alembic revision -m "add_custom_index"
```

The revision ID will be automatically generated in the format `YYYYMMDD_description`.

## Common Mistakes to Avoid

### 1. Never assume filename = revision ID

The filename and revision ID are separate values. Always check the `revision` variable inside the migration file.

**Wrong:**
```python
# In my_migration.py, assuming previous file is named 20260119_add_feature.py
down_revision = '20260119_add_feature'  # WRONG - this is the filename
```

**Right:**
```python
# Open 20260119_add_feature.py and find: revision = '8b2e9f4c3a1d'
down_revision = '8b2e9f4c3a1d'  # CORRECT - actual revision ID
```

### 2. Always validate before committing

```bash
python scripts/validate_migrations.py
```

### 3. Check migration state before and after changes

```bash
alembic current
alembic history --verbose
```

## Validation Script

Run the migration validator to check chain integrity:

```bash
# Basic validation
python scripts/validate_migrations.py

# Show fix suggestions for errors
python scripts/validate_migrations.py --fix-suggestions
```

The validator checks for:
- Broken revision chains (down_revision references non-existent revision)
- Multiple heads (branch conflicts)
- Circular dependencies
- Filename vs revision ID mismatches (warning)

## Running Migrations

### Development

```bash
cd backend
alembic upgrade head
```

### Docker

```bash
# Migrations run automatically on container start
docker compose up -d backend

# Manual migration
docker compose exec backend alembic upgrade head
```

### Rolling Back

```bash
# Downgrade one revision
alembic downgrade -1

# Downgrade to specific revision
alembic downgrade 20260115_add_object_events

# Downgrade to base (removes all migrations)
alembic downgrade base
```

## Troubleshooting

### "Can't locate revision" / KeyError

This means a broken chain - a migration references a revision that doesn't exist.

```bash
# Find the problem
python scripts/validate_migrations.py --fix-suggestions

# The output will show which file has the wrong down_revision
# and suggest what it should be
```

### Multiple heads detected

This happens when two migrations have the same `down_revision`:

```bash
# Show all heads
alembic heads

# Merge the branches
alembic merge -m "merge_branches" head1 head2
```

### Migration partially applied

If a migration failed mid-way:

1. Check what was applied: `alembic current`
2. Manually fix the database state (add/remove columns as needed)
3. Stamp to the correct revision: `alembic stamp <revision>`

### Database is ahead of migrations

If the database has changes that migrations don't know about:

```bash
# See what Alembic thinks the current state is
alembic current

# Stamp to indicate the database is at a specific revision
alembic stamp <revision>
```
