#!/bin/bash
set -e

echo "========================================"
echo "Running database migrations..."
echo "========================================"

# Function to handle migration failures
handle_migration_failure() {
    local exit_code=$1
    echo ""
    echo "========================================"
    echo "MIGRATION FAILED (exit code: $exit_code)"
    echo "========================================"
    echo ""
    echo "Possible causes:"
    echo "  1. Broken migration chain (down_revision references non-existent revision)"
    echo "  2. Database connection issue"
    echo "  3. SQL syntax error in migration"
    echo "  4. Table/column already exists (migration partially applied)"
    echo ""
    echo "Debugging steps:"
    echo "  1. Check current migration state: alembic current"
    echo "  2. View migration history: alembic history"
    echo "  3. Validate migration chain: python scripts/validate_migrations.py"
    echo "  4. Check logs above for specific error"
    echo ""
    echo "To manually apply migrations:"
    echo "  docker compose exec backend alembic upgrade head"
    echo ""
    exit "$exit_code"
}

# Check if alembic_version table exists - if not, but tables exist,
# we need to stamp the database first (handles pre-Alembic databases)
TABLES_EXIST=$(python -c "
import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
import os

async def check():
    engine = create_async_engine(os.environ['DATABASE_URL'])
    async with engine.connect() as conn:
        # Check if cameras table exists (indicates existing db)
        result = await conn.execute(text(
            \"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'cameras')\"
        ))
        cameras_exist = result.scalar()

        # Check if alembic_version table exists
        result = await conn.execute(text(
            \"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'alembic_version')\"
        ))
        alembic_exists = result.scalar()

        if cameras_exist and not alembic_exists:
            print('stamp_needed')
        else:
            print('ok')
    await engine.dispose()

asyncio.run(check())
" 2>/dev/null || echo "ok")

if [ "$TABLES_EXIST" = "stamp_needed" ]; then
    echo "Existing database detected without Alembic tracking. Stamping with bfc3f1b7eb27..."
    alembic stamp bfc3f1b7eb27
fi

# Validate migration chain before attempting upgrade
echo "Validating migration chain..."
if ! python scripts/validate_migrations.py 2>/dev/null; then
    echo ""
    echo "WARNING: Migration chain validation failed!"
    echo "Attempting migration anyway, but this may fail."
    echo ""
fi

# Show current state
echo ""
echo "Current migration state:"
alembic current 2>/dev/null || echo "(no migrations applied yet)"
echo ""

# Run migrations with error handling
echo "Applying migrations..."
if ! alembic upgrade head; then
    handle_migration_failure $?
fi

echo ""
echo "Migrations completed successfully!"
echo "Current head:"
alembic current
echo ""

echo "Starting application..."
exec "$@"
