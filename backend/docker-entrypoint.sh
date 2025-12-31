#!/bin/bash
set -e

echo "Running database migrations..."

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

alembic upgrade head

echo "Starting application..."
exec "$@"
