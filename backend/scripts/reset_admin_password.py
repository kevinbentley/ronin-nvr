#!/usr/bin/env python3
"""Reset the admin user's password directly in the database."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.config import get_settings
from app.models.user import User
from app.services.auth import hash_password
from app.services.encryption import generate_random_password


async def reset_password(
    username: str, new_password: str | None, create_if_missing: bool = False
) -> None:
    """Reset a user's password."""
    settings = get_settings()

    engine = create_async_engine(settings.database_url, echo=False)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # Find the user
        result = await session.execute(
            select(User).where(User.username == username)
        )
        user = result.scalar_one_or_none()

        # Generate password if not provided
        if new_password is None:
            new_password = generate_random_password()
            print(f"Generated password: {new_password}")

        if not user:
            if create_if_missing:
                # Create the user
                hashed = hash_password(new_password)
                is_admin = username == "admin"
                new_user = User(
                    username=username,
                    hashed_password=hashed,
                    is_admin=is_admin,
                    is_active=True,
                )
                session.add(new_user)
                await session.commit()
                print(f"\nCreated new user '{username}'.")
                if is_admin:
                    print("(User has admin privileges)")
            else:
                print(f"Error: User '{username}' not found.")
                print("Use --create to create the user if it doesn't exist.")
                print("\nExisting users:")
                all_users = await session.execute(select(User.username))
                for (name,) in all_users:
                    print(f"  - {name}")
                await engine.dispose()
                sys.exit(1)
        else:
            # Update existing user
            hashed = hash_password(new_password)
            await session.execute(
                update(User).where(User.username == username).values(hashed_password=hashed)
            )
            await session.commit()

            print(f"\nPassword updated successfully for user '{username}'.")
            if user.is_admin:
                print("(This user has admin privileges)")

    await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset a user's password in the RoninNVR database."
    )
    parser.add_argument(
        "-u", "--username",
        default="admin",
        help="Username to reset password for (default: admin)"
    )
    parser.add_argument(
        "-p", "--password",
        help="New password (if not provided, generates a random one)"
    )
    parser.add_argument(
        "-c", "--create",
        action="store_true",
        help="Create the user if it doesn't exist"
    )

    args = parser.parse_args()

    asyncio.run(reset_password(args.username, args.password, args.create))


if __name__ == "__main__":
    main()
