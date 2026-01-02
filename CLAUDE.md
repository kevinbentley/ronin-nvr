# Overview
This document contains critical information about working with this codebase. Follow these guidelines precisely.

This project will use React for any GUI. All controls, buttons, etc. will use an event driven architecture, so when a button is pressed, the UI remains responsive while the process executes.

Standalone command line tools such as those that process or generate data files should include argument processing, and the usage should be documented in a readme file.

When writing a project plan, always create a TODO.md with the tasks and phases. Mark things as complete in the readme as the process runs.

Each time a major feature or phase is developed, make sure to thoroughly test it, then commit it to a local git repo (do not push).

## Safety
When deleting files, make sure to stash or commit them to the repository first, so they can be recovered if something goes awry.

## Core Development Rules for Python

1. Package Management
   - Use pip, maintain an up to date requirements.txt file
   - Create a venv and a shell script to automatically make/start in it.

2. Code Quality
   - Type hints required for all code
   - Public APIs must have docstrings
   - Functions must be focused and small
   - Follow existing patterns exactly
   - Line length: 88 chars maximum

3. Testing Requirements
   - Framework: `uv run pytest`
   - Async testing: use anyio, not asyncio
   - Coverage: test edge cases and errors
   - New features require tests
   - Bug fixes require regression tests

4. Code Style
    - PEP 8 naming (snake_case for functions/variables)
    - Class names in PascalCase
    - Constants in UPPER_SNAKE_CASE
    - Document with docstrings
    - Use f-strings for formatting

- For commits fixing bugs or adding features based on user reports add:
  ```bash
  git commit --trailer "Reported-by:<name>"
  ```
  Where `<name>` is the name of the user.

- For commits related to a Github issue, add
  ```bash
  git commit --trailer "Github-Issue:#<number>"
  ```
- NEVER ever mention a `co-authored-by` or similar aspects. In particular, never
  mention the tool used to create the commit message or PR.

## Development Philosophy

- **Simplicity**: Write simple, straightforward code
- **Readability**: Make code easy to understand
- **Performance**: Consider performance without sacrificing readability
- **Maintainability**: Write code that's easy to update
- **Testability**: Ensure code is testable
- **Reusability**: Create reusable components and functions
- **Less Code = Less Debt**: Minimize code footprint

## Coding Best Practices

- **Early Returns**: Use to avoid nested conditions
- **Descriptive Names**: Use clear variable/function names (prefix handlers with "handle")
- **Constants Over Functions**: Use constants where possible
- **DRY Code**: Don't repeat yourself
- **Functional Style**: Prefer functional, immutable approaches when not verbose
- **Minimal Changes**: Only modify code related to the task at hand
- **Function Ordering**: Define composing functions before their components
- **TODO Comments**: Mark issues in existing code with "TODO:" prefix
- **Simplicity**: Prioritize simplicity and readability over clever solutions
- **Build Iteratively** Start with minimal functionality and verify it works before adding complexity
- **Run Tests**: Test your code frequently with realistic inputs and validate outputs
- **Build Test Environments**: Create testing environments for components that are difficult to validate directly
- **Functional Code**: Use functional and stateless approaches where they improve clarity
- **Clean logic**: Keep core logic clean and push implementation details to the edges
- **File Organsiation**: Balance file organization with simplicity - use an appropriate number of files for the project scale

## Core Development Rules for React Native

1. Project Structure
   - Use feature-based folder structure (group by feature, not by type)
   - Keep components small and focused on a single responsibility
   - Separate business logic from UI components using hooks

2. Code Quality
   - TypeScript required for all code
   - Use functional components with hooks (no class components)
   - Props must have explicit type definitions
   - Line length: 100 chars maximum

3. Naming Conventions
   - Components: PascalCase (e.g., `UserProfile.tsx`)
   - Hooks: camelCase with `use` prefix (e.g., `useAuth.ts`)
   - Utilities: camelCase (e.g., `formatDate.ts`)
   - Constants: UPPER_SNAKE_CASE
   - Event handlers: prefix with `handle` (e.g., `handlePress`)

4. Component Best Practices
   - Destructure props at the top of components
   - Use `React.memo()` for expensive pure components
   - Avoid inline styles; use StyleSheet.create()
   - Keep styles at the bottom of the file or in separate `.styles.ts` files
   - Use `useCallback` for event handlers passed to child components
   - Use `useMemo` for expensive computations

5. State Management
   - Prefer local state with `useState` when possible
   - Use Context for app-wide state (auth, theme, etc.)
   - Consider Zustand or Redux Toolkit for complex state
   - Avoid prop drilling beyond 2-3 levels

6. Testing
   - Framework: Jest + React Native Testing Library
   - Test component behavior, not implementation
   - Mock native modules appropriately
   - Snapshot tests for UI regression only

## Core Development Rules for Rust

1. Project Structure
   - Use Cargo workspaces for multi-crate projects
   - Keep `lib.rs` as the public API surface
   - Use `mod.rs` or inline modules appropriately
   - Separate binary and library code

2. Code Quality
   - Run `cargo clippy` before commits (treat warnings as errors)
   - Run `cargo fmt` for consistent formatting
   - All public items must have doc comments (`///`)
   - Use `#[must_use]` for functions with important return values
   - Line length: 100 chars maximum

3. Naming Conventions
   - Types, traits, enums: PascalCase
   - Functions, variables, modules: snake_case
   - Constants and statics: UPPER_SNAKE_CASE
   - Lifetimes: short lowercase (`'a`, `'b`)
   - Type parameters: single uppercase letter or descriptive PascalCase

4. Error Handling
   - Use `Result<T, E>` for recoverable errors
   - Use `?` operator for error propagation
   - Define custom error types with `thiserror` or manual impl
   - Reserve `panic!` for unrecoverable states only
   - Use `anyhow` for application code, `thiserror` for libraries

5. Memory and Safety
   - Prefer borrowing over cloning
   - Use `Cow<'a, T>` for flexible ownership
   - Avoid `unsafe` unless absolutely necessary; document why
   - Use `Arc` and `Mutex` for shared state across threads
   - Prefer `&str` over `String` in function parameters

6. Testing
   - Unit tests in the same file (`#[cfg(test)]` module)
   - Integration tests in `tests/` directory
   - Use `proptest` or `quickcheck` for property-based testing
   - Document test with `#[doc = "..."]` for complex scenarios

7. Performance
   - Use iterators over explicit loops
   - Prefer stack allocation over heap when reasonable
   - Use `#[inline]` judiciously for hot paths
   - Profile before optimizing

## Core Development Rules for C++

1. Language Standard
   - Use C++17 minimum; C++20 features where available
   - Enable `-Wall -Wextra -Wpedantic` warnings
   - Treat warnings as errors in CI (`-Werror`)

2. Project Structure
   - Use CMake for build system
   - Headers in `include/`, sources in `src/`
   - One class per header/source file pair
   - Use `#pragma once` for header guards

3. Naming Conventions
   - Classes, structs, enums: PascalCase
   - Functions, variables: camelCase or snake_case (be consistent)
   - Member variables: `m_` prefix or trailing underscore
   - Constants and macros: UPPER_SNAKE_CASE
   - Namespaces: lowercase

4. Code Quality
   - Prefer `const` correctness throughout
   - Use `nullptr` instead of `NULL` or `0`
   - Use `auto` for complex types, explicit types for primitives
   - Prefer range-based for loops
   - Use `override` and `final` keywords appropriately
   - Line length: 100 chars maximum

5. Memory Management
   - Use smart pointers (`std::unique_ptr`, `std::shared_ptr`)
   - Avoid raw `new`/`delete`; use RAII
   - Prefer stack allocation over heap
   - Use `std::move` for transferring ownership
   - Follow Rule of Zero/Five for resource management

6. Modern C++ Practices
   - Use `std::optional` for nullable values
   - Use `std::variant` over unions
   - Use `std::string_view` for non-owning string references
   - Prefer `constexpr` for compile-time computation
   - Use structured bindings for tuple/pair decomposition

7. Error Handling
   - Use exceptions for exceptional cases only
   - Use `std::expected` (C++23) or result types for expected failures
   - Document exception guarantees (noexcept where appropriate)
   - Never throw in destructors

8. Testing
   - Framework: Google Test or Catch2
   - Use test fixtures for shared setup
   - Name tests descriptively: `TestClass_Method_Condition_ExpectedResult`

## Core Development Rules for Bash Shell Scripts

1. Script Setup
   - Always start with shebang: `#!/usr/bin/env bash`
   - Enable strict mode: `set -euo pipefail`
   - Add `IFS=$'\n\t'` for safer word splitting
   - Include usage function and help text

2. Code Quality
   - Run `shellcheck` before commits
   - Quote all variable expansions: `"${var}"`
   - Use `[[ ]]` for conditionals (not `[ ]`)
   - Use `$(command)` for substitution (not backticks)
   - Line length: 80 chars maximum

3. Naming Conventions
   - Script names: lowercase with hyphens (e.g., `build-release.sh`)
   - Functions: snake_case
   - Local variables: lowercase snake_case
   - Environment/exported variables: UPPER_SNAKE_CASE
   - Constants: readonly UPPER_SNAKE_CASE

4. Best Practices
   - Declare variables with `local` in functions
   - Use `readonly` for constants
   - Use `declare -r` and `declare -i` for type hints
   - Prefer `printf` over `echo` for portability
   - Use arrays for lists of items
   - Always use `"$@"` to pass arguments (not `$*`)

5. Error Handling
   - Check exit codes explicitly when needed
   - Use `trap` for cleanup on exit/error
   - Provide meaningful error messages to stderr
   - Exit with appropriate codes (0 = success, 1+ = error)

6. Script Structure
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail

   readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
   readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

   usage() {
       cat <<EOF
   Usage: ${SCRIPT_NAME} [options] <arguments>

   Description of what this script does.

   Options:
       -h, --help      Show this help message
       -v, --verbose   Enable verbose output
   EOF
   }

   main() {
       # Script logic here
   }

   main "$@"
   ```

7. Portability
   - Target Linux and MacOS platforms
   - Use POSIX-compatible constructs when possible
   - Document bash-specific features used
   - Document and maintain a SYSTEM.md file that tracks dependencies including things like CMake, gcc versions, packages like ffmpg, etc.

## Docker Compose Development

This project uses Docker Compose to orchestrate all services. There are two compose files:
- `docker-compose.yml` - GPU-enabled configuration (requires nvidia-container-toolkit)
- `docker-compose.cpu.yml` - CPU-only configuration (no GPU support)

### Services

| Service | Container Name | Purpose | Port |
|---------|---------------|---------|------|
| postgres | ronin-postgres | PostgreSQL 16 database | 5432 |
| backend | ronin-backend | FastAPI application server | 8000 |
| frontend | ronin-frontend | React app served via Nginx | 80 |
| live-detection | ronin-live-detection | Real-time ML detection worker | - |
| ml-worker | (scaled) | Historical recording analysis | - |
| transcode-worker | (scaled) | H.265 video transcoding | - |

### Common Commands

```bash
# Start all default services (live detection enabled)
docker compose up -d

# Start with historical ML processing enabled
docker compose --profile historical up -d

# CPU-only mode (no NVIDIA GPU)
docker compose -f docker-compose.cpu.yml up -d

# View logs for specific service
docker compose logs -f backend
docker compose logs -f live-detection

# Rebuild after code changes
docker compose build backend
docker compose up -d backend

# Rebuild all services
docker compose build
docker compose up -d

# Stop all services
docker compose down

# Stop and remove volumes (WARNING: deletes database)
docker compose down -v

# Check service status
docker compose ps

# Restart a specific service
docker compose restart backend
```

### Testing During Development

```bash
# Run backend tests (from host, uses local venv)
cd backend && uv run pytest

# Run specific test file
cd backend && uv run pytest tests/unit/test_camera_stream.py

# Run tests inside container
docker compose exec backend pytest

# Check backend health
curl http://localhost:8000/api/health

# View database directly
docker compose exec postgres psql -U ronin_nvr_user -d ronin_nvr
```

### Building and Debugging

```bash
# Build specific service with no cache
docker compose build --no-cache backend

# View build output
docker compose build --progress=plain backend

# Shell into running container
docker compose exec backend bash
docker compose exec postgres bash

# View container resource usage
docker stats

# Check GPU availability in worker
docker compose exec live-detection nvidia-smi

# Tail all logs
docker compose logs -f

# View last 100 lines of backend logs
docker compose logs --tail=100 backend
```

### Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Required for production
JWT_SECRET_KEY=<generate-with-python-secrets>
ENCRYPTION_KEY=<generate-fernet-key>

# Database (defaults work for development)
POSTGRES_USER=ronin_nvr_user
POSTGRES_PASSWORD=ronin_pass
POSTGRES_DB=ronin_nvr

# ML tuning
ML_CONFIDENCE_THRESHOLD=0.5
LIVE_DETECTION_FPS=1.0
LIVE_DETECTION_COOLDOWN=30.0
```

### Volume Mounts (Production)

The GPU compose file uses host paths for persistent data:
- `/opt2/ronin/postgres` - Database files
- `/opt2/ronin/storage` - Video recordings
- `/opt2/ronin/ml_models` - ONNX model cache

The CPU compose file uses named Docker volumes instead.

### Database Migrations

Migrations run automatically on backend startup via `docker-entrypoint.sh`:
```bash
# Manual migration (inside container)
docker compose exec backend alembic upgrade head

# Create new migration
docker compose exec backend alembic revision --autogenerate -m "description"

# Check migration status
docker compose exec backend alembic current
```

### Profiles

- **Default profile**: Starts postgres, backend, frontend, live-detection, transcode-worker
- **historical profile**: Adds ml-worker for processing completed recordings

```bash
# Enable historical processing
docker compose --profile historical up -d

# Disable it later
docker compose stop ml-worker
```