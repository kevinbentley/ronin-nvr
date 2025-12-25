# System Dependencies

This document tracks external system dependencies required to run RoninNVR.

## Required Software

### Python
- **Version**: 3.11 or higher
- **Installation**:
  - macOS: `brew install python@3.11`
  - Ubuntu/Debian: `sudo apt install python3.11 python3.11-venv`

### PostgreSQL
- **Version**: 14 or higher
- **Installation**:
  - macOS: `brew install postgresql@14`
  - Ubuntu/Debian: `sudo apt install postgresql-14`
- **Setup**:
  ```bash
  # Create database and user
  createdb ronin_nvr
  createuser ronin_nvr_user
  psql -c "ALTER USER ronin_nvr_user WITH PASSWORD 'your_password';"
  psql -c "GRANT ALL PRIVILEGES ON DATABASE ronin_nvr TO ronin_nvr_user;"
  ```

### FFmpeg
- **Version**: 5.x or higher (with libx264 and libx265 support)
- **Installation**:
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
- **Verify installation**:
  ```bash
  ffmpeg -version
  ffmpeg -encoders | grep -E "libx264|libx265"
  ```

### Node.js (for frontend)
- **Version**: 18 or higher
- **Installation**:
  - macOS: `brew install node@18`
  - Ubuntu/Debian: Use NodeSource repository or nvm

## Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# Database
DATABASE_URL=postgresql+asyncpg://ronin_nvr_user:your_password@localhost:5432/ronin_nvr

# Storage
STORAGE_ROOT=/path/to/storage

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=true
```

## Development Setup

1. Install system dependencies (see above)
2. Set up the database
3. Create `.env` file
4. Run `./setup_venv.sh` to create Python environment
5. Run migrations: `cd backend && alembic upgrade head`
6. Start server: `cd backend && source venv/bin/activate && uvicorn app.main:app --reload`

## Platform Notes

### macOS
- Tested on macOS Ventura and Sonoma
- Use Homebrew for package management

### Linux
- Tested on Ubuntu 22.04 LTS
- Ensure PostgreSQL service is running: `sudo systemctl start postgresql`
