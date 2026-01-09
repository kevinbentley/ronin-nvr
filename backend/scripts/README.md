# Backend Scripts

Utility scripts for managing RoninNVR installations.

## Camera Export/Import

These scripts allow you to export camera configurations from one installation and import them into another.

### Export Cameras

Export all configured cameras to a JSON file.

```bash
cd backend
source venv/bin/activate

# Basic export (passwords excluded)
python scripts/export_cameras.py -o cameras.json

# Include passwords (handle with care!)
python scripts/export_cameras.py -o cameras.json --include-passwords

# Pretty-print for readability
python scripts/export_cameras.py -o cameras.json --pretty
```

**Options:**
- `-o, --output` - Output file path (default: cameras.json)
- `--include-passwords` - Include camera passwords in export
- `--pretty` - Format JSON with indentation

**Exported Fields:**
- Connection: name, host, port, path, username, password (optional), transport
- Settings: recording_enabled
- ONVIF: onvif_port, onvif_enabled, onvif_profile_token, onvif_device_info, onvif_events_enabled

**Not Exported (runtime state):**
- id, status, last_seen, error_message, created_at, updated_at

### Import Cameras

Import camera configurations from a JSON file.

```bash
cd backend
source venv/bin/activate

# Preview what would happen (no changes made)
python scripts/import_cameras.py -i cameras.json --dry-run

# Import, fail if duplicates exist
python scripts/import_cameras.py -i cameras.json

# Import, skip existing cameras
python scripts/import_cameras.py -i cameras.json --skip-existing

# Import, update existing cameras
python scripts/import_cameras.py -i cameras.json --update-existing
```

**Options:**
- `-i, --input` - Input JSON file path (required)
- `--skip-existing` - Skip cameras that already exist (by name)
- `--update-existing` - Update cameras that already exist (by name)
- `--dry-run` - Validate and show what would happen without making changes

**Duplicate Handling:**
- Default: Fails if a camera with the same name exists
- `--skip-existing`: Leaves existing cameras unchanged
- `--update-existing`: Updates existing cameras with imported data

### Export File Format

```json
{
  "version": "1.0",
  "exported_at": "2026-01-08T22:00:00+00:00",
  "camera_count": 2,
  "includes_passwords": false,
  "cameras": [
    {
      "name": "Front Door",
      "host": "192.168.1.100",
      "port": 554,
      "path": "/stream1",
      "username": "admin",
      "transport": "tcp",
      "recording_enabled": true,
      "onvif_port": 80,
      "onvif_enabled": true,
      "onvif_profile_token": null,
      "onvif_device_info": null,
      "onvif_events_enabled": false
    }
  ]
}
```

### Typical Migration Workflow

1. **On source system:**
   ```bash
   python scripts/export_cameras.py -o cameras.json --include-passwords --pretty
   ```

2. **Transfer file** to target system securely (contains passwords!)

3. **On target system:**
   ```bash
   # Preview first
   python scripts/import_cameras.py -i cameras.json --dry-run

   # Then import
   python scripts/import_cameras.py -i cameras.json
   ```

4. **Delete the export file** after successful import (contains passwords)

### Security Notes

- By default, passwords are NOT included in exports
- When using `--include-passwords`, the export file contains plaintext credentials
- Store export files securely and delete after use
- Transfer files over secure channels only
