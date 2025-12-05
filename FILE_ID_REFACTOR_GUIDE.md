# FileID Refactoring Guide

## Overview
The `docx_converter_testing.py` script has been refactored to use **fileID as the primary unique identifier** instead of file paths. This change provides better portability, caching, and API cleanliness.

## Key Changes

### 1. **New Global Registry System**
- `FILE_ID_REGISTRY`: Maps fileID → file_path
- `FILE_PATH_CACHE`: Maps file_path → fileID (reverse lookup)

Functions to manage the registry:
- `register_file_id(file_id, file_path)` - Register a new mapping
- `resolve_file_path(file_id)` - Convert fileID to file path
- `get_file_id_or_register(file_path)` - Get or create fileID for a path

### 2. **Updated Main Endpoint: `/docling/convert-file`**

**Old Request:**
```json
{
  "file_path": "/full/path/to/document.pdf",
  "image_mode": "ask",
  "do_chunking": true
}
```

**New Request:**
```json
{
  "file_id": "my_document",
  "image_mode": "ask",
  "do_chunking": true
}
```

### 3. **New Helper Endpoints**

#### Register File and Get FileID
```
POST /file-id/register
Content-Type: application/json

{
  "file_path": "/path/to/document.docx"
}

Response:
{
  "success": true,
  "file_id": "document",
  "file_path": "/path/to/document.docx",
  "filename": "document.docx"
}
```

#### Resolve FileID to File Path
```
GET /file-id/resolve?file_id=document

Response:
{
  "success": true,
  "file_id": "document",
  "file_path": "/path/to/document.docx",
  "filename": "document.docx"
}
```

### 4. **Updated Error Responses**
All error responses now include `file_id`:
```json
{
  "success": false,
  "error": true,
  "file_id": "my_document",
  "file_path": "/path/to/file",
  "error_type": "FILE_NOT_FOUND",
  "error_message": "File not found",
  "timestamp": "2025-12-06T...",
  ...
}
```

### 5. **Metadata Changes**
Metadata now includes:
- `file_id`: The unique identifier for the file
- `guid_source`: Always "file_id" (simplified from previous "docx_settings", "pdf_metadata", "filename_fallback")
- `has_guid`: Always `true` when using fileID registry

## Migration Path for Existing Code

### For n8n Workflows:

**Before:**
```n8n
POST http://localhost:5001/docling/convert-file
{
  "file_path": "{{$json.path}}"
}
```

**After - Option 1 (Two-step):**
```n8n
// Step 1: Register file and get fileID
POST http://localhost:5001/file-id/register
{
  "file_path": "{{$json.path}}"
}

// Step 2: Use fileID for conversion
POST http://localhost:5001/docling/convert-file
{
  "file_id": "{{$json.body.file_id}}"
}
```

**After - Option 2 (Combined, if you track fileIDs):**
```n8n
POST http://localhost:5001/docling/convert-file
{
  "file_id": "{{$json.known_file_id}}"
}
```

## FileID Sources

FileIDs are generated in this priority order:
1. **PDF**: From custom `/AppGUID` metadata (if present)
2. **DOCX**: From `word/settings.xml` docId (if present)
3. **Fallback**: Derived from filename (spaces/hyphens replaced with underscores, lowercased)

Example fallbacks:
- `My Document.pdf` → `my_document`
- `Report-2025.docx` → `report_2025`

## Important Notes

⚠️ **Registry is In-Memory**
- File registrations are stored in-memory only
- Registry persists for the duration of the Flask process
- Registrations are lost on server restart
- All files should be re-registered after restart via `/file-id/register`

✅ **Backwards Compatibility**
- `create_error_response()` and other helpers accept either fileID or file_path
- Path detection is automatic (anything starting with `/` is treated as path)
- Graceful fallback to file registration if needed

## Updated Function Signatures

### Core Conversion
```python
# Before
def docling_convert_file():
    file_path = data['file_path']

# After
def docling_convert_file():
    file_id = data['file_id']
    file_path = resolve_file_path(file_id)
```

### Error Handling
```python
# Before & After (both work)
create_error_response(file_path, "Error message")
create_error_response(file_id, "Error message")
```

### Image Manifest Building
```python
# Before
image_manifest = build_pdf_image_manifest(file_path, guid)

# After
image_manifest = build_pdf_image_manifest(file_path, file_id)
```

## Response Structure Example

```json
{
  "success": true,
  "used_converter": "docling",
  "file_id": "quarterly_report",
  "file_path": "/mnt/.../quarterly_report.pdf",
  "filename": "quarterly_report.pdf",
  "conversion_status": "success",
  "full_text": "...",
  "metadata": {
    "file_id": "quarterly_report",
    "filename": "quarterly_report.pdf",
    "has_guid": true,
    "guid_source": "file_id",
    ...
  },
  "chunks": [...],
  "lastModified": "2025-12-06T...",
  "createdDate": "2025-12-06T...",
  "image_descriptions": [...]
}
```

## Testing

Quick test sequence:
```bash
# 1. Register a file
curl -X POST http://localhost:5001/file-id/register \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/test.pdf"}'

# 2. Use the returned file_id for conversion
curl -X POST http://localhost:5001/docling/convert-file \
  -H "Content-Type: application/json" \
  -d '{"file_id": "test", "image_mode": "force"}'

# 3. Resolve fileID back to path
curl "http://localhost:5001/file-id/resolve?file_id=test"
```

## Benefits of This Refactor

✅ **Cleaner API**: FileIDs are shorter and more semantic than full paths
✅ **Better Caching**: FileIDs can be used as cache keys more reliably
✅ **Security**: File paths not exposed in API responses by default
✅ **Portability**: FileIDs remain consistent even if file moves (within BASE_DIR)
✅ **Consistency**: All responses include file_id for easy tracking

## Future Enhancements

Potential improvements:
- Persist FILE_ID_REGISTRY to disk (JSON file)
- Add file_id validation endpoint
- Batch fileID registration
- Automatic registry cleanup for missing files
