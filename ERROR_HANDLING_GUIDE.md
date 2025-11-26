# Error Handling Guide

## Overview

The application now includes comprehensive error control to gracefully handle corrupt or unreadable files. Instead of crashing with a 500 Error, the app will return a **200 OK response with an error flag**, allowing n8n to continue processing the next file.

## Key Changes

### 1. Error Response Helper Function

A new function `create_error_response()` provides standardized error responses:

```python
def create_error_response(file_path, error_message, error_type="CONVERSION_ERROR", details=None):
    """
    Creates a standardized error response that returns 200 OK with error flag.
    """
    response = {
        "success": False,
        "error": True,
        "error_type": error_type,
        "error_message": error_message,
        "file_path": file_path,
        "filename": os.path.basename(file_path) if file_path else "unknown",
        "conversion_status": "failed",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "error_details": details  # optional extra info
    }
    return jsonify(response), 200  # Always 200 OK!
```

**Benefits:**
- Consistent error structure across all endpoints
- HTTP 200 OK allows n8n to process the response without treating it as a failure
- Error flag (`"error": True`) lets n8n identify problematic files
- Detailed error types help with debugging and logging

### 2. Error Types

The system uses specific error types for better diagnostics:

| Error Type | Description | Recovery |
|-----------|-------------|----------|
| `FILE_NOT_FOUND` | File does not exist or was deleted | Skip file |
| `SECURITY_ERROR` | File path outside allowed directory | Skip file |
| `FILE_TYPE_ERROR` | Could not determine file extension | Skip file |
| `UNSUPPORTED_FORMAT` | File format not supported (.doc, .txt, etc.) | Skip file |
| `DOCX_CONVERSION_ERROR` | Native DOCX parser failed | Skip file |
| `DOCX_PARSING_ERROR` | DOCX content could not be parsed | Skip file |
| `TEXT_EXTRACTION_ERROR` | Could not extract text from document | Skip file |
| `DOCLING_CONVERSION_ERROR` | Docling converter crashed | Skip file |
| `DOCUMENT_EXPORT_ERROR` | Failed to export document to dict/markdown | Skip file |
| `GUID_ERROR` | Failed to generate/extract GUID | Continue (fallback to filename) |
| `CONVERTER_SELECTION_ERROR` | Could not decide between native/docling | Skip file |
| `PERMISSION_ERROR` | File is not readable | Skip file |
| `UNEXPECTED_ERROR` | Unhandled exception | Skip file |

### 3. Single File Conversion (`/docling/convert-file`)

#### Response on Success
```json
{
  "success": true,
  "error": false,
  "used_converter": "docling",
  "file_path": "/path/to/file.pdf",
  "filename": "file.pdf",
  "conversion_status": "completed",
  "full_text": "...",
  "metadata": { ... },
  "chunks": [ ... ]
}
```

#### Response on Failure (HTTP 200)
```json
{
  "success": false,
  "error": true,
  "error_type": "DOCX_CONVERSION_ERROR",
  "error_message": "Failed to convert DOCX: [specific reason]",
  "file_path": "/path/to/file.docx",
  "filename": "file.docx",
  "conversion_status": "failed",
  "timestamp": "2025-11-19T21:43:44.123456Z",
  "error_details": {
    "step": "docx_to_json"
  }
}
```

#### How n8n Should Handle
```javascript
// In n8n workflow:
if (response.body.error === true) {
  // Log the error but continue
  console.log(`File failed: ${response.body.error_type} - ${response.body.error_message}`);
  // Process next file
} else if (response.body.success === true) {
  // Process the converted content
  let converted = response.body;
}
```

### 4. Batch Conversion (`/docling/convert-all`)

#### Request
```json
POST /docling/convert-all
```

#### Response (Always HTTP 200)
```json
{
  "success": true,
  "message": "Batch conversion complete: 45 succeeded, 3 failed, 2 skipped",
  "results": {
    "total_files": 50,
    "successful": 45,
    "failed": 3,
    "skipped": 2,
    "file_results": [
      {
        "file_path": "/path/to/file1.pdf",
        "filename": "file1.pdf",
        "status": "completed",
        "success": true,
        "error": null,
        "error_type": null
      },
      {
        "file_path": "/path/to/corrupt.docx",
        "filename": "corrupt.docx",
        "status": "failed",
        "success": false,
        "error": "Conversion failed: Unable to parse DOCX structure",
        "error_type": "DOCX_PARSING_ERROR"
      },
      {
        "file_path": "/path/to/locked.pdf",
        "filename": "locked.pdf",
        "status": "skipped",
        "success": false,
        "error": "File is not readable",
        "error_type": "PERMISSION_ERROR"
      }
    ]
  }
}
```

#### How n8n Should Handle
```javascript
// In n8n workflow:
let results = response.body.results;

// Process successful conversions
results.file_results
  .filter(r => r.success === true)
  .forEach(r => {
    console.log(`✓ Converted: ${r.filename}`);
  });

// Log failures for review
results.file_results
  .filter(r => r.status === "failed")
  .forEach(r => {
    console.log(`✗ Failed: ${r.filename} - ${r.error_type}`);
  });

// Log skipped files
results.file_results
  .filter(r => r.status === "skipped")
  .forEach(r => {
    console.log(`⊘ Skipped: ${r.filename} - ${r.error}`);
  });
```

## Error Handling Flow

### Single File Conversion

```
Request → Validate JSON → Validate Path → Check Permissions
    ↓         (400)           (200+error)    (200+error)
    ├─ Determine file type (200+error on failure)
    ├─ Count images (warn, don't fail)
    ├─ Extract/Generate GUID (warn, use fallback)
    ├─ Select converter (200+error on failure)
    └─ Convert
        ├─ Native DOCX: docx_to_json → text → metadata (200+error per step)
        ├─ Docling: convert → export → enhance (200+error per step)
        └─ Success: return full document (200+success)
```

### Batch Conversion

```
Request → For each file
    ├─ Check file exists (200+skipped)
    ├─ Check readable (200+skipped)
    └─ Convert
        ├─ Success: record (200+success)
        ├─ Fail: record error + continue (200+failed)
        └─ Error: record error + continue (200+error)
    → Aggregate results → Return summary (200+results)
```

## Non-Fatal vs Fatal Errors

### Non-Fatal (Logs warning, continues)
- Image counting fails (raw_total_images = 0)
- GUID extraction fails (fallback to filename)
- Custom metadata extraction fails (None values)
- AI title/description generation fails (None values)
- Chunking fails (returns empty chunks array)
- Image manifest building fails (no images)
- Image analysis fails (no descriptions)

### Fatal (Returns 200+error)
- File does not exist
- File path outside allowed directory
- File not readable (permission denied)
- File type not supported
- DOCX parsing fails
- Docling conversion fails
- Document export fails
- Text extraction fails

## Testing Error Scenarios

### Test 1: Corrupt DOCX
```bash
curl -X POST http://localhost:5001/docling/convert-file \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/corrupt.docx"}'
```

**Expected Response (200):**
```json
{
  "success": false,
  "error": true,
  "error_type": "DOCX_PARSING_ERROR",
  "error_message": "Failed to convert DOCX: ...",
  "file_path": "/path/to/corrupt.docx",
  "conversion_status": "failed"
}
```

### Test 2: Missing File
```bash
curl -X POST http://localhost:5001/docling/convert-file \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/nonexistent/file.pdf"}'
```

**Expected Response (200):**
```json
{
  "success": false,
  "error": true,
  "error_type": "FILE_NOT_FOUND",
  "error_message": "File not found: /nonexistent/file.pdf",
  "conversion_status": "failed"
}
```

### Test 3: Batch with Mixed Files
```bash
curl -X POST http://localhost:5001/docling/convert-all
```

**Expected Response (200):**
- Good files: `"status": "completed"`
- Corrupt files: `"status": "failed"`
- Missing files: `"status": "skipped"`
- Summary counts each category

## Implementation Details

### Wrapped Functions
All major conversion steps are now wrapped in try-except blocks:

1. **JSON parsing** → Returns 400 on invalid JSON
2. **File validation** → Returns 200+error on path issues
3. **GUID operations** → Warns and falls back
4. **Converter selection** → Returns 200+error on failure
5. **Image counting** → Warns but doesn't fail
6. **Docx conversion** → Returns 200+error with detailed step
7. **Docling conversion** → Returns 200+error with exception type
8. **Text extraction** → Returns 200+error with context
9. **Metadata generation** → Warns or skips non-critical data
10. **Chunking** → Warns but returns empty array on failure

### Logging Strategy

- **ERROR level**: Fatal issues that cause file skipping
- **WARNING level**: Non-fatal issues (missing metadata, etc.)
- **INFO level**: Normal flow (file selection, converter choice)

Example logs:
```
ERROR - [DOCX_PARSING_ERROR] /path/to/corrupt.docx: Failed to convert DOCX: [reason]
WARNING - Could not extract GUID from /path/to/file.pdf: [reason]
INFO - convert-file: /path/to/file.pdf selected converter = docling
```

## n8n Integration Tips

### Basic Error Handling
```javascript
const convertFile = async (filePath) => {
  const response = await $http.post('http://localhost:5001/docling/convert-file', {
    file_path: filePath
  });
  
  if (response.body.error === true) {
    // File failed - log and skip
    $log.error(`Failed: ${response.body.error_type}`);
    return null;
  }
  
  return response.body;
};
```

### Batch Processing
```javascript
const convertAll = async () => {
  const response = await $http.post('http://localhost:5001/docling/convert-all', {});
  
  const results = response.body.results;
  const successful = results.file_results.filter(r => r.success);
  const failed = results.file_results.filter(r => r.status === 'failed');
  
  $log.info(`Converted ${successful.length} files, ${failed.length} failed`);
  
  return {
    successful: successful,
    failed: failed,
    summary: response.body.message
  };
};
```

### Error Monitoring
```javascript
// Log all errors to a database or monitoring service
const logConversionError = async (fileResult) => {
  if (fileResult.status === 'failed' || fileResult.status === 'error') {
    await $http.post('http://monitoring/log', {
      file: fileResult.filename,
      errorType: fileResult.error_type,
      errorMessage: fileResult.error,
      timestamp: new Date()
    });
  }
};
```

## Migration Path for Existing Workflows

If you have existing n8n workflows expecting 500 errors:

**Before:**
```javascript
try {
  const response = await $http.post('/docling/convert-file', data);
  // process
} catch (error) {
  if (error.statusCode === 500) {
    // handle error
  }
}
```

**After:**
```javascript
const response = await $http.post('/docling/convert-file', data);
if (response.body.error === true) {
  // handle error (now status 200 with error flag)
} else {
  // process
}
```

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Corrupt File** | 500 Error, server crashes | 200 OK with error flag |
| **n8n Response** | Treated as failure, workflow stops | Processed as normal, error flag checked |
| **Error Info** | Generic error message | Specific error type + details |
| **Batch Processing** | Stops at first error | Continues, reports summary |
| **Recovery** | Manual intervention needed | Automatic skip and continue |

The application is now **resilient and production-ready** for handling bulk file conversions with n8n!
