# API Testing Guide for `convert-file`

This guide provides `curl` commands to test the `convert-file` endpoint, covering both successful conversions and various error conditions.

## Prerequisites

Ensure the Flask server is running:

```bash
python docs_converter.py
```

The server typically runs on `http://localhost:5001`.

> [!IMPORTANT]
> **Terminal Selection**: The commands below use standard `curl` syntax intended for **Bash** (WSL, Git Bash, or macOS/Linux).
> 
> If you are using **PowerShell**:
> *   Use `curl.exe` instead of `curl` to bypass the PowerShell alias.
> *   Example: `curl.exe -X POST ...`
> *   Or simply open a **WSL** terminal to run these commands as written.

---

## 1. Testing Successful Conversion

### Option A: Upload a File (Recommended)
This uploads a local file directly to the converter.

```bash
# Replace 'path/to/document.pdf' with a real PDF or DOCX file
curl -X POST http://localhost:5001/docling/convert-file \
  -F "file=@path/to/document.pdf"
```

### Option B: Convert by File ID
If you have a file already indexed (visible in `/list-files`), you can convert it by ID.

```bash
# Replace with a valid file_id from /list-files
curl -X POST http://localhost:5001/docling/convert-file \
  -H "Content-Type: application/json" \
  -d '{"file_id": "YOUR_VALID_FILE_ID_HERE"}'
```

---

## 2. Testing Error Conditions

Use these commands to verify the error handling logic.

### Scenario 1: Unsupported File Format (`UNSUPPORTED_FORMAT`)
Upload a file with an extension other than `.pdf` or `.docx` (e.g., `.txt`).

1. Create a dummy text file:
   ```bash
   echo "This is a test" > invalid_file.txt
   ```

2. Upload it:
   ```bash
   curl -X POST http://localhost:5001/docling/convert-file \
     -F "file=@invalid_file.txt"
   ```

**Expected Output:**
```json
{
  "success": false,
  "error": true,
  "error_type": "UNSUPPORTED_FORMAT",
  "error_message": "Unsupported file type: .txt ...",
  ...
}
```

### Scenario 2: File ID Not Found (`FILE_ID_NOT_FOUND`)
Request a conversion for a File ID that does not exist.

```bash
curl -X POST http://localhost:5001/docling/convert-file \
  -H "Content-Type: application/json" \
  -d '{"file_id": "non-existent-id-12345"}'
```

**Expected Output:**
```json
{
  "success": false,
  "error": true,
  "error_type": "FILE_ID_NOT_FOUND",
  ...
}
```

### Scenario 3: Missing File ID (`MISSING_FILE_ID`)
Send a request without a file or file_id.

```bash
curl -X POST http://localhost:5001/docling/convert-file \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Expected Output:**
```json
{
  "error": "Missing 'file_id' in JSON request body or 'file' in form data"
}
```
*(Note: This returns a 400 Bad Request directly)*

### Scenario 4: Corrupt File (`DOCLING_CONVERSION_FAILED`)
Upload a file that has a valid extension but invalid content (e.g., random bytes).

1. Create a corrupt PDF:
   ```powershell
   # PowerShell
   Set-Content -Path corrupt.pdf -Value "This is not a valid PDF file content"
   ```
   OR
   ```bash
   # Bash (if using WSL terminal)
   echo "This is not a valid PDF" > corrupt.pdf
   ```

2. Upload it:
   ```bash
   curl -X POST http://localhost:5001/docling/convert-file \
     -F "file=@corrupt.pdf"
   ```

**Expected Output:**
```json
{
  "success": false,
  "error": true,
  "error_type": "DOCLING_CONVERSION_FAILED",
  "error_message": "Docling conversion failed: ...",
  ...
}
```

### Scenario 5: Path Validation Error (`SECURITY_ERROR`)
*Note: This is harder to trigger via upload, as uploads are saved to temp. It applies when resolving IDs to paths outside the allowed directory.*

---

## Error Code Reference

The `error_type` field in the JSON response will contain one of the following codes:

| Error Code | Description |
| :--- | :--- |
| `UNSUPPORTED_FORMAT` | File extension is not .pdf or .docx |
| `FILE_ID_NOT_FOUND` | The provided `file_id` could not be resolved to a file |
| `FILE_NOT_FOUND` | The file path associated with the ID does not exist on disk |
| `SECURITY_ERROR` | The file path is outside the allowed `BASE_DIR` |
| `DOCLING_CONVERSION_FAILED` | The Docling library failed to convert the file |
| `DOCX_CONVERSION_FAILED` | Native DOCX conversion failed (if fallback used) |
| `UNEXPECTED_ERROR` | An unhandled exception occurred |
