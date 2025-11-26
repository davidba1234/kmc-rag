# Visual Guide: Error Control Flow

## Single File Conversion Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Request: POST /docling/convert-file                            │
│  Body: { "file_path": "...", ... }                              │
└────────────────────┬────────────────────────────────────────────┘
                     │
         ┌───────────▼──────────────┐
         │ Parse JSON Request       │
         │ ✓ Valid JSON?            │
         └───────────┬──────────────┘
                     │
            ┌────────▼──────────┐
            │ NO → 400 Error    │
            │ (Bad Request)     │
            └───────────────────┘
                     │
            ┌────────▼──────────┐
            │ YES → Continue    │
            └────────┬──────────┘
                     │
    ┌────────────────▼────────────────┐
    │ Validate File Path              │
    │ ├─ File exists?                 │
    │ ├─ Within BASE_DIR?             │
    │ └─ Readable?                    │
    └────────────┬──────────┬─────────┘
                 │          │
        ┌────────▼──┐    ┌──▼──────────────┐
        │ NO        │    │ YES → Continue   │
        │ ↓         │    └────────┬─────────┘
        │ 200+error │             │
        │ flag      │    ┌────────▼─────────┐
        │ type:     │    │ Get File Type    │
        │ FILE_NOT_ │    │ (.pdf/.docx?)    │
        │ FOUND     │    └────────┬─────────┘
        └──────────┘             │
                        ┌────────▼─────────┐
                        │ UNSUPPORTED?     │
                        └────────┬─────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ NO → Continue           │
                    │ YES → 200+error (       │
                    │       UNSUPPORTED_      │
                    │       FORMAT)           │
                    └────────┬─────────────────┘
                             │
                 ┌───────────▼───────────┐
                 │ Count Images (PDF)    │
                 │ ⚠️ Warn if fails      │
                 └───────────┬───────────┘
                             │
            ┌────────────────▼────────────────┐
            │ Extract/Generate GUID           │
            │ ├─ Try PDF metadata             │
            │ ├─ Try DOCX settings            │
            │ ├─ Fall back to filename        │
            │ └─ ⚠️ Warn if extraction fails  │
            └────────────┬─────────────────────┘
                         │
            ┌────────────▼────────────────┐
            │ Select Converter            │
            │ ├─ PDF → Docling            │
            │ ├─ DOCX → Check if complex  │
            │ └─ Error if other type      │
            └────────────┬────────────────┘
                         │
        ┌────────────────▼─────────────────┐
        │ Conversion Path Decision          │
        └────────────┬──────────┬──────────┘
                     │          │
         ┌───────────▼──┐   ┌───▼───────────┐
         │ Native DOCX  │   │ Docling       │
         │ Parser       │   │ Converter     │
         └───────────┬──┘   └───┬───────────┘
                     │          │
       ┌─────────────▼──┐   ┌────▼────────────┐
       │ docx_to_json() │   │ Docling .convert│
       │ ↓ Parse DOCX   │   │ ↓ Heavy proc    │
       │ ✓ Extract text │   │ ✓ Extract text  │
       │ ✓ Extract table│   │ ✓ Export dict   │
       │ ✓ Get metadata │   │ ✓ Get markdown  │
       └────────┬───────┘   │ ✓ Image support │
                │           └────┬────────────┘
                │                │
    ┌───────────▼────────────────▼──┐
    │ Prepare Metadata               │
    │ ├─ Generate AI title           │
    │ ├─ Generate AI description     │
    │ ├─ ⚠️ Warn if AI generation    │
    │ │  fails (continue with None)  │
    │ └─ Count paragraphs, tables    │
    └────────────┬───────────────────┘
                 │
    ┌────────────▼──────────────┐
    │ Chunking (if enabled)     │
    │ ✓ Perform chunking        │
    │ ⚠️ Warn if fails (empty)  │
    └────────────┬──────────────┘
                 │
    ┌────────────▼──────────────────────┐
    │ Build Success Response            │
    │ HTTP 200 OK                       │
    │ {                                 │
    │   "success": true,                │
    │   "error": false,                 │
    │   "full_text": "...",             │
    │   "metadata": {...},              │
    │   "chunks": [...]                 │
    │ }                                 │
    └───────────────────────────────────┘
```

---

## Error Handling Decision Tree

```
                    START
                     │
    ┌────────────────▼─────────────────┐
    │ Critical Error?                  │
    │ (File not found, corrupt, etc.)  │
    └────────┬──────────────┬──────────┘
             │              │
        ┌────▼─────┐    ┌───▼──────────────┐
        │ YES       │    │ NO               │
        │ ↓         │    │ ↓                │
        │ Return    │    │ Log as WARNING   │
        │ 200+      │    │ Continue with    │
        │ error     │    │ fallback/None    │
        │ flag      │    │ values           │
        └──────────┘    └───┬──────────────┘
                            │
                    ┌───────▼────────┐
                    │ SUCCESS        │
                    │ Return 200+OK  │
                    │ with result    │
                    └────────────────┘
```

---

## Batch Processing Flow

```
┌────────────────────────────────────┐
│ Request: POST /docling/convert-all │
└────────────────┬───────────────────┘
                 │
    ┌────────────▼──────────────┐
    │ Find all supported files  │
    │ (*.docx, *.pdf)           │
    └────────────┬──────────────┘
                 │
        ┌────────▼───────────┐
        │ For EACH file:     │
        └────────┬───────────┘
                 │
    ┌────────────▼──────────────┐
    │ Check file exists?        │
    │ YES → Continue            │
    │ NO → Mark SKIPPED         │
    └────────────┬──────────────┘
                 │
    ┌────────────▼──────────────┐
    │ Check readable?           │
    │ (permissions OK)          │
    │ YES → Continue            │
    │ NO → Mark SKIPPED         │
    └────────────┬──────────────┘
                 │
    ┌────────────▼──────────────┐
    │ Attempt Conversion        │
    └────────┬──────────┬───────┘
             │          │
    ┌────────▼───┐  ┌───▼─────────────┐
    │ SUCCESS    │  │ FAILURE         │
    │ ↓          │  │ ↓               │
    │ Mark       │  │ Mark FAILED     │
    │ COMPLETED  │  │ Log error type  │
    │ Record ok  │  │ Continue loop   │
    │ Continue   │  │                 │
    │ loop       │  └─────────────────┘
    └────┬───────┘
         │
    ┌────▼────────────────────────────┐
    │ ALL FILES PROCESSED             │
    │ ↓                               │
    │ Aggregate Results:              │
    │ ├─ Total files                  │
    │ ├─ Successful count             │
    │ ├─ Failed count                 │
    │ ├─ Skipped count                │
    │ └─ Per-file status list         │
    └────┬─────────────────────────────┘
         │
    ┌────▼────────────────────────────┐
    │ Return 200 OK                   │
    │ {                               │
    │   "success": true,              │
    │   "message": "...",             │
    │   "results": {                  │
    │     "total_files": 50,          │
    │     "successful": 45,           │
    │     "failed": 3,                │
    │     "skipped": 2,               │
    │     "file_results": [...]       │
    │   }                             │
    │ }                               │
    └─────────────────────────────────┘
```

---

## n8n Workflow Integration Pattern

```
┌─────────────────────────────────────┐
│ n8n Workflow                        │
└────────┬────────────────────────────┘
         │
┌────────▼──────────────────────────────┐
│ HTTP Request                          │
│ POST /docling/convert-file            │
└────────┬──────────────────────────────┘
         │
┌────────▼──────────────────────────────┐
│ Receive Response                      │
│ (Always HTTP 200)                     │
└────────┬─────────────────┬────────────┘
         │                 │
    ┌────▼────────┐   ┌────▼──────────────┐
    │ Error flag? │   │ Check:            │
    │ error:true  │   │ response.body.    │
    └────┬────────┘   │ error === true    │
         │            └────┬──────────────┘
    ┌────▼───────────┐     │
    │ YES → Handle   │  ┌──▼─────────────┐
    │ Error Path     │  │ TRUE            │
    │ ├─ Log error   │  │ ↓               │
    │ ├─ Skip file   │  │ Handle Error    │
    │ ├─ Continue    │  │ ├─ Log          │
    │ │  to next     │  │ ├─ Skip file    │
    │ └─ file        │  │ └─ Continue     │
    └─────────────────  │                 │
                        └─────────────────┘
                             │
    ┌────────────────────────▼────────────┐
    │ FALSE                               │
    │ ↓                                   │
    │ Success Path                        │
    │ ├─ Extract full_text                │
    │ ├─ Extract metadata                 │
    │ ├─ Save to database                 │
    │ └─ Continue to next step            │
    └─────────────────────────────────────┘
```

---

## Error Type Categorization

```
Error Types
│
├─ FILE ISSUES
│  ├─ FILE_NOT_FOUND (file deleted/missing)
│  ├─ PERMISSION_ERROR (locked/unreadable)
│  └─ SECURITY_ERROR (outside allowed dir)
│
├─ FORMAT ISSUES
│  ├─ UNSUPPORTED_FORMAT (.doc, .txt, etc.)
│  ├─ FILE_TYPE_ERROR (can't detect type)
│  └─ DOCX_PARSING_ERROR (corrupt DOCX)
│
├─ CONVERSION ISSUES
│  ├─ DOCX_CONVERSION_ERROR (native parser failed)
│  ├─ DOCLING_CONVERSION_ERROR (heavy parser failed)
│  ├─ TEXT_EXTRACTION_ERROR (can't get content)
│  └─ DOCUMENT_EXPORT_ERROR (export to dict failed)
│
├─ PROCESSING ISSUES
│  ├─ GUID_ERROR (can't generate GUID)
│  ├─ CONVERTER_SELECTION_ERROR (choose parser)
│  └─ MISSING_TOKEN (selection_token required)
│
└─ UNEXPECTED
   └─ UNEXPECTED_ERROR (unhandled exception)

Legend:
✓ = Handled gracefully (skip file)
⚠ = Warning only (use fallback/None)
✗ = Fatal (stop processing)
```

---

## Response Status Codes

```
HTTP Status by Situation:
│
├─ 200 OK (Success with data)
│  └─ "success": true, "error": false
│     Full document conversion result
│
├─ 200 OK (Success with error flag)
│  └─ "success": false, "error": true
│     File couldn't be converted
│     Includes: error_type, error_message
│     Example: DOCX_PARSING_ERROR
│
├─ 200 OK (Batch results)
│  └─ "success": true, "results": {...}
│     Per-file status (completed/failed/skipped)
│     Summary counts
│     Detailed file_results array
│
├─ 400 Bad Request
│  └─ Invalid JSON in request
│     Missing required fields
│     (Returns error message, not error flag)
│
└─ 500 Internal Server Error
   └─ (Should NOT happen with new code)
      If it does → check error handling
```

---

## Error Recovery Strategies

```
┌─ Non-Fatal Error (Warn, Continue)
│  ├─ Image count fails → raw_total_images = 0
│  ├─ GUID extract fails → Use filename fallback
│  ├─ Metadata extract fails → Use None values
│  ├─ AI generation fails → ai_title = None
│  ├─ Chunking fails → chunks = []
│  └─ → Return 200 + success (partial result)
│
└─ Fatal Error (Return Error Response)
   ├─ File not found → Skip file
   ├─ File not readable → Skip file
   ├─ Format unsupported → Skip file
   ├─ DOCX corrupt → Skip file
   ├─ Docling fails → Skip file
   ├─ Text extraction fails → Skip file
   └─ → Return 200 + error flag

Batch Processing:
├─ Accumulate skipped files
├─ Accumulate failed files
├─ Continue loop
└─ Return aggregated results
```

---

## Monitoring & Alerts

```
Key Metrics:
│
├─ Success Rate (%)
│  └─ successful / total_files
│
├─ Failure Rate (%)
│  └─ failed / total_files
│
├─ Skip Rate (%)
│  └─ skipped / total_files
│
├─ Error Distribution
│  ├─ FILE_NOT_FOUND: 10%
│  ├─ DOCX_PARSING_ERROR: 45%
│  ├─ DOCLING_CONVERSION_ERROR: 30%
│  └─ Other: 15%
│
├─ Processing Time
│  ├─ Avg per file
│  ├─ Median
│  └─ P95/P99
│
└─ Retry Rate
   └─ Files that fail once but succeed later
```

---

## Summary Table

| Aspect | Before | After |
|--------|--------|-------|
| **File Error Response** | HTTP 500 | HTTP 200 + error flag |
| **App Behavior** | Crashes | Continues |
| **n8n Workflow** | Stops | Continues |
| **Error Info** | Generic | Specific type + details |
| **Batch Processing** | Stops at 1st error | Completes all, reports summary |
| **Per-file Tracking** | None | Detailed status for each file |
| **Recovery** | Manual | Automatic skip |
| **Logging** | Minimal | Comprehensive |
| **Fallbacks** | None | Multiple strategies |

This implementation makes your app **production-ready** for resilient bulk file processing!
