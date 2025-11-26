# n8n Integration Examples

## Overview
These examples show how to use the error-resilient API endpoints in n8n workflows.

---

## Example 1: Simple File Conversion with Error Handling

**Use Case:** Convert a single file and log any errors

### n8n Workflow Steps

1. **HTTP Request** (POST)
   ```
   URL: http://localhost:5001/docling/convert-file
   Method: POST
   Body:
   {
     "file_path": "{{$input.first().file_path}}",
     "do_chunking": true,
     "chunking_options": {}
   }
   ```

2. **IF** - Check for error
   ```javascript
   // Condition:
   {{$input.first().body.error === true}}
   
   // TRUE branch: Log error and stop
   - Set node: Create error log entry
   - Log: {{$input.first().body.error_type}} - {{$input.first().body.error_message}}
   
   // FALSE branch: Process success
   - Set node: Extract converted data
   ```

3. **Extract Data** (Success branch)
   ```javascript
   {{$input.first().body.full_text}}
   {{$input.first().body.metadata}}
   {{$input.first().body.chunks}}
   ```

---

## Example 2: Batch File Processing with Skip Logic

**Use Case:** Convert multiple files, skip errors, continue with next

### n8n Workflow

1. **Get file list** (or use array of file paths)
   ```javascript
   [
     "/path/to/file1.pdf",
     "/path/to/file2.docx",
     "/path/to/corrupted.pdf",
     "/path/to/file4.docx"
   ]
   ```

2. **Loop over files** (Loop node)
   ```
   For each file in array:
   ```

3. **HTTP Request** (inside loop)
   ```
   URL: http://localhost:5001/docling/convert-file
   Body:
   {
     "file_path": "{{$nodeExecutionData[0].item.value}}"
   }
   ```

4. **Switch** - Check error flag
   ```javascript
   // Case 1: Error occurred
   if ({{$input.first().body.error === true}}) {
     - Log error: {{$input.first().body.filename}} failed
     - Continue to next file
   }
   
   // Case 2: Success
   if ({{$input.first().body.success === true}}) {
     - Save to database
     - Add to success array
     - Continue to next file
   }
   ```

5. **Aggregate Results** (After loop)
   ```javascript
   // Collect all successful conversions
   let successful = [];
   let failed = [];
   
   // Process results from loop
   ```

---

## Example 3: Batch Conversion with Summary Report

**Use Case:** Convert all files at once, get detailed per-file results

### n8n Workflow

1. **HTTP Request** (POST /docling/convert-all)
   ```
   URL: http://localhost:5001/docling/convert-all
   Method: POST
   ```

2. **Process Results**
   ```javascript
   let response = $input.first().body;
   let results = response.results;
   
   return {
     total: results.total_files,
     successful: results.successful,
     failed: results.failed,
     skipped: results.skipped,
     summary: response.message,
     details: results.file_results
   };
   ```

3. **Split Results by Status**
   ```javascript
   // Successful files
   let successful = results.file_results.filter(f => f.status === 'completed');
   
   // Failed files
   let failed = results.file_results.filter(f => f.status === 'failed');
   
   // Skipped files
   let skipped = results.file_results.filter(f => f.status === 'skipped');
   ```

4. **Save to Databases**
   - **DB1: Successful conversions** - loop through `successful` array
   - **DB2: Failed conversions** - loop through `failed` array
   - **DB3: Audit log** - timestamp, summary, counts

5. **Send Summary Email**
   ```
   Subject: Conversion Report - {{result.total}} files processed
   
   Body:
   ✓ Successful: {{result.successful}}
   ✗ Failed: {{result.failed}}
   ⊘ Skipped: {{result.skipped}}
   
   Summary: {{result.summary}}
   
   Failed Files:
   {{result.details.filter(f => f.status === 'failed').map(f => 
     '- ' + f.filename + ': ' + f.error_type).join('\n')}}
   ```

---

## Example 4: Advanced Error Recovery with Retry

**Use Case:** Retry failed files with different settings, track persistent failures

### n8n Workflow

1. **Initial Conversion** (batch or single)
2. **Identify Failures**
   ```javascript
   let failed = results.file_results.filter(f => f.status === 'failed');
   ```

3. **Retry Loop** (with different parameters)
   ```javascript
   for (let file of failed) {
     // Retry with docling even if simple
     let retry = {
       file_path: file.file_path,
       force_refresh: true,
       do_chunking: false  // Disable chunking to reduce failures
     };
   }
   ```

4. **Track Retries**
   ```javascript
   let persistentFailures = [];
   for (let retryResult of retryResults) {
     if (retryResult.error === true) {
       persistentFailures.push({
         file: retryResult.filename,
         original_error: retryResult.error_type,
         retry_error: retryResult.error_type,
         message: retryResult.error_message
       });
     }
   }
   ```

5. **Alert on Persistent Failures**
   ```javascript
   if (persistentFailures.length > 0) {
     // Send alert to team
     await $http.post('slack-webhook-url', {
       text: `⚠️ ${persistentFailures.length} files failed to convert after retry`,
       blocks: persistentFailures.map(f => ({
         type: "section",
         text: {
           type: "mrkdwn",
           text: `*${f.file}*\nError: ${f.original_error}`
         }
       }))
     });
   }
   ```

---

## Example 5: Conditional Processing Based on Error Type

**Use Case:** Handle different error types differently

### n8n Workflow Logic

```javascript
const response = {{$input.first().body}};

if (response.error === true) {
  switch(response.error_type) {
    case 'FILE_NOT_FOUND':
      // Log as missing file, skip
      await logMissingFile(response.file_path);
      break;
      
    case 'UNSUPPORTED_FORMAT':
      // Notify user format not supported
      await notifyUser(`File format not supported: ${response.file_path}`);
      break;
      
    case 'DOCX_CONVERSION_ERROR':
    case 'DOCX_PARSING_ERROR':
      // Try to repair DOCX or request user re-save
      await requestDocxRepair(response.file_path);
      break;
      
    case 'DOCLING_CONVERSION_ERROR':
      // PDF issue - try with lower quality settings
      await retryPdfWithSettings(response.file_path, {quality: 'low'});
      break;
      
    case 'PERMISSION_ERROR':
      // File locked - retry after delay
      await retryAfterDelay(response.file_path, 5000);
      break;
      
    default:
      // Unexpected error - log and alert
      await logUnexpectedError(response);
  }
}
```

---

## Example 6: Data Enrichment Pipeline

**Use Case:** Convert files, extract structured data, enhance metadata

### n8n Workflow

1. **Convert File** (endpoint: `/docling/convert-file`)
   ```
   Returns: full_text, metadata, chunks
   ```

2. **If Success** (check error flag)
   ```
   - Extract metadata
   - Split full_text into paragraphs
   - Identify key topics
   - Extract entities
   ```

3. **Prepare Database Record**
   ```javascript
   {
     file_id: {{$input.first().body.metadata.file_id}},
     filename: {{$input.first().body.filename}},
     content: {{$input.first().body.full_text}},
     chunks: {{$input.first().body.chunks}},
     metadata: {{$input.first().body.metadata}},
     processing_status: "completed",
     processed_at: {{new Date().toISOString()}}
   }
   ```

4. **Insert to Database**
   ```
   INSERT INTO converted_documents ...
   ```

5. **If Failure** (error flag true)
   ```javascript
   {
     file_id: {{$input.first().body.file_path}},
     filename: {{$input.first().body.filename}},
     error_type: {{$input.first().body.error_type}},
     error_message: {{$input.first().body.error_message}},
     processing_status: "failed",
     failed_at: {{new Date().toISOString()}}
   }
   ```

6. **Insert to Error Log**
   ```
   INSERT INTO conversion_errors ...
   ```

---

## Example 7: Chunking with Error Handling

**Use Case:** Convert document and chunk it for vector embedding

### n8n Workflow

1. **Convert with Chunking Enabled**
   ```json
   {
     "file_path": "{{$input.first().file_path}}",
     "do_chunking": true,
     "chunking_options": {
       "method": "semantic",
       "max_tokens": 1000,
       "overlap": 100
     }
   }
   ```

2. **Check Response**
   ```javascript
   // If error:
   if (response.error === true) {
     // Log without chunks
     return { success: false, error: response.error_type };
   }
   
   // If success but no chunks (chunking failed):
   if (response.success === true && (!response.chunks || response.chunks.length === 0)) {
     // Use full_text as single chunk
     response.chunks = [{
       content: response.full_text,
       chunk_index: 0,
       start_char: 0,
       end_char: response.full_text.length
     }];
   }
   ```

3. **Embed Each Chunk**
   ```javascript
   for (let chunk of response.chunks) {
     let embedding = await embedText(chunk.content);
     chunk.embedding = embedding;
   }
   ```

4. **Store in Vector Database**
   ```
   FOR EACH chunk:
     INSERT INTO vector_store (
       document_id,
       chunk_index,
       content,
       embedding
     ) VALUES (...)
   ```

---

## Example 8: Image Processing with Batch Conversion

**Use Case:** Extract and analyze images during batch conversion

### n8n Workflow

1. **Batch Convert** (endpoint: `/docling/convert-all`)

2. **Process PDF Results** (with images)
   ```javascript
   let results = response.results.file_results;
   
   let pdfResults = results.filter(r => r.file_path.endsWith('.pdf'));
   
   for (let pdf of pdfResults) {
     if (pdf.success) {
       // File was converted, images may have been processed
       let imageCount = {{$input.first().body.results}}.find(
         r => r.file_path === pdf.file_path
       ).image_count_analyzed;
       
       console.log(`${pdf.filename}: ${imageCount} images processed`);
     }
   }
   ```

3. **Handle PDFs with Images**
   ```javascript
   // Re-request with image selection
   let pdfWithImages = {
     file_path: pdf.file_path,
     image_mode: 'ask',
     do_chunking: false  // Images processing takes time
   };
   ```

---

## Error Handling Utilities (Reusable)

### Utility 1: Universal Error Logger
```javascript
async function logConversionError(response) {
  if (response.error === true) {
    return {
      timestamp: new Date().toISOString(),
      file: response.filename,
      file_path: response.file_path,
      error_type: response.error_type,
      error_message: response.error_message,
      error_details: response.error_details || {},
      severity: getSeverity(response.error_type)
    };
  }
  return null;
}

function getSeverity(errorType) {
  const criticalErrors = [
    'DOCX_CONVERSION_ERROR',
    'DOCLING_CONVERSION_ERROR',
    'TEXT_EXTRACTION_ERROR'
  ];
  
  return criticalErrors.includes(errorType) ? 'HIGH' : 'MEDIUM';
}
```

### Utility 2: Batch Summary Generator
```javascript
function generateBatchSummary(results) {
  return {
    total: results.total_files,
    successful: results.successful,
    failed: results.failed,
    skipped: results.skipped,
    success_rate: ((results.successful / results.total_files) * 100).toFixed(2),
    failed_by_type: groupByErrorType(results.file_results),
    processing_time: calculateTime()
  };
}

function groupByErrorType(fileResults) {
  const errors = fileResults.filter(r => r.status === 'failed');
  return errors.reduce((acc, r) => {
    acc[r.error_type] = (acc[r.error_type] || 0) + 1;
    return acc;
  }, {});
}
```

---

## Performance Tips

1. **Batch Operations**
   - Use `/docling/convert-all` for bulk files (more efficient)
   - Use `/docling/convert-file` for single files (lower latency)

2. **Disable Optional Features for Speed**
   ```json
   {
     "do_chunking": false,
     "image_mode": "skip"
   }
   ```

3. **Retry Strategy**
   - Don't retry immediately (files may be locked)
   - Wait 30-60 seconds between retries
   - Max 3 retries before permanent failure

4. **Parallel Processing**
   - Split files across multiple n8n instances
   - Use queue/batch system for large datasets

---

## Monitoring and Alerts

### Key Metrics to Track
- Success rate (%)
- Failed rate by error type
- Average processing time per file
- Retry rate
- Peak error times

### Alert Thresholds
- Success rate < 90%
- Any "UNEXPECTED_ERROR" occurs
- Batch processing time > 1 hour
- Same file fails repeatedly

---

## References

- See `ERROR_HANDLING_GUIDE.md` for complete error types
- See `ERROR_CONTROL_QUICK_REF.md` for quick reference
- API Endpoints: `/docling/convert-file`, `/docling/convert-all`
