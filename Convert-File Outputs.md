# Chat Conversation

Note: _This is purely the output of the chat conversation and does not contain any raw data, codebase snippets, etc. used to generate the output._

### User Input

can you give an explanation of the output returned by the function convert-file - all possibilities including the metadata added for successful conversions

*Listed directory [kmc-rag](file://wsl.localhost/Ubuntu/home/dba/kmc-rag) *

*Viewed [docs_converter.py](file://wsl.localhost/Ubuntu/home/dba/kmc-rag/docs_converter.py) *

*Viewed [docs_converter.py](file://wsl.localhost/Ubuntu/home/dba/kmc-rag/docs_converter.py) *

### Planner Response

The `convert-file` function (mapped to `/docling/convert-file`) has three main categories of output depending on the execution path: **Survey Mode**, **Successful Conversion**, and **Error**.

### 1. Successful Conversion Output
This is the standard response when a file is fully processed. It returns a JSON object containing the converted content, metadata, and execution stats.

**Top-Level Fields:**
*   `success`: `true`
*   `used_converter`: `"docling"`
*   [file_id](cci:1://file://wsl.localhost/Ubuntu/home/dba/kmc-rag/docs_converter.py:85:0-89:64): The unique identifier for the file (UUID).
*   [file_path](cci:1://file://wsl.localhost/Ubuntu/home/dba/kmc-rag/docs_converter.py:91:0-93:40): Absolute path to the file (null if it was a temporary upload).
*   `filename`: Name of the file.
*   `conversion_status`: Status from the Docling converter (e.g., "success").
*   `is_upload`: `true` if the file was uploaded directly, `false` if processed from the local file system.
*   `lastModified`: ISO timestamp of file modification.
*   `createdDate`: ISO timestamp of file creation.
*   `full_text`: The converted document content in Markdown format.
*   `document`: The raw structured dictionary exported by Docling (contains headers, body, tables, etc.).
*   `chunks`: A list of semantic chunks (if `do_chunking` was set to `true`).
*   [image_manifest](cci:1://file://wsl.localhost/Ubuntu/home/dba/kmc-rag/docs_converter.py:462:0-507:19): List of available images (if `image_mode="selected"` was used).
*   [image_descriptions](cci:1://file://wsl.localhost/Ubuntu/home/dba/kmc-rag/docs_converter.py:117:0-119:13): List of AI-generated descriptions for images (if vision analysis was run).
*   `summary`: A dictionary of statistics for this run.
    *   `total_files_processed`: 1
    *   `docx_count` / `pdf_count`: 1 or 0
    *   `images_total`: Total images found in the document.
    *   `images_selected`: Number of images selected for analysis.
    *   `images_analyzed`: Number of images successfully described by AI.
    *   `images_ignored`: Number of images skipped.

**Metadata Object ([metadata](cci:1://file://wsl.localhost/Ubuntu/home/dba/kmc-rag/docs_converter.py:179:0-208:28) field):**
The function generates a comprehensive metadata object attached to the response (and to each chunk).
*   **Identity & Location:**
    *   [file_id](cci:1://file://wsl.localhost/Ubuntu/home/dba/kmc-rag/docs_converter.py:85:0-89:64): UUID.
    *   `filename`: Original filename.
    *   `full_path`: Absolute path.
    *   `subfolder`: Name of the parent directory.
    *   `file_extension`: `.pdf` or `.docx`.
    *   `has_guid`: `true` (indicates a valid ID exists).
    *   `guid_source`: `"upload"` or `"file_id"`.
*   **Content Stats:**
    *   `file_size`: Size in bytes.
    *   `file_size_mb`: Size in MB.
    *   `word_count`: Total words in the converted text.
    *   `character_count`: Total characters.
    *   `paragraph_count`: Number of paragraphs.
    *   `table_count`: Number of tables detected.
    *   `image_count_total`: Total images in the file.
    *   `image_count_selected`: Images chosen for analysis.
    *   `image_count_analyzed`: Images successfully processed.
*   **Timestamps:**
    *   `lastModified`: File modification time (UTC).
    *   `createdDate`: File creation time (UTC).
    *   `DateReviewed`: Custom property extracted from DOCX (if present).
    *   `DateNext`: Custom property extracted from DOCX (if present).
*   **AI Enrichment:**
    *   [ai_title](cci:1://file://wsl.localhost/Ubuntu/home/dba/kmc-rag/docs_converter.py:293:0-335:42): Generated title based on content.
    *   `ai_description`: Generated 1-2 sentence summary.
*   **Origin:**
    *   `origin`: Source metadata from Docling (if available).

---

### 2. Survey Mode Output ("Ask Mode")
Returned when `image_mode` is set to `"ask"`. This is a "dry run" that scans the file for images but stops before full conversion to let the user select which images to analyze.

*   `success`: `true`
*   `status`: `"pending_approval"` (if images found) or `"pending_add"`.
*   `skipped`: `true` (indicates conversion did not happen).
*   `needs_image_selection`: `true` if images were found.
*   `selection_token`: A unique token required to resume processing with selected images.
*   [image_manifest](cci:1://file://wsl.localhost/Ubuntu/home/dba/kmc-rag/docs_converter.py:462:0-507:19): A list of all images found, including:
    *   [id](cci:1://file://wsl.localhost/Ubuntu/home/dba/kmc-rag/docs_converter.py:679:0-713:64): Unique image ID.
    *   `page`: Page number.
    *   `width`/`height`: Dimensions.
    *   [thumb_b64](cci:1://file://wsl.localhost/Ubuntu/home/dba/kmc-rag/docs_converter.py:441:0-460:50): Base64 encoded thumbnail.
*   `image_count_total`: Number of images found.

---

### 3. Error Output
Returned when any part of the process fails. The HTTP status code is usually **200 OK** (to allow workflow tools like n8n to handle the error logic), but the JSON body indicates failure.

*   `success`: `false`
*   [error](cci:1://file://wsl.localhost/Ubuntu/home/dba/kmc-rag/docs_converter.py:123:0-175:33): `true`
*   `error_type`: A specific error code (e.g., `FILE_NOT_FOUND`, `UNSUPPORTED_FORMAT`, `DOCLING_CONVERSION_FAILED`).
*   `error_message`: Human-readable description of the error.
*   `error_details`: Optional dictionary with technical details (e.g., exception type, specific step where it failed).
*   `conversion_status`: `"failed"`
*   [file_id](cci:1://file://wsl.localhost/Ubuntu/home/dba/kmc-rag/docs_converter.py:85:0-89:64): The ID of the file that failed.
*   `filename`: The filename (if known).