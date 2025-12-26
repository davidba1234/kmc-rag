# Chat Conversation

Note: _This is purely the output of the chat conversation and does not contain any raw data, codebase snippets, etc. used to generate the output._

### User Input

explain th parameters that the api convert-file accepts. When it is actually converting and if it is uploaded a file, what additional parameters are needed?

*Listed directory [kmc-rag](file://wsl.localhost/Ubuntu/home/dba/kmc-rag) *

*Grep searched codebase*

*Viewed [docs_converter.py](file://wsl.localhost/Ubuntu/home/dba/kmc-rag/docs_converter.py) *

### Planner Response

Based on the code analysis of [docs_converter.py](cci:7://file://wsl.localhost/Ubuntu/home/dba/kmc-rag/docs_converter.py:0:0-0:0), here are the details for the `/docling/convert-file` API endpoint.

### **Endpoint Parameters**

The API accepts parameters in two ways depending on whether you are uploading a new file or converting an existing one.

#### **1. Common Options (The [data](cci:1://file://wsl.localhost/Ubuntu/home/dba/kmc-rag/docs_converter.py:179:0-208:28) object)**
These parameters control the conversion process.
*   **`do_chunking`** (Boolean, default: `False`): If `True`, performs semantic chunking on the text.
*   **`chunking_options`** (Object, default: `{}`): Settings for chunking, e.g., `{"threshold": 0.75, "chunk_size": 512}`.
*   **`image_mode`** (String, default: `"ask"`): Controls how images are handled.
    *   `"ask"`: **Survey Mode.** Scans the file for images and returns a manifest/count. **Does NOT perform full conversion.**
    *   `"force"`: **Direct Conversion.** Converts the file immediately, processing all images (or ignoring them if not selected).
    *   `"selected"`: **Selective Conversion.** Converts the file and analyzes only specific images using OpenAI Vision.
*   **`selection_token`** (String): Required if `image_mode` is `"selected"`. Obtained from the `"ask"` response.
*   **`selected_image_ids`** (List of Strings): Required if `image_mode` is `"selected"`. The IDs of images to analyze.
*   **`vision_model`** (String, default: `"gpt-4o-mini"`): The OpenAI model to use for image description.

---

### **When is it actually converting?**

The actual Docling conversion (heavy processing) **ONLY** happens when `image_mode` is set to **`"force"`** or **`"selected"`**.

If `image_mode` is `"ask"` (which is the default), the API will:
1.  Scan the file.
2.  Count images.
3.  Return a JSON response with `status: "pending_approval"` (if images found) or `"pending_add"`.
4.  **It stops there** and waits for a second request with `"selected"` or `"force"`.

---

### **If uploading a file, what additional parameters are needed?**

When uploading a file, you must use `multipart/form-data`.

1.  **[file](cci:1://file://wsl.localhost/Ubuntu/home/dba/kmc-rag/docs_converter.py:603:0-676:46)**: The actual file object.
2.  **[data](cci:1://file://wsl.localhost/Ubuntu/home/dba/kmc-rag/docs_converter.py:179:0-208:28)**: A **JSON string** containing the options listed above.

**Critical Note for Uploads:**
If you want to upload and convert in a single step, you **MUST** provide the [data](cci:1://file://wsl.localhost/Ubuntu/home/dba/kmc-rag/docs_converter.py:179:0-208:28) parameter with `image_mode` set to `"force"`.

**Example `curl` for Upload + Convert:**
```bash
curl -X POST http://localhost:5000/docling/convert-file \
  -F "file=@/path/to/document.pdf" \
  -F 'data={"image_mode": "force", "do_chunking": true}'
```

If you omit the [data](cci:1://file://wsl.localhost/Ubuntu/home/dba/kmc-rag/docs_converter.py:179:0-208:28) parameter, it defaults to `image_mode="ask"`, so it will just upload the file and return the survey result (image counts) without converting the text.