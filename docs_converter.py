from dotenv import load_dotenv
load_dotenv()
from flask import Flask, jsonify, request
import os
import json
import glob
import zipfile
import xml.etree.ElementTree as ET
from docx import Document
from pathlib import Path
from datetime import datetime, timezone
import logging
import time
import fitz  # PyMuPDF
import pikepdf # For writing PDF metadata
import uuid    # For generating GUIDs
import base64  # To encode images for API call
from openai import OpenAI # To call the vision model
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, TableStructureOptions, TableFormerMode
)
from docling.datamodel.base_models import InputFormat
import pdfplumber
import tempfile
import zipfile
from chonkie import SemanticChunker # type: ignore
import hashlib
from io import BytesIO
try:
    from PIL import Image  # for thumbnails
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


# --- Basic Configuration ---
BASE_DIR = "/mnt/c/Users/Anderson/Documents/n8n/kmc-rag/test_docs"
print(f"BASE_DIR is set to: {BASE_DIR}")

# Force root logger configuration
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Clear existing handlers to avoid duplicates
if root_logger.hasHandlers():
    root_logger.handlers.clear()

# Create handlers
file_handler = logging.FileHandler(os.path.join(os.path.dirname(BASE_DIR), "flask_app.log"))
stream_handler = logging.StreamHandler()

# Set formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add handlers
root_logger.addHandler(file_handler)
root_logger.addHandler(stream_handler)
logger = logging.getLogger(__name__)

app = Flask(__name__)

JSON_CACHE_DIR = os.path.join(BASE_DIR, "json_cache")
JSON_CACHE_DIR = os.path.join(BASE_DIR, "json_cache")
IMAGE_OUTPUT_DIR = os.path.join(BASE_DIR, "image_output")
os.makedirs(JSON_CACHE_DIR, exist_ok=True)

SUPPORTED_EXTENSIONS = ['.docx', '.pdf']

# --- FileID Registry (Global Cache) ---
# Maps fileID -> file_path for quick lookups
FILE_ID_REGISTRY = {}
FILE_PATH_CACHE = {}  # Maps file_path -> fileID for reverse lookups

# --- ALL METADATA FIELDS ALWAYS INCLUDED ---
ALL_METADATA_FIELDS = [
    "file_id",
    "filename",
    "file_path",
    "subfolder",
    "file_extension",
    "file_size",
    "file_size_mb",
    "word_count",
    "character_count",
    "paragraph_count",
    "table_count",
    "lastModified",
    "createdDate",
    "has_guid",
    "guid_source",
    "DateReviewed",
    "DateNext",
    "ai_title",
    "ai_description"
]

def register_file_id(file_id: str, file_path: str) -> None:
    """Register a fileID to file_path mapping."""
    FILE_ID_REGISTRY[file_id] = file_path
    FILE_PATH_CACHE[file_path] = file_id
    logger.debug(f"Registered fileID: {file_id} -> {file_path}")

def resolve_file_path(file_id: str) -> str | None:
    """Resolve a fileID to its file_path. Returns None if not found."""
    return FILE_ID_REGISTRY.get(file_id)

def get_file_id_or_register(file_path: str) -> str:
    """Get fileID for a path, or register it if not already present."""
    if file_path in FILE_PATH_CACHE:
        return FILE_PATH_CACHE[file_path]
    
    ext = os.path.splitext(file_path)[1].lower()
    guid = None
    
    try:
        if ext == '.pdf':
            guid = read_pdf_guid(file_path)
        elif ext == '.docx':
            guid = extract_docx_guid(file_path)
    except Exception as e:
        logger.warning(f"Failed to extract GUID from {file_path}: {e}")
    
    if not guid:
        guid = os.path.splitext(os.path.basename(file_path))[0].replace(' ', '_').replace('-', '_').lower()
    
    register_file_id(guid, file_path)
    return guid

def get_image_descriptions_from_api(image_info_list):
    # No-op stub: image analysis handled by /docling/convert-file two-step flow.
    return []

# --- Error Handling Helper ---

def create_error_response(file_id_or_path, error_message, error_type="CONVERSION_ERROR", details=None):
    """
    Creates a standardized error response that returns 200 OK with error flag.
    Accepts either fileID or file_path as first parameter.
    
    Args:
        file_id_or_path: FileID or file path that failed
        error_message: Human-readable error message
        error_type: Type of error (CONVERSION_ERROR, PARSING_ERROR, FILE_NOT_FOUND, etc.)
        details: Optional dict with additional error details
    
    Returns:
        tuple: (jsonify response dict, 200 status code)
    """
    # Determine if input is fileID or path
    if file_id_or_path and file_id_or_path.startswith('/'):
        file_path = file_id_or_path
        file_id = FILE_PATH_CACHE.get(file_path)
        if not file_id:
            file_id = get_file_id_or_register(file_path) if os.path.exists(file_path) else None
    else:
        file_id = file_id_or_path
        file_path = resolve_file_path(file_id)
    
    # Attempt to include file timestamps (UTC) when available
    last_modified = None
    created_date = None
    try:
        if file_path and os.path.exists(file_path):
            st = os.stat(file_path)
            last_modified = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
            created_date = datetime.fromtimestamp(st.st_ctime, tz=timezone.utc).isoformat()
    except Exception as e:
        logger.warning(f"Failed to stat file for error response timestamps: {e}")

    response = {
        "success": False,
        "error": True,
        "error_type": error_type,
        "error_message": error_message,
        "file_id": file_id,
        "file_path": file_path,
        "filename": os.path.basename(file_path) if file_path else "unknown",
        "conversion_status": "failed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lastModified": last_modified,
        "createdDate": created_date
    }
    if details:
        response["error_details"] = details
    
    logger.error(f"[{error_type}] {file_id}: {error_message}")
    return jsonify(response), 200

# --- Helper Functions ---

def make_complete_metadata(file_path, extra_data=None):
    """Create complete metadata dictionary with all fields, using None for missing values."""
    file_stats = os.stat(file_path)
    parent_dir = os.path.basename(os.path.dirname(file_path))
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    # Base metadata that's always available
    metadata = {
        "filename": filename,
        "file_path": file_path,
        "subfolder": parent_dir,
        "file_extension": ext,
        "file_size": file_stats.st_size,
        "file_size_mb": round(file_stats.st_size / (1024 * 1024), 2),
        "lastModified": datetime.fromtimestamp(file_stats.st_mtime, tz=timezone.utc).isoformat(),
        "createdDate": datetime.fromtimestamp(file_stats.st_ctime, tz=timezone.utc).isoformat(),
    }

    # Initialize all fields from ALL_METADATA_FIELDS with None
    complete_metadata = {field: None for field in ALL_METADATA_FIELDS}
    
    # Update with base metadata
    complete_metadata.update(metadata)
    
    # Update with any extra data provided
    if extra_data and isinstance(extra_data, dict):
        complete_metadata.update(extra_data)

    return complete_metadata

def extract_docx_guid(docx_path):
    try:
        with zipfile.ZipFile(docx_path, 'r') as docx_zip:
            settings_path = 'word/settings.xml'
            if settings_path not in docx_zip.namelist():
                return None
            with docx_zip.open(settings_path) as settings_file:
                root = ET.fromstring(settings_file.read())
            namespaces = {
                'w15': 'http://schemas.microsoft.com/office/word/2012/wordml'
            }
            doc_id_element = root.find('.//w15:docId', namespaces)
            if doc_id_element is not None:
                doc_id = doc_id_element.get('{http://schemas.microsoft.com/office/word/2012/wordml}val')
                if doc_id:
                    return doc_id.strip('{}')
        return None
    except Exception as e:
        logger.warning(f"Could not extract GUID from {docx_path}: {e}")
        return None

def get_or_create_pdf_guid(pdf_path: str) -> str:
    """Reads a custom GUID from a PDF's metadata. If not present, generates one and saves it."""
    guid_key = '/AppGUID'
    try:
        with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
            docinfo = pdf.docinfo
            if guid_key in docinfo:
                logger.info(f"Found existing GUID in {pdf_path}")
                return str(docinfo[guid_key])

            logger.info(f"No GUID found in {pdf_path}. Generating a new one.")
            new_guid = str(uuid.uuid4())
            docinfo[guid_key] = new_guid
            pdf.save()
            logger.info(f"Saved new GUID to {pdf_path}")
            return new_guid
            
    except Exception as e:
        logger.error(f"Pikepdf failed to read or write GUID for {pdf_path}: {e}", exc_info=True)
        filename_without_ext = os.path.splitext(os.path.basename(pdf_path))[0]
        return filename_without_ext.replace(' ', '_').replace('-', '_').lower()

def read_pdf_guid(pdf_path: str) -> str | None:
    """Reads a custom GUID from a PDF's metadata without modifying the file."""
    guid_key = '/AppGUID'
    try:
        with pikepdf.open(pdf_path) as pdf:
            if guid_key in pdf.docinfo:
                return str(pdf.docinfo[guid_key])
        return None
    except Exception as e:
        logger.warning(f"Could not read GUID from {pdf_path} with pikepdf: {e}")
        return None

def extract_custom_metadata(doc_path: str):
    date_reviewed, date_next = None, None
    try:
        with zipfile.ZipFile(doc_path, 'r') as z:
            if 'docProps/custom.xml' not in z.namelist():
                return date_reviewed, date_next
            xml_content = z.read('docProps/custom.xml')
            root = ET.fromstring(xml_content)
            
            CP_NAMESPACE = 'http://schemas.openxmlformats.org/officeDocument/2006/custom-properties'
            VT_NAMESPACE = 'http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes'
            FQN_PROPERTY = f'{{{CP_NAMESPACE}}}property'
            FQN_FILETIME = f'{{{VT_NAMESPACE}}}filetime'

            for prop in root.findall(FQN_PROPERTY):
                name = prop.attrib.get('name')
                if name == 'DateReviewed':
                    value_elem = prop.find(FQN_FILETIME)
                    if value_elem is not None and value_elem.text:
                        date_reviewed = datetime.fromisoformat(value_elem.text.replace('Z', '+00:00')).isoformat()
                elif name == 'DateNext':
                    value_elem = prop.find(FQN_FILETIME)
                    if value_elem is not None and value_elem.text:
                        date_next = datetime.fromisoformat(value_elem.text.replace('Z', '+00:00')).isoformat()
    except Exception as e:
        logger.warning(f"Failed to extract custom metadata from {doc_path}: {e}")
    return date_reviewed, date_next

def generate_ai_title_description(full_text: str, fallback_filename: str) -> tuple[str, str]:
    def sample(text: str, max_chars: int = 12000) -> str:
        if len(text) <= max_chars:
            return text
        head = text[: max_chars // 2]
        tail = text[-max_chars // 2 :]
        return head + "\n...\n" + tail

    try:
        client = OpenAI()
    except Exception as e:
        logger.warning(f"OpenAI client unavailable, will fallback: {e}")
        client = None

    default_title = Path(fallback_filename).stem.replace("_", " ").replace("-", " ").strip()
    default_desc = "No AI description available."

    if not client:
        return default_title, default_desc

    prompt = (
        "You are given extracted document text. "
        "Return a concise JSON with fields 'title' and 'description'. "
        "Title <= 12 words, description 1–2 sentences. "
        "Do not include code fences.\n\n"
        f"TEXT START\n{sample(full_text)}\nTEXT END"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200,
        )
        content = resp.choices[0].message.content.strip()
        data = json.loads(content) if content.startswith("{") else {}
        title = (data.get("title") or default_title).strip()
        desc = (data.get("description") or default_desc).strip()
        return title, desc
    except Exception as e:
        logger.warning(f"AI title/description generation failed, using fallback: {e}")
        return default_title, default_desc

# def get_image_descriptions_from_api(image_info_list: list) -> list:
#     """Takes a list of image paths and calls a vision model for each."""
#     try:
#         client = OpenAI()
#     except Exception as e:
#         logger.error(f"OpenAI client failed to initialize. Is OPENAI_API_KEY set? Error: {e}")
#         return []

#     descriptions = []
#     for image_info in image_info_list:
#         path = image_info.get("path")
#         try:
#             with open(path, "rb") as image_file:
#                 base64_image = base64.b64encode(image_file.read()).decode('utf-8')

#             response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[
#                     {"role": "user", "content": [
#                         {"type": "text", "text": "Describe this image from a document in detail. If it's a chart or graph, explain what it shows."},
#                         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
#                     ]}
#                 ],
#                 max_tokens=300,
#             )
#             descriptions.append({"page": image_info.get("page"), "description": response.choices[0].message.content})
#         except Exception as e:
#             logger.error(f"Failed to get description for image {path}: {e}")
#     return descriptions

# --- Core Conversion Functions ---

def find_all_supported_files():
    supported_files = []
    
    # Use os.walk for more reliable file discovery
    for root, dirs, files in os.walk(BASE_DIR):
        for filename in files:
            # Skip temporary files
            if filename.startswith('~$'):
                continue
                
            # Check if file has supported extension (case-insensitive)
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in [ext.lower() for ext in SUPPORTED_EXTENSIONS]:
                file_path = os.path.join(root, filename)
                try:
                    file_stats = os.stat(file_path)
                    parent_dir = os.path.basename(os.path.dirname(file_path))
                    supported_files.append({
                        "filename": filename,
                        "full_path": file_path,
                        "subfolder": parent_dir,
                        "lastModified": datetime.fromtimestamp(file_stats.st_mtime, tz=timezone.utc).isoformat()
                    })
                except OSError as e:
                    logger.warning(f"Could not stat file {file_path}: {e}")
                    continue
    
    return supported_files

def docx_to_json(docx_path: str) -> dict:
    """Placeholder for removed native DOCX conversion function.
    The project now uses Docling for all conversions. This stub exists
    to keep legacy call sites from failing static analysis.
    """
    raise NotImplementedError("Native DOCX conversion removed; use Docling converter instead.")
def get_cache_path(file_path):
    filename_without_ext = os.path.splitext(os.path.basename(file_path))[0]
    parent_dir = os.path.basename(os.path.dirname(file_path))
    cache_name = f"{parent_dir}_{filename_without_ext}.json".replace(' ', '_').replace('-', '_')
    return os.path.join(JSON_CACHE_DIR, cache_name)

def is_cache_valid(file_path, cache_path):
    if not os.path.exists(cache_path):
        return False
    return os.path.getmtime(cache_path) > os.path.getmtime(file_path)

def perform_chunking(input_text, complete_metadata, options):
    """Performs semantic chunking and enriches with COMPLETE metadata for RAG."""
    try:
        chunker = SemanticChunker(
            threshold=options.get("threshold", 0.75),
            chunk_size=options.get("chunk_size", 512),
            device_map="cpu"
        )
        raw_chunks = chunker.chunk(input_text)

        enriched_chunks = []
        for chunk in raw_chunks:
            # Each chunk gets ALL metadata fields for comprehensive RAG retrieval
            enriched = {
                "text": chunk.text,
                "start_index": chunk.start_index,
                "end_index": chunk.end_index,
                "token_count": chunk.token_count,
                "metadata": complete_metadata.copy()  # Include ALL metadata fields
            }
            enriched_chunks.append(enriched)
        return enriched_chunks
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        return []

def _make_thumb_b64(image_bytes: bytes, thumb_px: int = 256) -> str:
    """
    Returns a data-URI base64 thumbnail. Uses Pillow if available; otherwise returns
    a JPEG-compressed preview from the original bytes (best-effort).
    """
    try:
        if PIL_AVAILABLE:
            with Image.open(BytesIO(image_bytes)) as im:
                im = im.convert("RGB")
                im.thumbnail((thumb_px, thumb_px), Image.LANCZOS)
                out = BytesIO()
                im.save(out, format="JPEG", quality=80, optimize=True)
                b64 = base64.b64encode(out.getvalue()).decode("utf-8")
                return f"data:image/jpeg;base64,{b64}"
        # Fallback: just base64 the original (can be large)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/{'jpeg'};base64,{b64}"
    except Exception:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/{'jpeg'};base64,{b64}"

def build_pdf_image_manifest(pdf_path: str, guid: str,
                             min_area: int = 12000,
                             max_images: int = 150,
                             thumb_px: int = 256) -> list[dict]:
    """
    Scans PDF images and returns a manifest with small thumbnails and metadata.
    Does not persist full images to disk; we re-extract for selected items later.
    """
    manifest = []
    try:
        doc = fitz.open(pdf_path)
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            for img in page.get_images(full=True):
                xref = img[0]
                width, height = img[2], img[3]
                area = (width or 0) * (height or 0)
                if area < min_area:
                    continue
                try:
                    base_image = doc.extract_image(xref)
                except Exception:
                    continue
                image_bytes = base_image.get("image", b"")
                ext = base_image.get("ext", "png")
                size_bytes = len(image_bytes)
                thumb_b64 = _make_thumb_b64(image_bytes, thumb_px=thumb_px)
                entry_id = f"{guid}:{page_idx+1}:{xref}"
                manifest.append({
                    "id": entry_id,
                    "page": page_idx + 1,
                    "xref": xref,
                    "width": width,
                    "height": height,
                    "size_bytes": size_bytes,
                    "ext": ext,
                    "thumb_b64": thumb_b64
                })
                if len(manifest) >= max_images:
                    break
            if len(manifest) >= max_images:
                break
        doc.close()
    except Exception as e:
        logger.error(f"build_pdf_image_manifest error for {pdf_path}: {e}", exc_info=True)
    return manifest

def _selection_state_path(token: str) -> str:
    return os.path.join(JSON_CACHE_DIR, f"image_selection_{token}.json")

def save_selection_state(token: str, state: dict) -> None:
    try:
        with open(_selection_state_path(token), "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save selection state: {e}")

def load_selection_state(token: str) -> dict | None:
    try:
        with open(_selection_state_path(token), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def describe_selected_images_with_openai(pdf_path: str,
                                         selection_state: dict,
                                         selected_ids: list[str],
                                         model: str = "gpt-4o-mini",
                                         per_image_max_tokens: int = 250,
                                         sleep_sec: float = 0.0) -> list[dict]:
    """
    Calls OpenAI Vision only for selected ids.
    Returns list of {id, page, description}.
    """
    try:
        client = OpenAI()
    except Exception as e:
        logger.error(f"OpenAI client failed. Is OPENAI_API_KEY set? {e}")
        return []

    id_to_info = {e["id"]: e for e in selection_state.get("manifest", [])}
    results = []
    try:
        doc = fitz.open(pdf_path)
        for sel_id in selected_ids:
            info = id_to_info.get(sel_id)
            if not info:
                continue
            page = info.get("page")
            xref = info.get("xref")
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get("image", b"")
                b64 = base64.b64encode(image_bytes).decode("utf-8")
                prompt = (
                    "You are analyzing an image extracted from a PDF document. "
                    "Provide a precise, 1–3 sentence description tailored for retrieval. "
                    "If it is a chart/graph, state title (if visible), axes/units, key trend(s), and any standout values. "
                    "If it is a table/diagram/photo/illustration, summarize what it conveys. "
                    "Do not speculate beyond visible content."
                )
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                        ]
                    }],
                    temperature=0.2,
                    max_tokens=per_image_max_tokens
                )
                desc = (resp.choices[0].message.content or "").strip()
                results.append({"id": sel_id, "page": page, "description": desc})
                if sleep_sec:
                    time.sleep(sleep_sec)
            except Exception as e:
                logger.warning(f"Vision call failed for image id={sel_id}: {e}")
        doc.close()
    except Exception as e:
        logger.error(f"Failed to open PDF for vision extraction: {e}", exc_info=True)
    return results

def append_image_descriptions_to_markdown(markdown_text: str, image_descs: list[dict]) -> str:
    if not image_descs:
        return markdown_text
    lines = []
    lines.append(markdown_text)
    lines.append("\n\n---\n### Image descriptions\n")
    # Sort by page then id for stable output
    for item in sorted(image_descs, key=lambda x: (x.get("page", 0), x.get("id", ""))):
        page = item.get("page")
        desc = item.get("description", "").strip()
        if desc:
            lines.append(f"- Page {page}: {desc}")
    return "\n".join(lines)

# --- API Endpoints ---
# --- API Endpoints ---

@app.route('/list-files', methods=['GET'])
def list_files():
    try:
        raw_file_list = find_all_supported_files()
        enriched_files = []

        for file_info in raw_file_list:
            path = file_info.get('full_path')
            if not path:
                logger.warning("Skipping file with no full_path in file_info")
                continue

            try:
                # Determine GUID/file_id and basic provenance without mutating files
                ext = os.path.splitext(path)[1].lower()
                guid = None
                guid_source = None
                date_reviewed = None
                date_next = None

                if ext == '.docx':
                    try:
                        guid = extract_docx_guid(path)
                        guid_source = 'docx_settings' if guid else 'filename_fallback'
                    except Exception as e:
                        logger.warning(f"Failed to read DOCX GUID for {path}: {e}")
                    try:
                        date_reviewed, date_next = extract_custom_metadata(path)
                    except Exception:
                        date_reviewed, date_next = None, None

                elif ext == '.pdf':
                    try:
                        guid = read_pdf_guid(path)
                        guid_source = 'pdf_metadata' if guid else 'filename_fallback'
                    except Exception as e:
                        logger.warning(f"Failed to read PDF GUID for {path}: {e}")

                # Fallback id when no GUID present
                if not guid:
                    filename_without_ext = os.path.splitext(os.path.basename(path))[0]
                    guid = f"fallback_{filename_without_ext.replace(' ', '_').lower()}"
                    if not guid_source:
                        guid_source = 'filename_fallback'

                extra = {
                    'file_id': guid,
                    'has_guid': bool(guid) and not str(guid).startswith('fallback_'),
                    'guid_source': guid_source,
                    'DateReviewed': date_reviewed,
                    'DateNext': date_next
                }

                # Build the full metadata dict using existing helper
                complete_meta = make_complete_metadata(path, extra)

                enriched_files.append(complete_meta)

            except Exception as inner_e:
                logger.error(f"Error processing file {path}: {inner_e}", exc_info=True)
                # Fallback minimal entry so list remains stable
                enriched_files.append({
                    'filename': os.path.basename(path) if path else None,
                    'file_path': path,
                    'file_id': None,
                    'lastModified': None,
                    'createdDate': None
                })

        return jsonify({"files": enriched_files, "count": len(enriched_files)})

    except Exception as e:
        logger.error(f"Error in /list-files: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/ensure-guids', methods=['POST'])
def ensure_guids():
    """Scans all supported files and ensures that any file missing a GUID gets one."""
    logger.info("Received request for /ensure-guids. Scanning all files.")
    try:
        all_files = find_all_supported_files()
        files_updated = 0
        
        for file_info in all_files:
            path = file_info.get('full_path')
            if not path:
                continue
            if path.lower().endswith('.docx'):
                # DOCX GUIDs are read-only in settings.xml; skip
                try:
                    if extract_docx_guid(path):
                        continue
                except Exception as e:
                    logger.warning(f"Failed reading DOCX GUID for {path}: {e}")
            elif path.lower().endswith('.pdf'):
                try:
                    existing_guid = read_pdf_guid(path)
                    if not existing_guid:
                        get_or_create_pdf_guid(path)
                        files_updated += 1
                except Exception as e:
                    logger.warning(f"Failed to ensure PDF GUID for {path}: {e}")

        message = f"Scan complete. Updated {files_updated} PDF file(s) with a new GUID."
        logger.info(message)
        return jsonify({"success": True, "message": message, "files_updated": files_updated})

    except Exception as e:
        logger.error(f"An error occurred in /ensure-guids: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/docling/convert-file', methods=['POST'])
def docling_convert_file():
    """
    Convert a single file (DOCX or PDF) using its fileID as the unique identifier,
    or accept a direct file upload.
    Returns 200 OK with error flag on failure, allowing n8n to skip and continue.
    """
    file_id = None
    file_path = None
    temp_file_path = None
    is_upload = False

    try:
        # ===== STEP 1: Determine input method =====
        # Check if a file was uploaded
        uploaded_file = request.files.get('file')
        # FIX: Check if we have a file, even if the filename is generic/missing
        if uploaded_file:
            is_upload = True
            
            # 1. Try to get real filename from the 'data' JSON field first
            json_data = {}
            raw_data = request.form.get('data')
            if raw_data:
                try:
                    json_data = json.loads(raw_data)
                except Exception as e:
                    logger.warning(f"Failed to parse 'data' form field: {e}")

            # 2. Use the explicitly provided filename, or fallback to the upload object
            filename = json_data.get('filename') or uploaded_file.filename or "unknown_file"
            
            logger.info(f"Processing uploaded file. Detected filename: {filename}")

            # 3. Validate file extension
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                return create_error_response(filename, f"Unsupported file type: {ext} (derived from {filename})", "UNSUPPORTED_FORMAT")

            # Save uploaded file to temporary location
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                    uploaded_file.save(temp_file.name)
                    temp_file_path = temp_file.name
                    file_path = temp_file_path

                # Generate file_id for uploaded file
                file_id = str(uuid.uuid4())
                logger.info(f"Generated file_id for upload: {file_id}")

            except Exception as e:
                return create_error_response(filename, f"Failed to save uploaded file: {str(e)}", "FILE_SAVE_ERROR")

        else:
            # Original file_id based processing
            try:
                data = request.get_json(force=True)
                if not data or 'file_id' not in data:
                    return jsonify({"error": "Missing 'file_id' in JSON request body or 'file' in form data"}), 400
                file_id = data['file_id']
            except Exception as e:
                return jsonify({"error": f"Invalid request: {str(e)}"}), 400

            # Initialize filename for non-upload case
            filename = data.get('filename') or "unknown_file"

            # ===== STEP 2: Resolve fileID to file_path (with auto-register fallback) =====
            try:
                file_path = resolve_file_path(file_id)

                # If not in registry, try to find file by ID and auto-register
                if not file_path:
                    # Look for a file that matches this ID (could be from external source)
                    # This supports calling convert-file with an ID even if list-files wasn't called first
                    all_files = find_all_supported_files()
                    for file_info in all_files:
                        candidate_path = file_info['full_path']
                        # Get the ID that would be assigned to this file
                        candidate_id = get_file_id_or_register(candidate_path)
                        if candidate_id == file_id:
                            file_path = candidate_path
                            break

                    if not file_path:
                        return create_error_response(file_id, f"FileID not found and could not locate matching file: {file_id}", "FILE_ID_NOT_FOUND")
            except Exception as e:
                return create_error_response(file_id, f"Failed to resolve fileID: {str(e)}", "FILE_ID_RESOLUTION_ERROR")
            
            if filename == "unknown_file" and file_path:
                filename = os.path.basename(file_path)
        
        # ===== STEP 3: Validate file path =====
        try:
            if not os.path.exists(file_path):
                return create_error_response(file_id, f"File not found: {file_path}", "FILE_NOT_FOUND")
            # Only check BASE_DIR security for non-uploaded files
            if not is_upload and not os.path.abspath(file_path).startswith(os.path.abspath(BASE_DIR)):
                return create_error_response(file_id, "File path is outside allowed directory", "SECURITY_ERROR")
        except Exception as e:
            return create_error_response(file_id, f"Failed to validate file path: {str(e)}", "PATH_VALIDATION_ERROR")
        
        # ===== STEP 4: Extract extension =====
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                return create_error_response(file_id, f"Unsupported file type: {ext}", "UNSUPPORTED_FORMAT")
        except Exception as e:
            return create_error_response(file_id, f"Failed to determine file type: {str(e)}", "FILE_TYPE_ERROR")

        # ===== STEP 5: File timestamps (UTC) =====
        try:
            stat = os.stat(file_path)
            last_modified_iso = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            created_date_iso = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat()
        except Exception as e:
            logger.warning(f"Failed to stat file for timestamps: {e}")
            # For uploaded files, use current time
            if is_upload:
                now = datetime.now(timezone.utc)
                last_modified_iso = now.isoformat()
                created_date_iso = now.isoformat()
            else:
                last_modified_iso = None
                created_date_iso = None

        # [Existing Params] - get from JSON data if present
        data = {}
        if not is_upload:
            try:
                data = request.get_json(force=True) or {}
            except:
                data = {}
        else:
            # For uploads, check if additional JSON data was sent
            json_data = request.form.get('data')
            if json_data:
                try:
                    data = json.loads(json_data)
                except:
                    data = {}
            
            logger.debug(f"Request form keys: {list(request.form.keys())}")
            logger.debug(f"Raw data field: {request.form.get('data')}")
            logger.debug(f"Parsed data: {data}")

            # Use file_id from JSON data if provided (for uploads in selected mode)
            if data.get('file_id'):
                file_id = data['file_id']
                logger.info(f"Using file_id from JSON data: {file_id}")

        force_refresh = data.get("force_refresh", False)
        do_chunking = data.get("do_chunking", False)
        chunking_options = data.get("chunking_options", {})
        image_mode = data.get("image_mode", "ask")
        selection_token = data.get("selection_token")
        selected_image_ids = data.get("selected_image_ids", []) or []

        user_provided_path = data.get('file_path')
        user_provided_subfolder = data.get('subfolder')
        
        logger.debug(f"Extracted user_provided_path: {user_provided_path}")
        logger.debug(f"Extracted user_provided_subfolder: {user_provided_subfolder}")
        logger.debug(f"Data keys available: {list(data.keys())}")

        # ===== STEP 6: Count images (PDF & DOCX) =====
        # Fast pre-check to see what we are dealing with
        raw_total_images = 0
        try:
            if ext == '.pdf':
                with fitz.open(file_path) as count_doc:
                    for page in count_doc:
                        raw_total_images += len(page.get_images(full=True))
            
            elif ext == '.docx':
                try:
                    with zipfile.ZipFile(file_path, 'r') as z:
                        media_files = [n for n in z.namelist() if n.startswith('word/media/')]
                        raw_total_images = len(media_files)
                except Exception as e:
                    logger.warning(f"Could not count images in DOCX {file_path}: {e}")
                    
        except Exception as e:
            logger.warning(f"Image counting failed for {file_path}: {e}")

        # ===== STEP 7: SURVEY MODE (image_mode="ask") =====
        # If the workflow asks us to survey, we return metadata immediately.
        # We do NOT run Docling here. We wait for user confirmation.
        if image_mode == "ask":
            logger.info(f"Surveying file (Ask Mode): {file_id} - Images found: {raw_total_images}")
            
            
            # 1. Prepare Image Manifest (if images exist)
            image_manifest = []
            selection_token = None
            
            if ext == ".pdf" and raw_total_images > 0:
                try:
                    image_manifest = build_pdf_image_manifest(file_path, file_id)
                    if image_manifest:
                        token_seed = f"{file_id}:{file_path}:{time.time()}:{len(image_manifest)}"
                        selection_token = hashlib.sha256(token_seed.encode("utf-8")).hexdigest()[:32]
                        
                        # Save state so we can retrieve these specific images later
                        # FIX: Use user_provided_path if available, as it is stable across uploads
                        stable_path = user_provided_path if user_provided_path else file_path
                        save_selection_state(selection_token, {
                            "file_path": stable_path,
                            "file_id": file_id,
                            "manifest": image_manifest,
                            "created_at": datetime.now(timezone.utc).isoformat()
                        })
                except Exception as e:
                    logger.warning(f"Failed to build image manifest: {e}")

            # 2. Return 'Pending' JSON
            # This stops the script here. The UI will pick this up.
            # Status distinguishes between files with images (pending_approval) and without (pending_add)
            status = "pending_approval" if len(image_manifest) > 0 else "pending_add"
            
            # Determine return path: user provided > real path (if not upload) > None
            return_path = user_provided_path if user_provided_path else (file_path if not is_upload else None)

            return jsonify({
                "success": True,
                "status": status,
                "skipped": True,
                "file_path": return_path,
                "subfolder": user_provided_subfolder, # Include subfolder if available
                "filename": filename,
                "lastModified": last_modified_iso,
                "createdDate": created_date_iso,
                "file_id": file_id,
                "image_count_total": raw_total_images,
                "is_upload": is_upload,

                # Image Selection Data
                "needs_image_selection": (len(image_manifest) > 0),
                "image_manifest": image_manifest,
                "selection_token": selection_token
            })

        # The code below only runs if image_mode is "selected" or "force"
        # (i.e. AFTER the user has clicked 'Proceed' in the UI).
        # ==================================================================
        # [SLOW PATH] STANDARD CONVERSION
        # ==================================================================
        
        # FIX: Define use_docling here. 
        # Since you want Docling for all files, we set this to True.
        # This ensures the script skips "[Path A: Native DOCX]" below and goes straight to Docling.
        use_docling = True

        # [Path A: Native DOCX]
        if ext == '.docx' and not use_docling:
            try:
                try:
                    content = docx_to_json(file_path)
                except Exception as e:
                    return create_error_response(file_id, f"Failed to convert DOCX: {str(e)}", "DOCX_CONVERSION_ERROR", {"step": "docx_to_json"})
                
                if "error" in content:
                    return create_error_response(file_id, content.get("error", "Unknown error in DOCX conversion"), "DOCX_PARSING_ERROR", {"step": "docx_parsing"})
                
                try:
                    date_reviewed, date_next = extract_custom_metadata(file_path)
                except Exception as e:
                    logger.warning(f"Failed to extract custom metadata from DOCX: {e}")
                    date_reviewed, date_next = None, None
                
                try:
                    full_text = "\n\n".join(content.get("paragraphs", []))
                except Exception as e:
                    return create_error_response(file_id, f"Failed to extract text from DOCX: {str(e)}", "TEXT_EXTRACTION_ERROR")
                
                try:
                    ai_title, ai_description = generate_ai_title_description(full_text, os.path.basename(file_path))
                except Exception as e:
                    logger.warning(f"Failed to generate AI title/description: {e}")
                    ai_title, ai_description = None, None
                
                extra_data = {
                    "file_id": file_id,
                    "paragraph_count": len(content.get("paragraphs", [])),
                    "table_count": len(content.get("tables", [])),
                    "word_count": len(full_text.split()),
                    "character_count": len(full_text),
                    "has_guid": True,
                    "guid_source": "file_id",
                    "DateReviewed": date_reviewed,
                    "DateNext": date_next,
                    "ai_title": ai_title,
                    "ai_description": ai_description,
                    "image_count_total": 0,
                    "image_count_selected": 0,
                    "image_count_analyzed": 0,
                }
                
                # Override path/subfolder if provided
                if user_provided_path:
                    extra_data["file_path"] = user_provided_path
                if user_provided_subfolder:
                    extra_data["subfolder"] = user_provided_subfolder

                complete_metadata = make_complete_metadata(file_path, extra_data)
                chunks = []
                if do_chunking:
                    try:
                        chunks = perform_chunking(full_text, complete_metadata, chunking_options)
                    except Exception as e:
                        logger.warning(f"Failed to chunk DOCX: {e}")
                        chunks = []
                # Build a summary for this conversion run
                summary = {
                    "total_files_processed": 1,
                    "docx_count": 1 if ext == '.docx' else 0,
                    "pdf_count": 1 if ext == '.pdf' else 0,
                    # raw_total_images counted earlier; for native DOCX we expect 0
                    "images_total": raw_total_images,
                    "images_selected": 0,
                    "images_analyzed": 0,
                    "images_ignored": max(0, raw_total_images - 0)
                }

                return jsonify({
                    "success": True,
                    "used_converter": "native_docx",
                    "file_path": user_provided_path if user_provided_path else file_path,
                    "filename": content.get("filename"),
                    "full_text": full_text,
                    "structured_content": {
                        "paragraphs": content.get("paragraphs"),
                        "tables": content.get("tables"),
                        "metadata": content.get("metadata")
                    },
                    "metadata": complete_metadata,
                    "chunks": chunks,
                    "lastModified": last_modified_iso,
                    "createdDate": created_date_iso,
                    "summary": summary
                })
            except Exception as e:
                return create_error_response(file_id, f"DOCX conversion failed: {str(e)}", "DOCX_CONVERSION_FAILED", {"exception_type": type(e).__name__})

        # [Path B: Heavy Docling Conversion]
        try:
            logger.info("Running Docling converter (Heavy Process)...")
            try:
                conversion_result = docling_converter.convert(file_path)
            except Exception as e:
                return create_error_response(file_id, f"Docling conversion failed: {str(e)}", "DOCLING_CONVERSION_ERROR", {"exception_type": type(e).__name__})
            
            try:
                document = conversion_result.document
                document_dict = document.export_to_dict()
                full_text_md = document.export_to_markdown()
            except Exception as e:
                return create_error_response(file_id, f"Failed to export document: {str(e)}", "DOCUMENT_EXPORT_ERROR", {"exception_type": type(e).__name__})

            extra_meta = {}
            if ext == ".docx":
                try:
                    date_reviewed, date_next = extract_custom_metadata(file_path)
                    extra_meta.update({"DateReviewed": date_reviewed, "DateNext": date_next})
                except Exception as e:
                    logger.warning(f"Failed to extract custom metadata: {e}")

            
            # Handle Image Mode = SELECTED
            image_descs = []
            selection_state = None
            image_processing_error = None
            image_processing_error_type = None

            if ext == ".pdf" and image_mode == "selected":
                try:
                    # NEW: Check if manifest was passed directly (Stateless/Distributed Fix)
                    passed_manifest = data.get("image_manifest")

                    if passed_manifest:
                        logger.info(f"Stateless Mode: Using image_manifest provided in request data. ({len(passed_manifest)} items)")
                        # Construct a state object on the fly
                        selection_state = {
                            "file_path": user_provided_path if user_provided_path else file_path,
                            "manifest": passed_manifest
                        }

                    # FALLBACK: If no manifest passed, try loading from local token file (Legacy Mode)
                    elif selection_token:
                        selection_state = load_selection_state(selection_token)
                        # Only validate path if we are loading from a cached token
                        current_stable_path = user_provided_path if user_provided_path else file_path
                        expected_path = selection_state.get("file_path") if selection_state else "Unknown"

                        if not selection_state or expected_path != current_stable_path:
                             # Track error but allow proceeding if we are in a mixed environment
                            error_msg = f"Token path mismatch: Expected {expected_path}, Got {current_stable_path}"
                            logger.warning(f"{error_msg} (ignoring due to distributed setup)")
                            image_processing_error = error_msg
                            image_processing_error_type = "IMAGE_MANIFEST_PATH_MISMATCH"

                    else:
                        return create_error_response(file_id, "Missing 'image_manifest' in data or valid 'selection_token'", "MISSING_DATA")

                    # Proceed with extraction using whatever selection_state we found/created
                    if selection_state:
                        try:
                            # Validate manifest structure
                            manifest = selection_state.get("manifest", [])
                            if not isinstance(manifest, list) or len(manifest) == 0:
                                image_processing_error = "Image manifest is empty or invalid"
                                image_processing_error_type = "IMAGE_MANIFEST_INVALID"
                                logger.warning(f"Invalid image manifest for {file_id}")
                            else:
                                # ... (Keep existing extraction logic) ...
                                allowed_keys = {"pdf_path", "selection_state", "selected_ids", "model", "per_image_max_tokens", "sleep_sec"}
                                describe_kwargs = {
                                    "pdf_path": file_path, # This is the actual temp file path on the server
                                    "selection_state": selection_state,
                                    "selected_ids": selected_image_ids,
                                    "model": data.get("vision_model", "gpt-4o-mini"),
                                    "per_image_max_tokens": int(data.get("vision_max_tokens", 250)),
                                    "sleep_sec": float(data.get("vision_sleep_sec", 0.0)),
                                }
                                describe_kwargs = {k: v for k, v in describe_kwargs.items() if k in allowed_keys}

                                # Analyze images
                                image_descs = describe_selected_images_with_openai(**describe_kwargs)
                        except Exception as e:
                            logger.warning(f"Failed to analyze selected images (inner): {e}")
                            image_processing_error = f"Failed to analyze selected images: {str(e)}"
                            image_processing_error_type = "IMAGE_ANALYSIS_FAILED"
                            image_descs = []

                    try:
                        # Append descriptions to Markdown
                        full_text_md = append_image_descriptions_to_markdown(full_text_md, image_descs)
                    except Exception as e:
                        logger.warning(f"Failed to append image descriptions: {e}")
                        if not image_processing_error:
                            image_processing_error = f"Failed to append image descriptions: {str(e)}"
                            image_processing_error_type = "IMAGE_APPEND_FAILED"

                except Exception as e:
                    logger.warning(f"Image analysis phase failed (non-fatal): {e}")
                    image_processing_error = f"Image analysis phase failed: {str(e)}"
                    image_processing_error_type = "IMAGE_PROCESSING_FAILED"

            # Final Metadata & Response Construction
            try:
                ai_title, ai_description = generate_ai_title_description(full_text_md, Path(file_path).name)
            except Exception as e:
                logger.warning(f"Failed to generate AI title/description: {e}")
                ai_title, ai_description = None, None

            image_count_selected = len(selected_image_ids) if selected_image_ids else 0
            image_count_analyzed = len(image_descs) if image_descs else 0

            extra_data = {
                "file_id": file_id,
                "paragraph_count": full_text_md.count('\n\n') + 1 if full_text_md else 0,
                "table_count": len(document_dict.get("tables", [])) if document_dict else 0,
                "word_count": len(full_text_md.split()) if full_text_md else 0,
                "character_count": len(full_text_md) if full_text_md else 0,
                "has_guid": True,
                "guid_source": "upload" if is_upload else "file_id",
                "ai_title": ai_title,
                "ai_description": ai_description,
                "image_count_total": raw_total_images,
                "image_count_selected": image_count_selected,
                "image_count_analyzed": image_count_analyzed,
                "is_upload": is_upload,
                **extra_meta,
            }

            origin_meta_raw = document_dict.get("origin", {}) if document_dict else {}
            if isinstance(origin_meta_raw, dict):
                extra_data.update(origin_meta_raw)
            elif origin_meta_raw:
                extra_data["origin"] = str(origin_meta_raw)
            
            # Ensure correct filename is used, overriding any origin metadata
            extra_data["filename"] = filename
            
            # Override path/subfolder if provided
            if user_provided_path:
                extra_data["file_path"] = user_provided_path
            if user_provided_subfolder:
                extra_data["subfolder"] = user_provided_subfolder

            complete_metadata = make_complete_metadata(file_path, extra_data)
            
            chunks = []
            if do_chunking:
                try:
                    chunks = perform_chunking(full_text_md, complete_metadata, chunking_options)
                except Exception as e:
                    logger.warning(f"Failed to chunk document: {e}")
                    chunks = []

            # Determine overall success based on image processing errors
            overall_success = not image_processing_error

            resp = {
                "success": overall_success,
                "used_converter": "docling",
                "file_id": file_id,
                "file_path": user_provided_path if user_provided_path else (file_path if not is_upload else None),
                "filename": filename,
                "conversion_status": str(conversion_result.status) if conversion_result else "unknown",
                "document": document_dict,
                "full_text": full_text_md,
                "metadata": complete_metadata,
                "chunks": chunks,
                "is_upload": is_upload
            }
            # Include timestamps at top-level for easy access
            resp["lastModified"] = last_modified_iso
            resp["createdDate"] = created_date_iso
            if selection_state:
                resp["image_manifest"] = selection_state.get("manifest", [])
            if image_descs:
                resp["image_descriptions"] = image_descs

            # Include image processing error information if any occurred
            if image_processing_error:
                resp["error"] = True
                resp["error_type"] = image_processing_error_type
                resp["error_message"] = image_processing_error
                resp["image_processing_error"] = image_processing_error
                resp["image_processing_error_type"] = image_processing_error_type

            # Build a summary for this conversion run (single-file)
            summary = {
                "total_files_processed": 1,
                "docx_count": 1 if ext == '.docx' else 0,
                "pdf_count": 1 if ext == '.pdf' else 0,
                "images_total": raw_total_images,
                "images_selected": image_count_selected,
                "images_analyzed": image_count_analyzed,
                # images ignored = images not analyzed (best-effort)
                "images_ignored": max(0, raw_total_images - image_count_analyzed)
            }
            # Include error information in summary if present
            if image_processing_error:
                summary["image_processing_error"] = image_processing_error
                summary["image_processing_error_type"] = image_processing_error_type

            resp["summary"] = summary

            return jsonify(resp)

        except Exception as e:
            return create_error_response(file_id, f"Docling conversion failed: {str(e)}", "DOCLING_CONVERSION_FAILED", {"exception_type": type(e).__name__})

    except Exception as e:
        # Catch-all for any unexpected errors
        logger.error(f"Unexpected error in docling_convert_file: {e}", exc_info=True)
        return create_error_response(file_id or "unknown", f"Unexpected server error: {str(e)}", "UNEXPECTED_ERROR", {"exception_type": type(e).__name__})

    finally:
        # Clean up temporary file if it was created
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_file_path}: {e}")

@app.route('/file-id/resolve', methods=['GET'])
def resolve_file_endpoint():
    """Resolve a fileID to its file path."""
    try:
        file_id = request.args.get('file_id')
        if not file_id:
            return jsonify({"error": "Missing 'file_id' parameter", "success": False}), 400
        
        file_path = resolve_file_path(file_id)
        
        if not file_path:
            return jsonify({"error": f"FileID not found: {file_id}", "success": False}), 404
        
        return jsonify({
            "success": True,
            "file_id": file_id,
            "file_path": file_path,
            "filename": os.path.basename(file_path)
        }), 200
    except Exception as e:
        logger.error(f"Error in /file-id/resolve: {e}", exc_info=True)
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    try:
        if not os.path.exists(JSON_CACHE_DIR):
            return jsonify({"success": True, "message": "Cache directory doesn't exist, nothing to clear", "files_deleted": 0})
        cache_files = glob.glob(os.path.join(JSON_CACHE_DIR, "*.json"))
        files_deleted = 0
        for cache_file in cache_files:
            try:
                os.remove(cache_file)
                files_deleted += 1
            except Exception as e:
                logger.warning(f"Could not delete {cache_file}: {e}")
        return jsonify({"success": True, "message": f"Cache cleared successfully", "files_deleted": files_deleted, "cache_directory": JSON_CACHE_DIR})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "base_directory": BASE_DIR})

# Docling configuration
pdf_opts = PdfPipelineOptions(
    do_ocr=True,
    do_table_structure=True,
    table_structure_options=TableStructureOptions(
        mode=TableFormerMode.ACCURATE
    )
)

docling_converter = DocumentConverter(
    format_options={
        InputFormat.PDF:  PdfFormatOption(pipeline_options=pdf_opts),
        InputFormat.DOCX: WordFormatOption(),
    }
)

    
@app.route('/image-selection/get', methods=['GET'])
def image_selection_get():
    token = request.args.get('token')
    if not token:
        return jsonify({"error": "missing token"}), 400
    state = load_selection_state(token)
    if not state:
        return jsonify({"error": "invalid or expired token"}), 404
    return jsonify({
        "file_path": state.get("file_path"),
        "guid": state.get("guid"),
        "image_manifest": state.get("manifest", []),
        "created_at": state.get("created_at")
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
