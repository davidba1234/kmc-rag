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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("flask_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DIR = "/mnt/c/Users/Anderson/Documents/n8n/kmc-rag/test_docs"
print(f"BASE_DIR is set to: {BASE_DIR}")
JSON_CACHE_DIR = os.path.join(BASE_DIR, "json_cache")
IMAGE_OUTPUT_DIR = os.path.join(BASE_DIR, "image_output")
os.makedirs(JSON_CACHE_DIR, exist_ok=True)

SUPPORTED_EXTENSIONS = ['.docx', '.pdf']

# --- ALL METADATA FIELDS ALWAYS INCLUDED ---
ALL_METADATA_FIELDS = [
    "file_id",
    "filename",
    "full_path",
    "subfolder",
    "file_extension",
    "file_size",
    "file_size_mb",
    "word_count",
    "character_count",
    "paragraph_count",
    "table_count",
    "last_modified_iso",
    "created_date_iso",
    "has_guid",
    "guid_source",
    "DateReviewed",
    "DateNext",
    "ai_title",
    "ai_description"
]

def get_image_descriptions_from_api(image_info_list):
    # No-op stub: image analysis handled by /docling/convert-file two-step flow.
    return []

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
        "full_path": file_path,
        "subfolder": parent_dir,
        "file_extension": ext,
        "file_size": file_stats.st_size,
        "file_size_mb": round(file_stats.st_size / (1024 * 1024), 2),
        "last_modified_iso": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
        "created_date_iso": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
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

def is_docx_complex(docx_path,
                    min_tables_for_complex=1,
                    min_images_for_complex=1,
                    max_paragraphs_for_simple=300,
                    avg_run_threshold=2.5):
    """Returns True if the DOCX should be processed by Docling (complex)."""
    try:
        with zipfile.ZipFile(docx_path, 'r') as z:
            namelist = z.namelist()
            media_files = [n for n in namelist if n.startswith('word/media/')]
            embedded_objects = [n for n in namelist if n.startswith('word/embeddings') or n.endswith('.bin')]
            has_images = len(media_files) >= min_images_for_complex
            has_embedded = len(embedded_objects) > 0

        doc = Document(docx_path)
        num_tables = len(doc.tables)
        num_paragraphs = len(doc.paragraphs)

        total_runs = 0
        paragraphs_sampled = 0
        for p in doc.paragraphs:
            if not p.text.strip():
                continue
            paragraphs_sampled += 1
            try:
                total_runs += len(p.runs)
            except Exception:
                total_runs += 1
            if paragraphs_sampled >= 200:
                break
        avg_runs = (total_runs / paragraphs_sampled) if paragraphs_sampled else 1

        if num_tables >= min_tables_for_complex:
            return True
        if has_images or has_embedded:
            return True
        if num_paragraphs > max_paragraphs_for_simple:
            return True
        if avg_runs > avg_run_threshold:
            return True

        return False
    except Exception as e:
        logger.warning(f"is_docx_complex failed for {docx_path}: {e}", exc_info=True)
        return True

def extract_tables_with_pdfplumber(pdf_path):
    """Extract tables from PDF using pdfplumber"""
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                for table in page_tables or []:
                    if table and len(table) > 0:
                        clean_table = []
                        for row in table:
                            clean_row = [cell.strip() if cell else "" for cell in row]
                            if any(clean_row):
                                clean_table.append(clean_row)
                        
                        if clean_table:
                            tables.append({
                                "page": page_num + 1,
                                "headers": clean_table[0] if clean_table else [],
                                "rows": clean_table[1:] if len(clean_table) > 1 else [],
                                "raw_data": clean_table
                            })
    except Exception as e:
        logger.warning(f"pdfplumber table extraction failed for {pdf_path}: {e}")
    return tables

def format_table_for_rag(table_data):
    """Format table data for better RAG retrieval"""
    if not table_data or not table_data.get("raw_data"):
        return ""
    
    raw_data = table_data["raw_data"]
    formatted_lines = []
    
    formatted_lines.append("\n--- TABLE START ---")
    
    for row_index, row in enumerate(raw_data):
        if row and any(cell.strip() for cell in row if cell):
            row_text = " | ".join([cell for cell in row if cell and cell.strip()])
            formatted_lines.append(row_text)
            
            if row_index == 0:
                formatted_lines.append("-" * 40)
    
    formatted_lines.append("--- TABLE END ---\n")
    return "\n".join(formatted_lines)

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

def docx_to_json(docx_path):
    logger.info(f"Starting DOCX conversion for file: {docx_path}")
    try:
        doc = Document(docx_path)
        filename_without_ext, _ = os.path.splitext(os.path.basename(docx_path))
        filename_based_id = filename_without_ext.replace(' ', '_').replace('-', '_').lower()
        
        guid = extract_docx_guid(docx_path)
        file_id = guid or filename_based_id
        has_guid = guid is not None

        date_reviewed, date_next = extract_custom_metadata(docx_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        
        # Create complete metadata with all fields
        extra_data = {
            "file_id": file_id,
            "paragraph_count": len(paragraphs),
            "table_count": len(doc.tables),
            "word_count": sum(len(p.split()) for p in paragraphs),
            "character_count": sum(len(p) for p in paragraphs),
            "has_guid": has_guid,
            "guid_source": "docx_settings" if has_guid else "filename_fallback",
            "DateReviewed": date_reviewed,
            "DateNext": date_next
        }
        
        metadata = make_complete_metadata(docx_path, extra_data)

        content = {
            "file_id": file_id,
            "filename": os.path.basename(docx_path),
            "full_path": docx_path,
            "subfolder": os.path.basename(os.path.dirname(docx_path)),
            "paragraphs": paragraphs,
            "tables": [[cell.text.strip() for cell in row.cells] for table in doc.tables for row in table.rows],
            "metadata": metadata
        }
        return content
    except Exception as e:
        logger.error(f"Failed to convert {docx_path}: {e}", exc_info=True)
        return {"error": f"Failed to convert {docx_path}: {str(e)}"}

def pdf_to_json(pdf_path):
    logger.info(f"Starting PDF conversion for file: {pdf_path}")
    try:
        file_id = get_or_create_pdf_guid(pdf_path)
        
        # Image extraction
        doc = fitz.open(pdf_path)
        image_info = []
        image_output_folder = os.path.join(IMAGE_OUTPUT_DIR, file_id)
        os.makedirs(image_output_folder, exist_ok=True)

        for page_index in range(len(doc)):
            page = doc[page_index]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                image_filename = f"page{page_index+1}_img{img_index+1}.{image_ext}"
                image_save_path = os.path.join(image_output_folder, image_filename)
                
                with open(image_save_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                image_info.append({
                    "path": image_save_path,
                    "page": page_index + 1,
                    "xref": xref
                })

        # Extract tables using pdfplumber
        extracted_tables = extract_tables_with_pdfplumber(pdf_path)
        
        # Extract text using PyMuPDF
        full_text = "".join(page.get_textpage().extractText() for page in doc)
        paragraphs = [p.strip() for p in full_text.split('\n') if p.strip()]
        
        # Convert tables to format expected by chunking code
        tables_for_chunking = []
        for table in extracted_tables:
            if table.get("raw_data"):
                tables_for_chunking.extend(table["raw_data"])

        # Create complete metadata with all fields
        extra_data = {
            "file_id": file_id,
            "paragraph_count": len(paragraphs),
            "table_count": len(extracted_tables),
            "word_count": len(full_text.split()),
            "character_count": len(full_text),
            "has_guid": True,
            "guid_source": "pdf_metadata",
            "DateReviewed": None,
            "DateNext": None
        }
        
        metadata = make_complete_metadata(pdf_path, extra_data)
        
        content = {
            "file_id": file_id,
            "filename": os.path.basename(pdf_path),
            "full_path": pdf_path,
            "subfolder": os.path.basename(os.path.dirname(pdf_path)),
            "paragraphs": paragraphs,
            "tables": tables_for_chunking,
            "images": image_info,
            "extracted_tables": extracted_tables,
            "metadata": metadata
        }
        
        doc.close()
        return content
    except Exception as e:
        logger.error(f"Failed to convert {pdf_path}: {e}", exc_info=True)
        return {"error": f"Failed to convert {pdf_path}: {str(e)}"}

def convert_file_to_json(file_path):
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    
    if extension == '.docx':
        return docx_to_json(file_path)
    elif extension == '.pdf':
        return pdf_to_json(file_path)
    else:
        logger.warning(f"Unsupported file type: {extension} for file {file_path}")
        return {"error": f"Unsupported file type: {extension}"}

# --- Caching and File System Functions ---

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
                        "last_modified_iso": datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                    })
                except OSError as e:
                    logger.warning(f"Could not stat file {file_path}: {e}")
                    continue
    
    return supported_files
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


@app.route('/debug-paths', methods=['GET'])
def debug_paths():
    return jsonify({
        "BASE_DIR": BASE_DIR,
        "BASE_DIR_exists": os.path.exists(BASE_DIR),
        "BASE_DIR_absolute": os.path.abspath(BASE_DIR),
        "files_in_base": os.listdir(BASE_DIR) if os.path.exists(BASE_DIR) else "Directory not found"
    })

@app.route('/debug-file-search', methods=['GET'])
def debug_file_search():
    debug_info = {
        "BASE_DIR": BASE_DIR,
        "BASE_DIR_exists": os.path.exists(BASE_DIR),
        "SUPPORTED_EXTENSIONS": SUPPORTED_EXTENSIONS,
        "search_results": {}
    }
    
    for ext in SUPPORTED_EXTENSIONS:
        pattern = os.path.join(BASE_DIR, "**", f"*{ext}")
        files = glob.glob(pattern, recursive=True)
        debug_info["search_results"][ext] = {
            "pattern": pattern,
            "files_found": files
        }
    
    # Also try manual directory walk
    manual_files = []
    if os.path.exists(BASE_DIR):
        for root, dirs, files in os.walk(BASE_DIR):
            for file in files:
                if any(file.lower().endswith(ext.lower()) for ext in SUPPORTED_EXTENSIONS):
                    manual_files.append(os.path.join(root, file))
    
    debug_info["manual_walk_results"] = manual_files
    return jsonify(debug_info)

@app.route('/ensure-guids', methods=['POST'])
def ensure_guids():
    """Scans all supported files and ensures that any file missing a GUID gets one."""
    logger.info("Received request for /ensure-guids. Scanning all files.")
    try:
        all_files = find_all_supported_files()
        files_updated = 0
        
        for file_info in all_files:
            path = file_info['full_path']
            if path.lower().endswith('.docx'):
                if extract_docx_guid(path):
                    continue  # DOCX GUIDs are read-only
            elif path.lower().endswith('.pdf'):
                existing_guid = read_pdf_guid(path)
                if not existing_guid:
                    get_or_create_pdf_guid(path)
                    files_updated += 1

        message = f"Scan complete. Updated {files_updated} PDF file(s) with a new GUID."
        logger.info(message)
        return jsonify({"success": True, "message": message, "files_updated": files_updated})

    except Exception as e:
        logger.error(f"An error occurred in /ensure-guids: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/list-files', methods=['GET'])
def list_files():
    try:
        file_info_list = find_all_supported_files() 
        
        for file_info in file_info_list:
            try:
                path = file_info['full_path']
                mod_time_unix = os.path.getmtime(path)
                mod_time_iso = datetime.fromtimestamp(mod_time_unix, tz=timezone.utc).isoformat()
                
                file_id = None
                if path.lower().endswith('.docx'):
                    file_id = extract_docx_guid(path)
                elif path.lower().endswith('.pdf'):
                    file_id = read_pdf_guid(path)

                if not file_id:
                    filename_without_ext = os.path.splitext(os.path.basename(path))[0]
                    file_id = f"fallback_{filename_without_ext.replace(' ', '_').lower()}"

                file_info['lastModified'] = mod_time_iso
                file_info['file_id'] = file_id

            except Exception as inner_e:
                logger.error(f"Error processing file {file_info.get('full_path', 'N/A')}: {inner_e}")

        return jsonify({"files": file_info_list, "count": len(file_info_list)})

    except Exception as e:
        logger.error(f"Error in /list-files: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/docling/convert-file', methods=['POST'])
def docling_convert_file():
    try:
        data = request.get_json(force=True)
        if not data or 'file_path' not in data:
            return jsonify({"error": "Missing 'file_path' in JSON request body"}), 400
        file_path = data['file_path']
        
        # [Existing Params]
        force_refresh = data.get("force_refresh", False)
        do_chunking = data.get("do_chunking", False)
        chunking_options = data.get("chunking_options", {})
        image_mode = data.get("image_mode", "ask")
        selection_token = data.get("selection_token")
        selected_image_ids = data.get("selected_image_ids", []) or []

        if not os.path.exists(file_path):
            return jsonify({"error": f"File not found: {file_path}"}), 404
        if not os.path.abspath(file_path).startswith(os.path.abspath(BASE_DIR)):
            return jsonify({"error": "File path is outside allowed directory"}), 403

        ext = os.path.splitext(file_path)[1].lower()

        # --- [1. COUNT IMAGES RAW (FAST)] ---
        raw_total_images = 0
        if ext == '.pdf':
            try:
                with fitz.open(file_path) as count_doc:
                    for page in count_doc:
                        raw_total_images += len(page.get_images(full=True))
            except Exception as e:
                logger.warning(f"Could not perform raw image count: {e}")

        # --- [2. GET/CREATE GUID] ---
        guid = None
        guid_source = "unknown"
        if ext == '.pdf':
            guid = get_or_create_pdf_guid(file_path)
            guid_source = "pdf_metadata"
        elif ext == '.docx':
            guid = extract_docx_guid(file_path)
            if guid:
                guid_source = "docx_settings"
            else:
                filename_without_ext = os.path.splitext(os.path.basename(file_path))[0]
                guid = filename_without_ext.replace(' ', '_').replace('-', '_').lower()
                guid_source = "filename_fallback"

        # --- [3. DECIDE CONVERTER] ---
        use_docling = False
        if ext == '.pdf':
            use_docling = True
        elif ext == '.docx':
            use_docling = is_docx_complex(file_path)
        else:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400

        # ==================================================================
        # [FAST PATH] IMAGE SCANNING (Moved BEFORE Docling Conversion)
        # ==================================================================
        if use_docling and ext == ".pdf" and image_mode == "ask":
            logger.info(f"Fast scanning PDF for images: {file_path}")
            # This uses fitz (PyMuPDF) which is instant compared to Docling
            image_manifest = build_pdf_image_manifest(file_path, guid)
            
            if len(image_manifest) > 0:
                token_seed = f"{guid}:{file_path}:{time.time()}:{len(image_manifest)}"
                token = hashlib.sha256(token_seed.encode("utf-8")).hexdigest()[:32]
                selection_state = {
                    "file_path": file_path,
                    "guid": guid,
                    "manifest": image_manifest,
                    "created_at": datetime.utcnow().isoformat() + "Z"
                }
                save_selection_state(token, selection_state)
                
                # RETURN IMMEDIATELY - Do not run Docling OCR yet
                return jsonify({
                    "success": True,
                    "used_converter": "docling_fast_scan",
                    "needs_image_selection": True,
                    "message": "Images found. Select images and resubmit with image_mode='selected'.",
                    "file_path": file_path,
                    "filename": os.path.basename(file_path),
                    "conversion_status": "pending_selection",
                    "image_manifest": image_manifest,
                    "selection_token": token,
                    "total_images_detected": raw_total_images 
                })
            else:
                logger.info("No images found during fast scan. Proceeding to auto-conversion.")
                # If no images, we fall through to the standard heavy conversion below.

        # ==================================================================
        # [SLOW PATH] STANDARD CONVERSION (Native DOCX or Docling)
        # ==================================================================

        # [Path A: Native DOCX]
        if ext == '.docx' and not use_docling:
            # ... (Keep your existing native DOCX logic exactly as is) ...
            content = docx_to_json(file_path)
            if "error" in content:
                return jsonify({"error": content["error"]}), 500
            
            # (Reconstruct metadata logic for native docx...)
            date_reviewed, date_next = extract_custom_metadata(file_path)
            full_text = "\n\n".join(content.get("paragraphs", []))
            ai_title, ai_description = generate_ai_title_description(full_text, os.path.basename(file_path))
            
            extra_data = {
                "file_id": guid,
                "paragraph_count": len(content.get("paragraphs", [])),
                "table_count": len(content.get("tables", [])),
                "word_count": len(full_text.split()),
                "character_count": len(full_text),
                "has_guid": bool(guid),
                "guid_source": guid_source,
                "DateReviewed": date_reviewed,
                "DateNext": date_next,
                "ai_title": ai_title,
                "ai_description": ai_description,
                "image_count_total": 0,
                "image_count_selected": 0,
                "image_count_analyzed": 0,
            }
            complete_metadata = make_complete_metadata(file_path, extra_data)
            chunks = []
            if do_chunking:
                chunks = perform_chunking(full_text, complete_metadata, chunking_options)

            return jsonify({
                "success": True,
                "used_converter": "native_docx",
                "file_path": file_path,
                "filename": content.get("filename"),
                "full_text": full_text,
                "structured_content": {
                    "paragraphs": content.get("paragraphs"),
                    "tables": content.get("tables"),
                    "metadata": content.get("metadata")
                },
                "metadata": complete_metadata,
                "chunks": chunks
            })

        # [Path B: Heavy Docling Conversion]
        # We only reach here if:
        # 1. It's a complex DOCX
        # 2. It's a PDF AND (image_mode='selected' OR image_mode='skip' OR image_mode='ask' but no images found)
        
        logger.info("Running Docling converter (Heavy Process)...")
        conversion_result = docling_converter.convert(file_path)
        document = conversion_result.document
        document_dict = document.export_to_dict()
        full_text_md = document.export_to_markdown()

        extra_meta = {}
        if ext == ".docx":
            date_reviewed, date_next = extract_custom_metadata(file_path)
            extra_meta.update({"DateReviewed": date_reviewed, "DateNext": date_next})

        # Handle Image Mode = SELECTED
        image_descs = []
        selection_state = None
        
        if ext == ".pdf" and image_mode == "selected":
            if not selection_token:
                return jsonify({"error": "selection_token is required when image_mode='selected'"}), 400
            
            selection_state = load_selection_state(selection_token)
            if not selection_state or selection_state.get("file_path") != file_path:
                return jsonify({"error": "Invalid or expired selection_token"}), 400

            allowed_keys = {"pdf_path", "selection_state", "selected_ids", "model", "per_image_max_tokens", "sleep_sec"}
            describe_kwargs = {
                "pdf_path": file_path,
                "selection_state": selection_state,
                "selected_ids": selected_image_ids,
                "model": data.get("vision_model", "gpt-4o-mini"),
                "per_image_max_tokens": int(data.get("vision_max_tokens", 250)),
                "sleep_sec": float(data.get("vision_sleep_sec", 0.0)),
            }
            describe_kwargs = {k: v for k, v in describe_kwargs.items() if k in allowed_keys}
        
            # Analyze images
            image_descs = describe_selected_images_with_openai(**describe_kwargs)
            # Append descriptions to Markdown
            full_text_md = append_image_descriptions_to_markdown(full_text_md, image_descs)

        # Final Metadata & Response Construction
        ai_title, ai_description = generate_ai_title_description(full_text_md, Path(file_path).name)

        image_count_selected = len(selected_image_ids) if selected_image_ids else 0
        image_count_analyzed = len(image_descs) if image_descs else 0

        extra_data = {
            "file_id": guid,
            "paragraph_count": full_text_md.count('\n\n') + 1,
            "table_count": len(document_dict.get("tables", [])),
            "word_count": len(full_text_md.split()),
            "character_count": len(full_text_md),
            "has_guid": bool(guid),
            "guid_source": guid_source,
            "ai_title": ai_title,
            "ai_description": ai_description,
            "image_count_total": raw_total_images,
            "image_count_selected": image_count_selected,
            "image_count_analyzed": image_count_analyzed,
            **extra_meta,
        }

        origin_meta_raw = document_dict.get("origin", {})
        if isinstance(origin_meta_raw, dict):
            extra_data.update(origin_meta_raw)
        elif origin_meta_raw:
            extra_data["origin"] = str(origin_meta_raw)

        complete_metadata = make_complete_metadata(file_path, extra_data)
        
        chunks = []
        if do_chunking:
            chunks = perform_chunking(full_text_md, complete_metadata, chunking_options)

        resp = {
            "success": True,
            "used_converter": "docling",
            "file_path": file_path,
            "filename": os.path.basename(file_path),
            "conversion_status": str(conversion_result.status),
            "document": document_dict,
            "full_text": full_text_md,
            "metadata": complete_metadata,
            "chunks": chunks
        }
        if selection_state:
            resp["image_manifest"] = selection_state.get("manifest", [])
        if image_descs:
            resp["image_descriptions"] = image_descs

        return jsonify(resp)

    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500
    try:
        data = request.get_json(force=True)
        if not data or 'file_path' not in data:
            return jsonify({"error": "Missing 'file_path' in JSON request body"}), 400
        file_path = data['file_path']
        
        # [Existing Params]
        force_refresh = data.get("force_refresh", False)
        do_chunking = data.get("do_chunking", False)
        chunking_options = data.get("chunking_options", {})
        image_mode = data.get("image_mode", "ask")
        selection_token = data.get("selection_token")
        selected_image_ids = data.get("selected_image_ids", []) or []

        if not os.path.exists(file_path):
            return jsonify({"error": f"File not found: {file_path}"}), 404
        if not os.path.abspath(file_path).startswith(os.path.abspath(BASE_DIR)):
            return jsonify({"error": "File path is outside allowed directory"}), 403

        ext = os.path.splitext(file_path)[1].lower()

        # --- [FIX: UNCONDITIONAL RAW IMAGE COUNT] ---
        # We count the images immediately. This ensures the total is always correct,
        # even if image_mode="skip" or if selection_state is never loaded.
        raw_total_images = 0
        if ext == '.pdf':
            try:
                with fitz.open(file_path) as count_doc:
                    for page in count_doc:
                        # get_images(full=True) counts everything, including vector/small images
                        raw_total_images += len(page.get_images(full=True))
            except Exception as e:
                logger.warning(f"Could not perform raw image count on {file_path}: {e}")
        # --------------------------------------------

        # [Existing GUID Logic]
        guid = None
        guid_source = "unknown"
        if ext == '.pdf':
            guid = get_or_create_pdf_guid(file_path)
            guid_source = "pdf_metadata"
        elif ext == '.docx':
            guid = extract_docx_guid(file_path)
            if guid:
                guid_source = "docx_settings"
            else:
                filename_without_ext = os.path.splitext(os.path.basename(file_path))[0]
                guid = filename_without_ext.replace(' ', '_').replace('-', '_').lower()
                guid_source = "filename_fallback"

        # [Existing Converter Decision]
        use_docling = False
        if ext == '.pdf':
            use_docling = True
        elif ext == '.docx':
            use_docling = is_docx_complex(file_path)
        else:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400

        logger.info(f"convert-file: {file_path} selected converter = {'docling' if use_docling else 'native'}")

        # [Existing Native DOCX Path]
        if ext == '.docx' and not use_docling:
            content = docx_to_json(file_path)
            if "error" in content:
                return jsonify({"error": content["error"]}), 500

            date_reviewed, date_next = extract_custom_metadata(file_path)
            full_text = "\n\n".join(content.get("paragraphs", []))
            ai_title, ai_description = generate_ai_title_description(full_text, os.path.basename(file_path))

            extra_data = {
                "file_id": guid,
                "paragraph_count": len(content.get("paragraphs", [])),
                "table_count": len(content.get("tables", [])),
                "word_count": len(full_text.split()),
                "character_count": len(full_text),
                "has_guid": bool(guid),
                "guid_source": guid_source,
                "DateReviewed": date_reviewed,
                "DateNext": date_next,
                "ai_title": ai_title,
                "ai_description": ai_description,
                "image_count_total": 0,
                "image_count_selected": 0,
                "image_count_analyzed": 0,
            }
            complete_metadata = make_complete_metadata(file_path, extra_data)
            chunks = []
            if do_chunking:
                chunks = perform_chunking(full_text, complete_metadata, chunking_options)

            return jsonify({
                "success": True,
                "used_converter": "native_docx",
                "file_path": file_path,
                "filename": content.get("filename"),
                "full_text": full_text,
                "structured_content": {
                    "paragraphs": content.get("paragraphs"),
                    "tables": content.get("tables"),
                    "metadata": content.get("metadata")
                },
                "metadata": complete_metadata,
                "chunks": chunks
            })

        # [Existing Docling Path]
        logger.info("Running Docling converter...")
        conversion_result = docling_converter.convert(file_path)
        document = conversion_result.document
        document_dict = document.export_to_dict()
        full_text_md = document.export_to_markdown()

        extra_meta = {}
        if ext == ".docx":
            date_reviewed, date_next = extract_custom_metadata(file_path)
            extra_meta.update({"DateReviewed": date_reviewed, "DateNext": date_next})

        image_manifest = []
        selection_state = None
        image_descs = []

        if ext == ".pdf":
            if image_mode == "ask":
                image_manifest = build_pdf_image_manifest(file_path, guid)
                
                if len(image_manifest) > 0:
                    token_seed = f"{guid}:{file_path}:{time.time()}:{len(image_manifest)}"
                    token = hashlib.sha256(token_seed.encode("utf-8")).hexdigest()[:32]
                    selection_state = {
                        "file_path": file_path,
                        "guid": guid,
                        "manifest": image_manifest,
                        "created_at": datetime.utcnow().isoformat() + "Z"
                    }
                    save_selection_state(token, selection_state)
                    
                    return jsonify({
                        "success": True,
                        "used_converter": "docling",
                        "needs_image_selection": True,
                        "message": "Select which images to analyze.",
                        "file_path": file_path,
                        "filename": os.path.basename(file_path),
                        "conversion_status": str(conversion_result.status),
                        "image_manifest": image_manifest,
                        "selection_token": token,
                        "total_images_detected": raw_total_images 
                    })

            elif image_mode == "selected":
                if not selection_token:
                    return jsonify({"error": "selection_token is required when image_mode='selected'"}), 400
                selection_state = load_selection_state(selection_token)
                if not selection_state or selection_state.get("file_path") != file_path:
                    return jsonify({"error": "Invalid or expired selection_token"}), 400

                allowed_keys = {"pdf_path", "selection_state", "selected_ids", "model", "per_image_max_tokens", "sleep_sec"}
                describe_kwargs = {
                    "pdf_path": file_path,
                    "selection_state": selection_state,
                    "selected_ids": selected_image_ids,
                    "model": data.get("vision_model", "gpt-4o-mini"),
                    "per_image_max_tokens": int(data.get("vision_max_tokens", 250)),
                    "sleep_sec": float(data.get("vision_sleep_sec", 0.0)),
                }
                describe_kwargs = {k: v for k, v in describe_kwargs.items() if k in allowed_keys}
           
                image_descs = describe_selected_images_with_openai(**describe_kwargs)
                full_text_md = append_image_descriptions_to_markdown(full_text_md, image_descs)

        # [Existing AI Title]
        ai_title, ai_description = generate_ai_title_description(full_text_md, Path(file_path).name)

        # --- [FIX: USE THE RAW COUNT] ---
        image_count_selected = len(selected_image_ids) if selected_image_ids else 0
        image_count_analyzed = len(image_descs) if image_descs else 0

        extra_data = {
            "file_id": guid,
            "paragraph_count": full_text_md.count('\n\n') + 1,
            "table_count": len(document_dict.get("tables", [])),
            "word_count": len(full_text_md.split()),
            "character_count": len(full_text_md),
            "has_guid": bool(guid),
            "guid_source": guid_source,
            "ai_title": ai_title,
            "ai_description": ai_description,
            
            # Use the raw count we calculated at the top
            "image_count_total": raw_total_images, 
            
            "image_count_selected": image_count_selected,
            "image_count_analyzed": image_count_analyzed,
            **extra_meta,
        }
        # ---------------------------------

        origin_meta_raw = document_dict.get("origin", {})
        if isinstance(origin_meta_raw, dict):
            extra_data.update(origin_meta_raw)
        elif origin_meta_raw:
            extra_data["origin"] = str(origin_meta_raw)

        complete_metadata = make_complete_metadata(file_path, extra_data)
        
        chunks = []
        if do_chunking:
            chunks = perform_chunking(full_text_md, complete_metadata, chunking_options)

        resp = {
            "success": True,
            "used_converter": "docling",
            "file_path": file_path,
            "filename": os.path.basename(file_path),
            "conversion_status": str(conversion_result.status),
            "document": document_dict,
            "full_text": full_text_md,
            "metadata": complete_metadata,
            "chunks": chunks
        }
        if selection_state:
            resp["image_manifest"] = selection_state.get("manifest", [])
        if image_descs:
            resp["image_descriptions"] = image_descs

        return jsonify(resp)

    except Exception as e:
        logger.error(f"Error processing file with Docling/native: {e}", exc_info=True)
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

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

@app.route('/docling/convert-file', methods=['POST'])
def docling_convert_file():
    try:
        data = request.get_json(force=True)
        if not data or 'file_path' not in data:
            return jsonify({"error": "Missing 'file_path' in JSON request body"}), 400
        file_path = data['file_path']
        force_refresh = data.get("force_refresh", False)
        do_chunking = data.get("do_chunking", False)
        chunking_options = data.get("chunking_options", {})

        # New image-selection params
        image_mode = data.get("image_mode", "ask")  # "ask" | "selected" | "skip"
        selection_token = data.get("selection_token")
        selected_image_ids = data.get("selected_image_ids", []) or []

        if not os.path.exists(file_path):
            return jsonify({"error": f"File not found: {file_path}"}), 404
        if not os.path.abspath(file_path).startswith(os.path.abspath(BASE_DIR)):
            return jsonify({"error": "File path is outside allowed directory"}), 403

        ext = os.path.splitext(file_path)[1].lower()

        # Get GUID
        guid = None
        guid_source = "unknown"
        if ext == '.pdf':
            guid = get_or_create_pdf_guid(file_path)
            guid_source = "pdf_metadata"
        elif ext == '.docx':
            guid = extract_docx_guid(file_path)
            if guid:
                guid_source = "docx_settings"
            else:
                filename_without_ext = os.path.splitext(os.path.basename(file_path))[0]
                guid = filename_without_ext.replace(' ', '_').replace('-', '_').lower()
                guid_source = "filename_fallback"

        # Decide converter
        use_docling = False
        if ext == '.pdf':
            use_docling = True
        elif ext == '.docx':
            use_docling = is_docx_complex(file_path)
        else:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400

        logger.info(f"convert-file: {file_path} selected converter = {'docling' if use_docling else 'native'}")

        # Native lightweight docx path (unchanged)
        if ext == '.docx' and not use_docling:
            content = docx_to_json(file_path)
            if "error" in content:
                return jsonify({"error": content["error"]}), 500

            date_reviewed, date_next = extract_custom_metadata(file_path)
            full_text = "\n\n".join(content.get("paragraphs", []))
            ai_title, ai_description = generate_ai_title_description(
                full_text,
                os.path.basename(file_path),
            )

            extra_data = {
                "file_id": guid,
                "paragraph_count": len(content.get("paragraphs", [])),
                "table_count": len(content.get("tables", [])),
                "word_count": len(full_text.split()),
                "character_count": len(full_text),
                "has_guid": bool(guid),
                "guid_source": guid_source,
                "DateReviewed": date_reviewed,
                "DateNext": date_next,
                "ai_title": ai_title,
                "ai_description": ai_description,
                "image_count_total": 0,
                "image_count_selected": 0,
                "image_count_analyzed": 0,
            }
            complete_metadata = make_complete_metadata(file_path, extra_data)

            chunks = []
            if do_chunking:
                chunks = perform_chunking(full_text, complete_metadata, chunking_options)

            return jsonify({
                "success": True,
                "used_converter": "native_docx",
                "file_path": file_path,
                "filename": content.get("filename"),
                "full_text": full_text,
                "structured_content": {
                    "paragraphs": content.get("paragraphs"),
                    "tables": content.get("tables"),
                    "metadata": content.get("metadata")
                },
                "metadata": complete_metadata,
                "chunks": chunks
            })

        # DOC LING path for PDFs and complex DOCX
        logger.info("Running Docling converter...")
        conversion_result = docling_converter.convert(file_path)
        document = conversion_result.document
        document_dict = document.export_to_dict()
        full_text_md = document.export_to_markdown()

        # Custom metadata for DOCX
        extra_meta = {}
        if ext == ".docx":
            date_reviewed, date_next = extract_custom_metadata(file_path)
            extra_meta.update({"DateReviewed": date_reviewed, "DateNext": date_next})

        # If PDF and image_mode == "ask", first return manifest for user selection
        image_manifest = []
        selection_state = None
        image_descs = []

        if ext == ".pdf":
            if image_mode == "ask":
                image_manifest = build_pdf_image_manifest(file_path, guid)
                
                # === NEW LOGIC START ===
                # Only interrupt the flow if images actually exist
                if len(image_manifest) > 0:
                    token_seed = f"{guid}:{file_path}:{time.time()}:{len(image_manifest)}"
                    token = hashlib.sha256(token_seed.encode("utf-8")).hexdigest()[:32]
                    selection_state = {
                        "file_path": file_path,
                        "guid": guid,
                        "manifest": image_manifest,
                        "created_at": datetime.utcnow().isoformat() + "Z"
                    }
                    save_selection_state(token, selection_state)
                    
                    # Return manifest for UI to choose images
                    return jsonify({
                        "success": True,
                        "used_converter": "docling",
                        "needs_image_selection": True,
                        "message": "Select which images to analyze and POST back using image_mode='selected'.",
                        "file_path": file_path,
                        "filename": os.path.basename(file_path),
                        "conversion_status": str(conversion_result.status),
                        "image_manifest": image_manifest,
                        "selection_token": token
                    })
                # If image_manifest is empty, we do NOT return. 
                # We fall through to the code below to process text immediately.
                # === NEW LOGIC END ===

            elif image_mode == "selected":
                if not selection_token:
                    return jsonify({"error": "selection_token is required when image_mode='selected'"}), 400
                selection_state = load_selection_state(selection_token)
                if not selection_state or selection_state.get("file_path") != file_path:
                    return jsonify({"error": "Invalid or expired selection_token for this file_path"}), 400

                allowed_keys = {
                    "pdf_path",
                    "selection_state",
                    "selected_ids",
                    "model",
                    "per_image_max_tokens",
                    "sleep_sec",
                }
                describe_kwargs = {
                    "pdf_path": file_path,
                    "selection_state": selection_state,
                    "selected_ids": selected_image_ids,
                    "model": data.get("vision_model", "gpt-4o-mini"),
                    "per_image_max_tokens": int(data.get("vision_max_tokens", 250)),
                    "sleep_sec": float(data.get("vision_sleep_sec", 0.0)),
                }
                describe_kwargs = {k: v for k, v in describe_kwargs.items() if k in allowed_keys}
           
                # Analyze only selected images
                image_descs = describe_selected_images_with_openai(**describe_kwargs)
                # Merge image descriptions into markdown
                full_text_md = append_image_descriptions_to_markdown(full_text_md, image_descs)
            
            else:
                # image_mode == "skip": do nothing
                pass

        # Generate AI title/description on the final text (with image descriptions if any)
        ai_title, ai_description = generate_ai_title_description(
            full_text_md,
            Path(file_path).name,
        )

        # Compose metadata
        image_count_total = len(selection_state["manifest"]) if selection_state else 0
        image_count_selected = len(selected_image_ids) if selected_image_ids else 0
        image_count_analyzed = len(image_descs) if image_descs else 0

        extra_data = {
            "file_id": guid,
            "paragraph_count": full_text_md.count('\n\n') + 1,
            "table_count": len(document_dict.get("tables", [])),
            "word_count": len(full_text_md.split()),
            "character_count": len(full_text_md),
            "has_guid": bool(guid),
            "guid_source": guid_source,
            "ai_title": ai_title,
            "ai_description": ai_description,
            "image_count_total": image_count_total,
            "image_count_selected": image_count_selected,
            "image_count_analyzed": image_count_analyzed,
            **extra_meta,
        }
        # Include docling origin metadata if present
        origin_meta_raw = document_dict.get("origin", {})
        if isinstance(origin_meta_raw, dict):
            extra_data.update(origin_meta_raw)
        elif origin_meta_raw:
            extra_data["origin"] = str(origin_meta_raw)

        complete_metadata = make_complete_metadata(file_path, extra_data)

        # Optional chunking
        chunks = []
        if do_chunking:
            chunks = perform_chunking(full_text_md, complete_metadata, chunking_options)

        # Build response
        resp = {
            "success": True,
            "used_converter": "docling",
            "file_path": file_path,
            "filename": os.path.basename(file_path),
            "conversion_status": str(conversion_result.status),
            "document": document_dict,
            "full_text": full_text_md,
            "metadata": complete_metadata,
            "chunks": chunks
        }
        # If we analyzed images, return both the manifest (for reference) and descriptions
        if selection_state:
            resp["image_manifest"] = selection_state.get("manifest", [])
        if image_descs:
            resp["image_descriptions"] = image_descs

        return jsonify(resp)

    except Exception as e:
        logger.error(f"Error processing file with Docling/native: {e}", exc_info=True)
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500
    
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

@app.route('/docling/convert-all', methods=['POST'])
def docling_convert_all():
    """Batch-convert all supported files under BASE_DIR using the Docling path."""
    results = []
    files = find_all_supported_files()
    for f in files:
        path = f.get("full_path")
        try:
            conversion_result = docling_converter.convert(path)
            results.append({
                "file": path,
                "status": str(conversion_result.status),
                "success": True
            })
        except Exception as e:
            logger.error(f"Batch convert failed for {path}: {e}", exc_info=True)
            results.append({"file": path, "success": False, "error": str(e)})
    return jsonify({"results": results, "count": len(results)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)