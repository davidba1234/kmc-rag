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
    PdfPipelineOptions, TableStructureOptions, TableFormerMode, PictureDescriptionVlmOptions, 
)
from docling_core.types.doc import ImageRefMode, PictureItem
from docling.datamodel.base_models import InputFormat
import pdfplumber
import tempfile
import zipfile
from chonkie import SemanticChunker # type: ignore
import threading


os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

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

BASE_DIR = r"C:\\Users\\ander\\Documents\\n8n\\test_docs"
JSON_CACHE_DIR = os.path.join(BASE_DIR, "json_cache")
IMAGE_OUTPUT_DIR = os.path.join(BASE_DIR, "image_output")
os.makedirs(JSON_CACHE_DIR, exist_ok=True)

try:
    BACKGROUND_JOBS
except NameError:
    BACKGROUND_JOBS = {}

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

def get_image_descriptions_from_api(image_info_list: list) -> list:
    """Takes a list of image paths and calls a vision model for each."""
    try:
        client = OpenAI()
    except Exception as e:
        logger.error(f"OpenAI client failed to initialize. Is OPENAI_API_KEY set? Error: {e}")
        return []

    descriptions = []
    for image_info in image_info_list:
        path = image_info.get("path")
        try:
            with open(path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": "Describe this image from a document in detail. If it's a chart or graph, explain what it shows."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                max_tokens=300,
            )
            descriptions.append({"page": image_info.get("page"), "description": response.choices[0].message.content})
        except Exception as e:
            logger.error(f"Failed to get description for image {path}: {e}")
    return descriptions

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
    for ext in SUPPORTED_EXTENSIONS:
        pattern = os.path.join(BASE_DIR, "**", f"*{ext}")
        for file_path in glob.glob(pattern, recursive=True):
            if not os.path.basename(file_path).startswith('~$'):
                file_stats = os.stat(file_path)
                parent_dir = os.path.basename(os.path.dirname(file_path))
                supported_files.append({
                    "filename": os.path.basename(file_path),
                    "full_path": file_path,
                    "subfolder": parent_dir,
                    "last_modified_iso": datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                })
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

def _gather_docling_picture_image_info(document, out_dir):
    """
    Returns a list of {"path": <image_path>, "page": <int or None>} for PictureItem(s)
    Tries to use paths Docling created (because generate_picture_images=True),
    and falls back to bytes if available.
    """
    os.makedirs(out_dir, exist_ok=True)
    results = []
    idx = 0
    for item, _lvl in document.iterate_items():
        if not isinstance(item, PictureItem):
            continue

        # Prefer existing file path emitted by Docling
        candidate_path = None
        for attr in ("image_path", "image_file", "path", "file_path"):
            p = getattr(item, attr, None)
            if isinstance(p, str) and os.path.exists(p):
                candidate_path = p
                break

        # Fall back to extracting bytes if available
        if not candidate_path:
            img_bytes = None
            for a in ("get_image_bytes", "image_bytes", "get_image"):
                m = getattr(item, a, None)
                if callable(m):
                    try:
                        b = m()
                        if isinstance(b, (bytes, bytearray)):
                            img_bytes = bytes(b)
                            break
                        # Handle PIL.Image return
                        if hasattr(b, "save"):
                            buf = io.BytesIO()
                            try:
                                b.save(buf, format="JPEG")
                                img_bytes = buf.getvalue()
                                break
                            except Exception:
                                pass
                    except Exception:
                        continue
            if img_bytes:
                candidate_path = os.path.join(out_dir, f"docling_img_{idx}.jpg")
                with open(candidate_path, "wb") as f:
                    f.write(img_bytes)

        if candidate_path and os.path.exists(candidate_path):
            results.append({
                "path": candidate_path,
                "page": getattr(item, "page", None)
            })
            idx += 1

    return results

def _inject_descriptions_into_markdown(document, descriptions):
    """
    Export markdown with placeholders and interleave [Image Description: ...] at each <!-- image -->.
    """
    md_with_ph = document.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)
    parts = md_with_ph.split("<!-- image -->")
    if len(parts) == 1:
        # No placeholders; append descriptions at the end
        tail = "\n".join(f"[Image Description: {d or 'Image'}]" for d in descriptions)
        return md_with_ph + ("\n" + tail if tail else "")

    out = [parts[0]]
    for i in range(1, len(parts)):
        d = descriptions[i-1] if (i-1) < len(descriptions) else "Image"
        out.append(f"[Image Description: {d or 'Image'}]")
        out.append(parts[i])
    return "".join(out)

def _make_guid_for_file(file_path, ext):
    """Match your existing GUID logic."""
    if ext == ".pdf":
        guid = get_or_create_pdf_guid(file_path)
        source = "pdf_metadata"
    elif ext == ".docx":
        guid = extract_docx_guid(file_path)
        if guid:
            source = "docx_settings"
        else:
            stem = os.path.splitext(os.path.basename(file_path))[0]
            guid = stem.replace(' ', '_').replace('-', '_').lower()
            source = "filename_fallback"
    else:
        guid, source = None, "unknown"
    return guid, source

def _docling_convert_worker(job_id, file_path, force_refresh=False, do_chunking=False, chunking_options=None):
    """
    Background worker for /docling/convert-file.
    Produces the same-shaped JSON you return today, but uses OpenAI for image descriptions.
    """
    BACKGROUND_JOBS[job_id]["status"] = "running"
    BACKGROUND_JOBS[job_id]["started_at"] = time.time()
    chunking_options = chunking_options or {}
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # GUID + source (same semantics as your current route)
        guid, guid_source = _make_guid_for_file(file_path, ext)

        # Decide converter (align with your current logic)
        use_docling = (ext == ".pdf") or (ext == ".docx" and is_docx_complex(file_path))
        logger.info(f"[job {job_id}] converter = {'docling' if use_docling else 'native_docx'} for {file_path}")

        # Native DOCX path (unchanged semantics)
        if ext == ".docx" and not use_docling:
            content = docx_to_json(file_path)
            if "error" in content:
                raise RuntimeError(content["error"])

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
                "DateReviewed": content.get("metadata", {}).get("DateReviewed"),
                "DateNext": content.get("metadata", {}).get("DateNext"),
                "ai_title": ai_title,
                "ai_description": ai_description,
            }
            complete_metadata = make_complete_metadata(file_path, extra_data)

            chunks = perform_chunking(full_text, complete_metadata, chunking_options) if do_chunking else []

            BACKGROUND_JOBS[job_id]["status"] = "done"
            BACKGROUND_JOBS[job_id]["finished_at"] = time.time()
            BACKGROUND_JOBS[job_id]["result"] = {
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
            }
            return

        # Docling path (PDFs + complex DOCX) with OpenAI descriptions
        # 1) Make sure local VLM is OFF and images are generated
        try:
            pdf_opts.do_picture_description = False
            pdf_opts.generate_picture_images = True
        except Exception:
            pass  # if options are immutable, rely on how you constructed docling_converter

        conversion_result = docling_converter.convert(file_path)
        document = conversion_result.document
        document_dict = document.export_to_dict()

        # 2) Gather image files Docling produced and call your OpenAI describer
        img_out_dir = os.path.join(IMAGE_OUTPUT_DIR, guid or "no_guid", "openai_img_desc")
        image_info_list = _gather_docling_picture_image_info(document, img_out_dir)
        logger.info(f"[job {job_id}] pictures found: {len(image_info_list)}")

        image_descriptions = get_image_descriptions_from_api(image_info_list) if image_info_list else []
        desc_texts = []
        for r in image_descriptions:
            d = r.get("description")
            # r['description'] can be a dict/structured for chat; ensure string
            if isinstance(d, str):
                desc_texts.append(d.strip())
            else:
                desc_texts.append(str(d) if d is not None else "Image")

        # 3) Inject descriptions into markdown placeholders
        full_text = _inject_descriptions_into_markdown(document, desc_texts)

        # 4) AI title/description and metadata
        ai_title, ai_description = generate_ai_title_description(full_text, os.path.basename(file_path))

        # origin meta from docling
        origin_meta_raw = document_dict.get("origin", {})
        origin_meta = origin_meta_raw if isinstance(origin_meta_raw, dict) else {"origin": str(origin_meta_raw)} if origin_meta_raw else {}

        extra_data = {
            "file_id": guid,
            "paragraph_count": full_text.count('\n\n') + 1,
            "table_count": len(document_dict.get("tables", [])),
            "word_count": len(full_text.split()),
            "character_count": len(full_text),
            "has_guid": bool(guid),
            "guid_source": guid_source,
            "ai_title": ai_title,
            "ai_description": ai_description,
            **origin_meta
        }
        complete_metadata = make_complete_metadata(file_path, extra_data)

        chunks = perform_chunking(full_text, complete_metadata, chunking_options) if do_chunking else []

        # 5) Store result
        BACKGROUND_JOBS[job_id]["status"] = "done"
        BACKGROUND_JOBS[job_id]["finished_at"] = time.time()
        BACKGROUND_JOBS[job_id]["result"] = {
            "success": True,
            "used_converter": "docling",
            "file_path": file_path,
            "filename": os.path.basename(file_path),
            "conversion_status": str(conversion_result.status),
            "document": document_dict,
            "full_text": full_text,
            "metadata": complete_metadata,
            "chunks": chunks
        }

    except Exception as e:
        logger.exception(f"[job {job_id}] docling conversion failed")
        BACKGROUND_JOBS[job_id]["status"] = "error"
        BACKGROUND_JOBS[job_id]["finished_at"] = time.time()
        BACKGROUND_JOBS[job_id]["result"] = {"success": False, "error": str(e)}
# --- API Endpoints ---

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

@app.route('/convert-file', methods=['POST'])
def convert_file():
    logger.info("Received request for /convert-file")
    data = request.get_json()
    file_path = data.get('file_path')
    force_refresh = data.get('force_refresh', False)

    if not file_path or not os.path.abspath(file_path).startswith(os.path.abspath(BASE_DIR)):
        return jsonify({"error": "Invalid or missing file_path"}), 400
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    cache_path = get_cache_path(file_path)

    try:
        if not force_refresh and is_cache_valid(file_path, cache_path):
            logger.info(f"Loading from cache: {cache_path}")
            with open(cache_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
        else:
            logger.info(f"Generating new content for: {file_path}")
            content = convert_file_to_json(file_path)
            if "error" in content:
                return jsonify({"error": content["error"]}), 500
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)

        # Assemble the final RAG text
        full_text_parts = []
        
        logger.info(f"Getting image descriptions for {content.get('filename')}...")
        images = content.get("images", [])
        if not isinstance(images, list):
            images = []
        image_descriptions = get_image_descriptions_from_api(images)

        images_by_page = {}
        for img_desc in image_descriptions:
            page_num = img_desc.get("page")
            if page_num not in images_by_page:
                images_by_page[page_num] = []
            images_by_page[page_num].append(f'[Image Description: {img_desc.get("description")}]')

        doc = fitz.open(file_path) if file_path.lower().endswith('.pdf') else None
        
        if doc:
            for page_num in range(1, doc.page_count + 1):
                page = doc.load_page(page_num - 1)
                full_text_parts.append(page.get_textpage().extractText())
                for table in content.get("extracted_tables", []):
                    if isinstance(table, dict) and table.get("page") == page_num:
                        full_text_parts.append(format_table_for_rag(table))
                if page_num in images_by_page:
                    full_text_parts.extend(images_by_page[page_num])
            doc.close()
        else:
            paragraphs = content.get("paragraphs", [])
            if isinstance(paragraphs, list):
                full_text_parts.extend(paragraphs)
            if content.get("extracted_tables"):
                for table in content.get("extracted_tables", []):
                    full_text_parts.append(format_table_for_rag(table))
            for page_num in sorted(images_by_page.keys()):
                 full_text_parts.extend(images_by_page[page_num])

        full_text_for_summary = "\n\n".join(full_text_parts)

        ai_title, ai_description = generate_ai_title_description(
            full_text_for_summary,
            content.get("filename", "document")
        )

        # Update metadata with AI fields
        metadata = content.get("metadata", {})
        metadata.update({
            "ai_title": ai_title,
            "ai_description": ai_description
        })

        return jsonify({
            "success": True,
            "file_info": {
                "filename": content.get("filename"),
                "subfolder": content.get("subfolder"),
                "file_id": content.get("file_id"),
                "full_path": content.get("full_path")
            },
            "full_text": full_text_for_summary,
            "structured_content": {
                "paragraphs": content.get("paragraphs"),
                "tables": content.get("tables"),
                "extracted_tables": content.get("extracted_tables"),
                "images": content.get("images")
            },
            "metadata": metadata  # All fields always included
        })
    except Exception as e:
        logger.error(f"An unexpected error occurred in /convert-file for {file_path}: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

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
# Configure picture description with a VLM
picture_desc_options = PictureDescriptionVlmOptions(
    repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
    prompt="Describe this document image concisely. If a chart/table/diagram, explain key points."
)

pdf_opts = PdfPipelineOptions(
    do_ocr=True,
    do_table_structure=True,
    table_structure_options=TableStructureOptions(mode=TableFormerMode.ACCURATE),
    picture_description_options=picture_desc_options
)
pdf_opts.do_picture_description = True
pdf_opts.generate_picture_images = True
pdf_opts.images_scale = 2.0

docling_converter = DocumentConverter(
    format_options={
        InputFormat.PDF:  PdfFormatOption(pipeline_options=pdf_opts),
        InputFormat.DOCX: WordFormatOption(),
    }
)


@app.route('/docling/convert-file', methods=['POST'])
def docling_convert_file():
    """
    Start a background conversion job that:
    - Runs Docling (or native DOCX for simple docs)
    - Disables local VLM and uses OpenAI image descriptions
    - Injects descriptions at <!-- image --> placeholders
    Returns: 202 with job_id and status URL to poll.
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    file_path = (data or {}).get("file_path")
    force_refresh = (data or {}).get("force_refresh", False)
    do_chunking = (data or {}).get("do_chunking", False)
    chunking_options = (data or {}).get("chunking_options", {})

    if not file_path:
        return jsonify({"error": "Missing 'file_path'"}), 400
    if not os.path.exists(file_path):
        return jsonify({"error": f"File not found: {file_path}"}), 404
    if not os.path.abspath(file_path).startswith(os.path.abspath(BASE_DIR)):
        return jsonify({"error": "File path is outside allowed directory"}), 403

    job_id = str(uuid.uuid4())
    BACKGROUND_JOBS[job_id] = {
        "status": "pending",
        "created_at": time.time(),
        "params": {
            "file_path": file_path,
            "force_refresh": force_refresh,
            "do_chunking": do_chunking,
            "chunking_options": chunking_options
        },
        "result": None
    }

    t = threading.Thread(
        target=_docling_convert_worker,
        args=(job_id, file_path, force_refresh, do_chunking, chunking_options),
        daemon=True
    )
    t.start()

    return jsonify({
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/docling/convert-status/{job_id}"
    }), 202

@app.route('/docling/convert-status/<job_id>', methods=['GET'])
def docling_convert_status(job_id):
    job = BACKGROUND_JOBS.get(job_id)
    if not job:
        return jsonify({"error": "job_id not found"}), 404
    return jsonify({
        "job_id": job_id,
        "status": job.get("status"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "result": job.get("result")
    }), 200
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
    print("Starting Flask app...")
    app.run(host='127.0.0.1', port=5001, debug=True)