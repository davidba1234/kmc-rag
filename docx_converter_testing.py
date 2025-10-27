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
import pikepdf # NEW: Added for writing PDF metadata
import uuid    # NEW: Added for generating GUIDs
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

# --- Basic Configuration ---
# Configure logging
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

SUPPORTED_EXTENSIONS = ['.docx', '.pdf']

# --- Hardcoded list of all metadata fields to be returned ---
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
    "DateNext"
]

# --- Helper Functions ---

def filter_metadata(metadata, requested_fields=None):
    fields_to_include = ALL_METADATA_FIELDS if requested_fields is None or requested_fields == "default" else requested_fields
    if fields_to_include == "all":
        return metadata
    return {field: metadata[field] for field in fields_to_include if field in metadata}

def make_common_metadata(file_path, extra=None):
    """Return common metadata dictionary for both converters."""
    file_stats = os.stat(file_path)
    parent_dir = os.path.basename(os.path.dirname(file_path))
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

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

    if extra and isinstance(extra, dict):
        metadata.update(extra)

    return metadata

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

# --- NEW: Function to get or create a persistent GUID in a PDF file ---
def get_or_create_pdf_guid(pdf_path: str) -> str:
    """
    Reads a custom GUID from a PDF's metadata. If not present, it generates one,
    writes it to the PDF's metadata, and saves the file.
    """
    guid_key = '/AppGUID'  # Custom metadata key for our GUID
    try:
        with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
            docinfo = pdf.docinfo
            if guid_key in docinfo:
                logger.info(f"Found existing GUID in {pdf_path}")
                return str(docinfo[guid_key])

            logger.info(f"No GUID found in {pdf_path}. Generating a new one.")
            new_guid = str(uuid.uuid4())
            docinfo[guid_key] = new_guid
            pdf.save() # Saves the changes to the original file
            logger.info(f"Saved new GUID to {pdf_path}")
            return new_guid
            
    except Exception as e:
        logger.error(f"Pikepdf failed to read or write GUID for {pdf_path}: {e}", exc_info=True)
        # Fallback to filename-based ID if pikepdf fails
        filename_without_ext = os.path.splitext(os.path.basename(pdf_path))[0]
        return filename_without_ext.replace(' ', '_').replace('-', '_').lower()

# --- NEW: Read-only function to get a GUID from a PDF for listing purposes ---
def read_pdf_guid(pdf_path: str) -> str | None:
    """
    Reads a custom GUID from a PDF's metadata without modifying the file.
    Returns None if the GUID is not found.
    """
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



# Heuristic: decide if DOCX is complex enough for docling
def is_docx_complex(docx_path,
                    min_tables_for_complex=1,
                    min_images_for_complex=1,
                    max_paragraphs_for_simple=300,
                    avg_run_threshold=2.5):
    """
    Returns True if the DOCX should be processed by Docling (complex),
    False if we should use your lightweight python-docx extractor (simple).
    Tune the thresholds as needed.
    """
    try:
        # Fast checks via zip (counts media files and presence of embedded objects)
        with zipfile.ZipFile(docx_path, 'r') as z:
            namelist = z.namelist()
            media_files = [n for n in namelist if n.startswith('word/media/')]
            embedded_objects = [n for n in namelist if n.startswith('word/embeddings') or n.endswith('.bin')]
            has_images = len(media_files) >= min_images_for_complex
            has_embedded = len(embedded_objects) > 0

        # Use python-docx for tables and paragraph/run complexity
        doc = Document(docx_path)
        num_tables = len(doc.tables)
        num_paragraphs = len(doc.paragraphs)

        # measure average runs per paragraph as a crude measure of formatting complexity
        total_runs = 0
        paragraphs_sampled = 0
        for p in doc.paragraphs:
            # skip empty paragraphs for run averaging
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

        # Decide complexity:
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
        # Conservative approach: if we fail to inspect, use docling to be safe
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
                        # Clean table data
                        clean_table = []
                        for row in table:
                            clean_row = [cell.strip() if cell else "" for cell in row]
                            if any(clean_row):  # Skip empty rows
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
            
            # Add separator after header row
            if row_index == 0:
                formatted_lines.append("-" * 40)
    
    formatted_lines.append("--- TABLE END ---\n")
    return "\n".join(formatted_lines)

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

        file_stats = os.stat(docx_path)
        parent_dir = os.path.basename(os.path.dirname(docx_path))
        date_reviewed, date_next = extract_custom_metadata(docx_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        
        metadata = {
            "file_id": file_id,
            "filename": os.path.basename(docx_path),
            "full_path": docx_path,
            "subfolder": parent_dir,
            "paragraph_count": len(paragraphs),
            "table_count": len(doc.tables),
            "file_size": file_stats.st_size,
            "file_size_mb": round(file_stats.st_size / (1024 * 1024), 2),
            "file_extension": ".docx",
            "last_modified_iso": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            "created_date_iso": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            "word_count": sum(len(p.split()) for p in paragraphs),
            "character_count": sum(len(p) for p in paragraphs),
            "has_guid": has_guid,
            "guid_source": "docx_settings" if has_guid else "filename_fallback",
            "DateReviewed": date_reviewed,
            "DateNext": date_next
        }

        content = {
            "file_id": file_id,
            "filename": os.path.basename(docx_path),
            "full_path": docx_path,
            "subfolder": parent_dir,
            "paragraphs": paragraphs,
            "tables": [[cell.text.strip() for cell in row.cells] for table in doc.tables for row in table.rows],
            "metadata": metadata
        }
        return content
    except Exception as e:
        logger.error(f"Failed to convert {docx_path}: {e}", exc_info=True)
        return {"error": f"Failed to convert {docx_path}: {str(e)}"}

# --- CHANGED: pdf_to_json now uses the new GUID function ---
def pdf_to_json(pdf_path):
    logger.info(f"Starting PDF conversion for file: {pdf_path}")
    try:
        # Get or create GUID
        file_id = get_or_create_pdf_guid(pdf_path) # Use this as a unique folder name for images
        
        # --- IMAGE EXTRACTION ---
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
        
        # Extract text using PyMuPDF (compatible with older PyMuPDF versions)
        full_text = "".join(page.get_textpage().extractText() for page in doc)
        paragraphs = [p.strip() for p in full_text.split('\n') if p.strip()]
        
        # Convert tables to the format expected by your chunking code
        tables_for_chunking = []
        for table in extracted_tables:
            if table.get("raw_data"):
                tables_for_chunking.extend(table["raw_data"])

        # Get file stats
        file_stats = os.stat(pdf_path)
        parent_dir = os.path.basename(os.path.dirname(pdf_path))

        metadata = {
            "file_id": file_id,
            "filename": os.path.basename(pdf_path),
            "full_path": pdf_path,
            "subfolder": parent_dir,
            "paragraph_count": len(paragraphs),
            "table_count": len(extracted_tables),  # Now actually counts tables
            "image_count": len(image_info),
            "file_size": file_stats.st_size,
            "file_size_mb": round(file_stats.st_size / (1024 * 1024), 2),
            "file_extension": ".pdf",
            "last_modified_iso": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            "created_date_iso": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            "word_count": len(full_text.split()),
            "character_count": len(full_text),
            "has_guid": True,
            "guid_source": "pdf_metadata",
            "DateReviewed": None,
            "DateNext": None
        }
        
        content = {
            "file_id": file_id,
            "filename": os.path.basename(pdf_path),
            "full_path": pdf_path,
            "subfolder": parent_dir,
            "paragraphs": paragraphs,
            "tables": tables_for_chunking,  # Now contains actual table data
            "images": image_info, # List of extracted image paths and info
            "extracted_tables": extracted_tables,  # Structured table data
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
    # If the source file was modified, its mtime will be newer than the cache file's.
    if not os.path.exists(cache_path):
        return False
    return os.path.getmtime(cache_path) > os.path.getmtime(file_path)

# In your Flask app
def create_table_summary(table_data):
    """Generate a descriptive summary of the table"""
    if not table_data or not table_data.get("headers"):
        return ""
    
    headers = table_data["headers"]
    row_count = len(table_data.get("rows", []))
    
    summary = f"Table with {row_count} rows containing: {', '.join(headers)}"
    return summary

# Update metadata
#metadata.update({
 #   "table_count": len(all_tables),
 #   "table_summaries": [create_table_summary(t) for t in all_tables],
 #   "has_tables": len(all_tables) > 0
#})

def get_image_descriptions_from_api(image_info_list: list) -> list:
    """
    Takes a list of image paths, calls a vision model for each,
    and returns a list of descriptions.
    """
    # API key is read automatically from the OPENAI_API_KEY environment variable
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

# --- API Endpoints ---
# --- ADD THIS NEW ENDPOINT TO YOUR SCRIPT ---
@app.route('/ensure-guids', methods=['POST'])
def ensure_guids():
    """
    Scans all supported files and ensures that any file missing a GUID gets one.
    This is a write operation and should be called before /list-files.
    """
    logger.info("Received request for /ensure-guids. Scanning all files.")
    try:
        all_files = find_all_supported_files()
        files_updated = 0
        
        for file_info in all_files:
            path = file_info['full_path']
            guid_found = False

            if path.lower().endswith('.docx'):
                # DOCX GUIDs are read-only from the file, we can't write them.
                if extract_docx_guid(path):
                    guid_found = True

            elif path.lower().endswith('.pdf'):
                # For PDFs, check if a GUID exists. If not, get_or_create_pdf_guid will create it.
                existing_guid = read_pdf_guid(path)
                if existing_guid:
                    guid_found = True
                else:
                    # The file has no GUID, so we create one.
                    # This function modifies the file.
                    get_or_create_pdf_guid(path)
                    files_updated += 1
                    guid_found = True # It has one now

        message = f"Scan complete. Updated {files_updated} PDF file(s) with a new GUID."
        logger.info(message)
        return jsonify({"success": True, "message": message, "files_updated": files_updated})

    except Exception as e:
        logger.error(f"An error occurred in /ensure-guids: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    
# --- CHANGED: /list-files now reads GUIDs from both DOCX and PDF files ---
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
                # Check for GUID based on file type
                if path.lower().endswith('.docx'):
                    file_id = extract_docx_guid(path)
                elif path.lower().endswith('.pdf'):
                    # Use the new read-only function for PDFs
                    file_id = read_pdf_guid(path)

                # Fallback Identifier
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
    requested_fields = data.get('metadata_fields')
    force_refresh = data.get('force_refresh', False)

    if not file_path or not os.path.abspath(file_path).startswith(os.path.abspath(BASE_DIR)):
        return jsonify({"error": "Invalid or missing file_path"}), 400
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    cache_path = get_cache_path(file_path)

    try:
        # Note: When a PDF's GUID is written, its mtime is updated,
        # which will correctly invalidate the cache.
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

        # --- Assemble the final RAG text here ---
        full_text_parts = []
        
        # NEW: Get image descriptions directly from the API
        logger.info(f"Getting image descriptions for {content.get('filename')}...")
        images = content.get("images")
        if not isinstance(images, list):
            images = []
        image_descriptions = get_image_descriptions_from_api(images)

        # Create a map of image descriptions by page for easy lookup
        images_by_page = {}
        for img_desc in image_descriptions:
            page_num = img_desc.get("page")
            if page_num not in images_by_page:
                images_by_page[page_num] = []
            images_by_page[page_num].append(f'[Image Description: {img_desc.get("description")}]')

        # Interleave paragraphs, tables, and image descriptions page by page
        # This logic is safe for both DOCX and PDF because docx will just have 1 page.
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
        else: # Fallback for DOCX or if PDF handling fails to open doc
            paragraphs = content.get("paragraphs")
            if isinstance(paragraphs, list):
                full_text_parts.extend(paragraphs)
            if content.get("extracted_tables"):
                for table in content.get("extracted_tables", []):
                    full_text_parts.append(format_table_for_rag(table))
            # For DOCX, just append all image descriptions at the end
            for page_num in sorted(images_by_page.keys()):
                 full_text_parts.extend(images_by_page[page_num])

        full_text_for_summary = "\n\n".join(full_text_parts)

        filtered_metadata = filter_metadata(content.get("metadata", {}), requested_fields)
            
            # Return a response that serves both the AI summarizer and the smart chunker
        return jsonify({
                "success": True,
                "file_info": {
                "filename": content.get("filename"),
                "subfolder": content.get("subfolder"),
                "file_id": content.get("file_id"),
                "full_path": content.get("full_path")
            },
            "full_text": full_text_for_summary, # Now includes image descriptions
            "structured_content": {
                "paragraphs": content.get("paragraphs"),
                "tables": content.get("tables"),
                "extracted_tables": content.get("extracted_tables"),
                "images": content.get("images")
            },
            "metadata": filtered_metadata
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

pdf_opts = PdfPipelineOptions(
    do_ocr=True,                      # enable OCR on scanned PDFs
    do_table_structure=True,          # run table structure model
    table_structure_options=TableStructureOptions(
        mode=TableFormerMode.ACCURATE # FAST or ACCURATE
    )
)


# Initialize Docling converter with desired format options
docling_converter = DocumentConverter(
    format_options={
        InputFormat.PDF:  PdfFormatOption(pipeline_options=pdf_opts),
        InputFormat.DOCX: WordFormatOption(),   # defaults are fine
    }
)

# ... (Your existing imports remain the same)

# NEW: Imports for Chonkie SemanticChunker


# ... (Your existing code up to the /docling/convert-file endpoint remains the same)

@app.route('/docling/convert-file', methods=['POST'])
def docling_convert_file():
    try:
        data = request.get_json(force=True)
        if not data or 'file_path' not in data:
            return jsonify({"error": "Missing 'file_path' in JSON request body"}), 400
        file_path = data['file_path']
        force_refresh = data.get("force_refresh", False)
        requested_fields = data.get("metadata_fields", None)
        do_chunking = data.get("do_chunking", False)  # NEW: Option to trigger chunking
        chunking_options = data.get("chunking_options", {})  # e.g., {"threshold": 0.75}

        # Basic validations (existing)
        if not os.path.exists(file_path):
            return jsonify({"error": f"File not found: {file_path}"}), 404
        if not os.path.abspath(file_path).startswith(os.path.abspath(BASE_DIR)):
            return jsonify({"error": "File path is outside allowed directory"}), 403

        ext = os.path.splitext(file_path)[1].lower()

        # NEW: Fetch GUID early using your functions
        guid = None
        guid_source = "unknown"
        if ext == '.pdf':
            guid = get_or_create_pdf_guid(file_path)  # Creates if missing, as per your function
            guid_source = "pdf_metadata"
        elif ext == '.docx':
            guid = extract_docx_guid(file_path)
            if guid:
                guid_source = "docx_settings"
            else:
                # Fallback to filename-based (as in your code)
                filename_without_ext = os.path.splitext(os.path.basename(file_path))[0]
                guid = filename_without_ext.replace(' ', '_').replace('-', '_').lower()
                guid_source = "filename_fallback"

        # Decide converter (existing logic)
        use_docling = False
        if ext == '.pdf':
            use_docling = True
        elif ext == '.docx':
            use_docling = is_docx_complex(file_path)
        else:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400

        logger.info(f"convert-file: {file_path} selected converter = {'docling' if use_docling else 'native'}")

        # ---------- Native lightweight docx path ----------
        if ext == '.docx' and not use_docling:
            content = docx_to_json(file_path)
            if "error" in content:
                return jsonify({"error": content["error"]}), 500

            date_reviewed, date_next = extract_custom_metadata(file_path)
            metadata_extra = {
                "has_guid": bool(guid),  # Updated to use your GUID
                "guid_source": guid_source,
                "DateReviewed": date_reviewed,
                "DateNext": date_next
            }
            common_meta = make_common_metadata(file_path, metadata_extra)
            common_meta["file_id"] = guid  # Attach your GUID here

            # Filter metadata (existing)
            metadata_dict = content.get("metadata", {})
            if not isinstance(metadata_dict, dict):
                metadata_dict = {}
            filtered_meta = filter_metadata({**metadata_dict, **common_meta}, requested_fields)

            full_text = "\n\n".join(content.get("paragraphs", []))

            # NEW: Chunking if requested
            chunks = []
            if do_chunking:
                chunks = perform_chunking(full_text, common_meta, chunking_options)  # See helper function below

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
                "metadata": filtered_meta,
                "chunks": chunks  # Enriched with GUID
            })

        # ---------- Docling path (for PDFs and complex DOCX) ----------
        logger.info("Running Docling converter...")
        conversion_result = docling_converter.convert(file_path)
        document = conversion_result.document
        document_dict = document.export_to_dict()

        # Custom metadata (existing, but attach your GUID)
        extra_meta = {}
        if ext == '.docx':
            date_reviewed, date_next = extract_custom_metadata(file_path)
            extra_meta.update({"DateReviewed": date_reviewed, "DateNext": date_next})
        extra_meta.update({"file_id": guid, "has_guid": bool(guid), "guid_source": guid_source})

        common_meta = make_common_metadata(file_path, extra_meta)

        # Merge and filter metadata (existing)
        origin_meta = document_dict.get("origin", {}) or {}
        merged_meta = {**common_meta, **origin_meta}
        filtered_meta = filter_metadata(merged_meta, requested_fields)

        # NEW: Export full_text from Docling (for chunking)
        full_text = document.export_to_markdown()  # Or export_to_text() for plain

        # NEW: Chunking if requested
        chunks = []
        if do_chunking:
            chunks = perform_chunking(full_text, merged_meta, chunking_options)  # See helper function below

        return jsonify({
            "success": True,
            "used_converter": "docling",
            "file_path": file_path,
            "filename": os.path.basename(file_path),
            "conversion_status": str(conversion_result.status),
            "document": document_dict,
            "full_text": full_text,
            "metadata": filtered_meta,
            "chunks": chunks  # Enriched with GUID
        })

    except Exception as e:
        logger.error(f"Error processing file with Docling/native: {e}", exc_info=True)
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

# NEW: Helper function for chunking and enrichment (call from both paths)
# In docx_converter_testing.py (update the perform_chunking function)
def perform_chunking(input_text, meta, options):
    """Performs semantic chunking, embedding, and enriches with metadata including your GUID."""
    try:
        # Initialize SemanticChunker with user-provided options
        chunker = SemanticChunker(
            embedding_model=options.get("embedding_model", "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"),
            threshold=options.get("threshold", 0.75),
            chunk_size=options.get("chunk_size", 512),
            device_map="cpu"  # Or "cuda" if GPU available
        )
        raw_chunks = chunker.chunk(input_text)

        # Load embedding model (do this here or globally to avoid repeated loads)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(options.get("embedding_model", "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"))

        # Enrich and embed each chunk
        enriched_chunks = []
        for chunk in raw_chunks:
            enriched = {
                "text": chunk.text,
                "start_index": chunk.start_index,
                "end_index": chunk.end_index,
                "token_count": chunk.token_count,
                "metadata": {
                    "file_id": meta.get("file_id"),  # Your GUID
                    "filename": meta.get("filename"),
                    "subfolder": meta.get("subfolder"),
                    "DateReviewed": meta.get("DateReviewed"),
                    "DateNext": meta.get("DateNext"),
                    "last_modified_iso": meta.get("last_modified_iso"),
                    # Add more as needed
                }
            }
            enriched_chunks.append(enriched)
        return enriched_chunks
    except Exception as e:
        logger.error(f"Chunking/embedding failed: {e}")
        return []

# ... (Your existing /docling/convert-all endpoint and app.run() remain the same)

@app.route('/docling/convert-all', methods=['POST'])
def docling_convert_all():
    """
    Batch-convert all supported files under BASE_DIR using the Docling path.
    Returns a list of results with status per file.
    """
    results = []
    files = find_all_supported_files()
    for f in files:
        path = f.get("full_path")
        try:
            # decide converter like your existing logic (here we force Docling)
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
