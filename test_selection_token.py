import sys
import unittest
import json
import io
import time
from unittest.mock import MagicMock, patch

# Mock heavy dependencies before importing docs_converter
mock_docling = MagicMock()
sys.modules["docling"] = mock_docling
sys.modules["docling.document_converter"] = mock_docling
sys.modules["docling.datamodel.pipeline_options"] = mock_docling
sys.modules["docling.datamodel.base_models"] = mock_docling
sys.modules["chonkie"] = MagicMock() # Also mock chonkie if it's heavy

mock_fitz = MagicMock()
sys.modules["fitz"] = mock_fitz

from docs_converter import app, save_selection_state, load_selection_state

class TestSelectionToken(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True
        
    def test_selection_token_flow(self):
        """
        Test the flow:
        1. Upload PDF with image_mode="ask" -> returns selection_token
        2. Upload same PDF with image_mode="selected" + token -> should succeed
        """
        
        # Configure global mock_fitz
        mock_fitz = sys.modules["fitz"]
        
        # Setup mock document with images
        mock_doc = MagicMock()
        mock_page = MagicMock()
        # return one image with size 200x200 (area 40000 > 12000)
        mock_page.get_images.return_value = [(1, 0, 200, 200, 8, "DeviceRGB", "", "img1", "DCTDecode")] 
        mock_page.get_text.return_value = "some text"
        mock_doc.__iter__.return_value = [mock_page]
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        
        # For extract_image in build_pdf_image_manifest
        mock_doc.extract_image.return_value = {"image": b"fake_image_bytes", "ext": "png"}
        
        # Mock context manager behavior
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        
        mock_fitz.open.return_value = mock_doc
        
        # --- Step 1: "ask" mode ---
        data_payload_ask = {
            "filename": "test.pdf",
            "file_path": "original/path/test.pdf",
            "image_mode": "ask"
        }
        
        data_ask = {
            'file': (io.BytesIO(b"%PDF-1.4..."), 'test.pdf'),
            'data': json.dumps(data_payload_ask)
        }
        
        print("--- Sending ASK request ---")
        resp_ask = self.client.post(
            '/docling/convert-file',
            data=data_ask,
            content_type='multipart/form-data'
        )
        
        self.assertEqual(resp_ask.status_code, 200)
        json_ask = resp_ask.get_json()
        print(f"ASK Response: {json_ask}")
        
        token = json_ask.get("selection_token")
        self.assertIsNotNone(token, "Should return a selection_token")
        
        # Verify state was saved
        state = load_selection_state(token)
        print(f"Saved State: {state}")
        
        # --- Step 2: "selected" mode ---
        # We simulate a NEW upload, so the temp file path will be different.
        # But we send the SAME file_path in JSON.
        
        data_payload_sel = {
            "filename": "test.pdf",
            "file_path": "original/path/test.pdf",
            "image_mode": "selected",
            "selection_token": token,
            "selected_image_ids": [json_ask["image_manifest"][0]["id"]]
        }
        
        data_sel = {
            'file': (io.BytesIO(b"%PDF-1.4..."), 'test.pdf'),
            'data': json.dumps(data_payload_sel)
        }
        
        # We also need to mock docling_converter for the second step
        # Since we mocked docling globally, we can just configure the mock
        mock_docling = sys.modules["docling"]
        # docling_converter is an instance of DocumentConverter
        # In docs_converter.py: docling_converter = DocumentConverter(...)
        # So sys.modules["docling.document_converter"].DocumentConverter() returns the instance
        
        mock_converter_instance = sys.modules["docling.document_converter"].DocumentConverter.return_value
        mock_result = MagicMock()
        mock_result.document.export_to_dict.return_value = {}
        mock_result.document.export_to_markdown.return_value = "markdown"
        mock_result.status = "success"
        mock_converter_instance.convert.return_value = mock_result
        
        print("--- Sending SELECTED request ---")
        resp_sel = self.client.post(
            '/docling/convert-file',
            data=data_sel,
            content_type='multipart/form-data'
        )
        
        json_sel = resp_sel.get_json()
        print(f"SELECTED Response: {json_sel}")
        
        # If the bug exists, this should fail with INVALID_TOKEN or similar
        # because the temp path in state != temp path in second request
        
        if json_sel.get("error"):
            print(f"FAILED with error: {json_sel.get('error_message')}")
        
        self.assertTrue(json_sel.get("success"), f"Request failed: {json_sel.get('error_message')}")

if __name__ == '__main__':
    unittest.main()
