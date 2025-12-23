import unittest
import json
import io
import zipfile
from docs_converter import app

class TestFilePathMetadata(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def create_dummy_docx(self):
        # Create a valid empty zip file acting as a docx
        b = io.BytesIO()
        with zipfile.ZipFile(b, 'w') as z:
            z.writestr('word/document.xml', '<root></root>')
        b.seek(0)
        return b

    def test_upload_with_metadata(self):
        dummy_docx = self.create_dummy_docx()
        
        # Define the metadata we want to verify
        target_path = "/original/path/to/file.docx"
        target_subfolder = "original_subfolder"
        
        data_payload = {
            "filename": "file.docx",
            "file_path": target_path,
            "subfolder": target_subfolder,
            "image_mode": "ask" # Use ask mode to return early
        }
        
        data = {
            'file': (dummy_docx, 'file.docx'),
            'data': json.dumps(data_payload)
        }
        
        response = self.client.post(
            '/docling/convert-file',
            data=data,
            content_type='multipart/form-data'
        )
        
        self.assertEqual(response.status_code, 200)
        json_response = response.get_json()
        
        print(f"\nResponse keys: {json_response.keys()}")
        if 'metadata' in json_response:
            print(f"Metadata: {json_response['metadata']}")
        
        # Check top-level file_path
        print(f"Top-level file_path: {json_response.get('file_path')}")
        
        # Verify failure (current behavior) or success (desired behavior)
        # Currently, file_path is None for uploads in 'ask' mode
        # And likely missing or temp path in metadata if we were to go further
        
        # The user wants file_path to be the original path
        # In 'ask' mode, the current code explicitly sets file_path to None if is_upload
        
import unittest
import json
import io
import zipfile
from docs_converter import app

class TestFilePathMetadata(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def create_dummy_docx(self):
        # Create a valid empty zip file acting as a docx
        b = io.BytesIO()
        with zipfile.ZipFile(b, 'w') as z:
            z.writestr('word/document.xml', '<root></root>')
        b.seek(0)
        return b

    def test_upload_with_metadata(self):
        dummy_docx = self.create_dummy_docx()
        
        # Define the metadata we want to verify
        target_path = "/original/path/to/file.docx"
        target_subfolder = "original_subfolder"
        
        data_payload = {
            "filename": "file.docx",
            "file_path": target_path,
            "subfolder": target_subfolder,
            "image_mode": "ask" # Use ask mode to return early
        }
        
        data = {
            'file': (dummy_docx, 'file.docx'),
            'data': json.dumps(data_payload)
        }
        
        response = self.client.post(
            '/docling/convert-file',
            data=data,
            content_type='multipart/form-data'
        )
        
        self.assertEqual(response.status_code, 200)
        json_response = response.get_json()
        
        print(f"\nResponse keys: {json_response.keys()}")
        if 'metadata' in json_response:
            print(f"Metadata: {json_response['metadata']}")
        
        # Check top-level file_path
        print(f"Top-level file_path: {json_response.get('file_path')}")
        
        # Verify failure (current behavior) or success (desired behavior)
        # Currently, file_path is None for uploads in 'ask' mode
        # And likely missing or temp path in metadata if we were to go further
        
        # The user wants file_path to be the original path
        # In 'ask' mode, the current code explicitly sets file_path to None if is_upload
        
        # We want to assert that it matches our target_path
        self.assertEqual(json_response.get('file_path'), target_path, "Top-level file_path should match provided path")
        
        # Also check metadata if it exists (ask mode might not return full metadata dict, let's check)
        # Ask mode returns: file_path, filename, lastModified, etc. flat.
        # It does NOT return a 'metadata' dict in 'ask' mode based on my reading.
        # It returns flat fields.
        
        # Wait, let's check the code for 'ask' mode return again.
        # Lines 922-938:
        # returns dict with keys: success, status, skipped, file_path, filename, ...
        
        # So for 'ask' mode, we check top-level file_path.
        self.assertEqual(json_response.get('file_path'), target_path, "Top-level file_path should match provided path")

    def test_upload_conversion_mode(self):
        """Test the full conversion path (mocking Docling) to verify metadata injection."""
        dummy_docx = self.create_dummy_docx()
        target_path = "/original/path/to/real_file.docx"
        target_subfolder = "real_subfolder"
        
        data_payload = {
            "filename": "real_file.docx",
            "file_path": target_path,
            "subfolder": target_subfolder,
            "image_mode": "force"  # Force conversion mode
        }
        
        data = {
            'file': (dummy_docx, 'real_file.docx'),
            'data': json.dumps(data_payload)
        }
        
        # Mock the docling_converter in docs_converter module
        from unittest.mock import MagicMock, patch
        
        # Create a mock result structure
        mock_doc = MagicMock()
        mock_doc.export_to_dict.return_value = {"tables": [], "metadata": {}}
        mock_doc.export_to_markdown.return_value = "Mocked markdown content"
        
        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_result.status = "success"
        
        with patch('docs_converter.docling_converter') as mock_converter:
            mock_converter.convert.return_value = mock_result
            
            response = self.client.post(
                '/docling/convert-file',
                data=data,
                content_type='multipart/form-data'
            )
            
            self.assertEqual(response.status_code, 200)
            json_response = response.get_json()
            
            # Verify top-level file_path
            self.assertEqual(json_response.get('file_path'), target_path, "Top-level file_path should match provided path in conversion mode")
            
            # Verify metadata
            metadata = json_response.get('metadata', {})
            self.assertEqual(metadata.get('full_path'), target_path, "Metadata full_path should match provided path")
            self.assertEqual(metadata.get('subfolder'), target_subfolder, "Metadata subfolder should match provided subfolder")
            
if __name__ == '__main__':
    unittest.main()
