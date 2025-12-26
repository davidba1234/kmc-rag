import requests
import json
import time

url = "http://localhost:5001/docling/convert-file"

def test():
    # 1. Upload a dummy PDF to get a valid file_id
    # We create a minimal valid PDF
    dummy_pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        b"2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n"
        b"3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Resources << >>\n>>\nendobj\n"
        b"xref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000060 00000 n\n0000000117 00000 n\n"
        b"trailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n223\n%%EOF"
    )
    
    files = {
        'file': ('test_dummy.pdf', dummy_pdf_content, 'application/pdf')
    }
    
    print("Uploading dummy file...")
    try:
        upload_response = requests.post(url, files=files, data={"data": json.dumps({"image_mode": "ask"})})
        print(f"Upload Response: {upload_response.status_code}")
        if upload_response.status_code != 200:
            print(f"Upload failed: {upload_response.text}")
            return
            
        file_data = upload_response.json()
        file_id = file_data.get("file_id")
        file_path = file_data.get("file_path")
        print(f"Got file_id: {file_id}")
        
        # 2. Test with INVALID token
        print("\nTesting invalid token...")
        
        # We must upload the file again, as per the real workflow
        files_2 = {
            'file': ('test_dummy.pdf', dummy_pdf_content, 'application/pdf')
        }
        
        data_2 = {
            "file_id": "new_random_id_will_be_generated", # This is ignored for uploads usually, or used if provided? Code generates new one.
            "image_mode": "selected",
            "selection_token": "definitely_invalid_token",
            "selected_image_ids": ["img1"],
            "filename": "test_dummy.pdf",
            "file_path": "/tmp/test_dummy.pdf" # Dummy path to satisfy logic
        }
        
        convert_response = requests.post(url, files=files_2, data={"data": json.dumps(data_2)})
        print(f"Convert Response Status: {convert_response.status_code}")
        
        resp_json = convert_response.json()
        print(json.dumps(resp_json, indent=2))
        
        if resp_json.get("error_type") == "INVALID_TOKEN":
            print("\nSUCCESS: Received expected INVALID_TOKEN error.")
        else:
            print("\nFAILURE: Did not receive expected INVALID_TOKEN error.")
            
    except Exception as e:
        print(f"Test failed with exception: {e}")

if __name__ == "__main__":
    test()
