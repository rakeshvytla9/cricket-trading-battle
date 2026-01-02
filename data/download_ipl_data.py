#!/usr/bin/env python3
"""
Download IPL ball-by-ball data from Cricsheet.org
Downloads all IPL matches in JSON format
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

# Configuration
CRICSHEET_IPL_URL = "https://cricsheet.org/downloads/ipl_json.zip"
DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"
ZIP_FILE = DATA_DIR / "ipl_json.zip"

def download_file(url: str, dest: Path) -> bool:
    """Download a file from URL to destination"""
    print(f"Downloading from {url}...")
    
    # Try with curl first (more reliable on macOS)
    import subprocess
    try:
        result = subprocess.run(
            ['curl', '-L', '-o', str(dest), url],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"Downloaded to {dest}")
            return True
        else:
            print(f"curl failed: {result.stderr}")
    except FileNotFoundError:
        pass  # curl not available, try urllib
    
    # Fallback: urllib with SSL context
    try:
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(url, context=ssl_context) as response:
            with open(dest, 'wb') as out_file:
                out_file.write(response.read())
        print(f"Downloaded to {dest}")
        return True
    except Exception as e:
        print(f"Error downloading: {e}")
        return False

def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract a zip file to destination directory"""
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"Error extracting: {e}")
        return False

def main():
    # Create directories
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download IPL data
    print("=" * 50)
    print("IPL Data Downloader - Cricsheet.org")
    print("=" * 50)
    
    if not download_file(CRICSHEET_IPL_URL, ZIP_FILE):
        sys.exit(1)
    
    # Extract
    if not extract_zip(ZIP_FILE, RAW_DIR):
        sys.exit(1)
    
    # Cleanup zip file
    ZIP_FILE.unlink()
    
    # Count files
    json_files = list(RAW_DIR.glob("*.json"))
    print(f"\n✅ Downloaded {len(json_files)} IPL match files")
    print(f"📁 Location: {RAW_DIR}")

if __name__ == "__main__":
    main()
