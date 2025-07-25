from fastmcp import FastMCP
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime

# Initialize FastMCP
mcp = FastMCP("mxllm-download")

# File path configuration
DATA_DIR = Path("../../Data")
DATABASES_DIR = Path("../../Databases")

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
DATABASES_DIR.mkdir(exist_ok=True)

# Fixed filename for custom text data
CUSTOM_TEXT_FILENAME = "custom_extracted.json"

@mcp.tool()
def process_custom_text(content: str, title: Optional[str] = None, authors: Optional[List[str]] = None, 
                      doi: Optional[str] = None, year: Optional[int] = None, 
                      journal: Optional[str] = None, abstract: Optional[str] = None, 
                      keywords: Optional[List[str]] = None) -> str:
    """
    Process user-provided custom text and save in the same format as acs_extracted.json
    
    Args:
        content: Text content
        title: Optional title
        authors: Optional list of authors
        doi: Optional DOI
        year: Optional publication year
        journal: Optional journal name
        abstract: Optional abstract
        keywords: Optional list of keywords
        
    Returns:
        JSON string with processing results
    """
    try:
        # Validate required fields
        if not content:
            return json.dumps({
                "status": "error", 
                "message": "Content field is required"
            })
            
        if not title:
            return json.dumps({
                "status": "error", 
                "message": "Title field is required"
            })
        
        # Format data in the same structure as acs_extracted.json
        new_entry = {
            "doi": doi or "custom",
            "title": title,
            "authors": authors or [],
            "journal": journal or "Custom Source",
            "year": year or 0,
            "abstract": abstract or "",
            "keywords": keywords or [],
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        # Fixed file path
        file_path = DATA_DIR / CUSTOM_TEXT_FILENAME
        
        # Read existing data if file exists
        existing_data = []
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = []
            except json.JSONDecodeError:
                # If file is corrupted, start fresh
                existing_data = []
        
        # Append new data
        existing_data.append(new_entry)
        
        # Write back to file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2)
            
        return json.dumps({
            "status": "success", 
            "message": f"Custom text processed and appended to {CUSTOM_TEXT_FILENAME}",
            "file_path": str(file_path),
            "entries_count": len(existing_data)
        })
        
    except Exception as e:
        return json.dumps({
            "status": "error", 
            "message": f"Failed to process custom text: {str(e)}"
        })

@mcp.tool()
def update_scopus(message: str) -> str:
    """
    Update Scopus data by processing the scopus.csv file
    
    Args:
        message: User update request message
        
    Returns:
        JSON string with processing results
    """
    # Check if message is related to scopus update
    if "scopus" not in message.lower() and "update" not in message.lower():
        return json.dumps({
            "status": "error", 
            "message": "Invalid request message. Must mention 'scopus' and 'update'."
        })
        
    # Check if scopus.csv exists
    scopus_path = DATABASES_DIR / "scopus.csv"
    if not scopus_path.exists():
        return json.dumps({
            "status": "error", 
            "message": f"scopus.csv not found in {str(DATABASES_DIR)}"
        })
        
    # Call process_doi.py
    result = run_process_doi()
    
    return json.dumps({
        "status": "success",
        "message": "Scopus update process completed",
        "result": result
    })

def run_process_doi() -> str:
    """
    Run the process_doi.py script to update publisher_doi.json
    
    Returns:
        Processing result message
    """
    try:
        script_path = DATABASES_DIR / "process_doi.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"process_doi.py output: {result.stdout}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running process_doi.py: {e.stderr}")
        return f"Error: {e.stderr}"

@mcp.tool()
def health_check() -> str:
    """
    Health check endpoint
    
    Returns:
        Service status information
    """
    return json.dumps({
        "status": "OK", 
        "service": "MXLLM Download MCP Service"
    })

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8888) 