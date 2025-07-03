# Download Parsers Usage Guide

This directory contains parsers for downloading and extracting data from various publishers (ACS, Elsevier, RSC, Springer, Wiley). The core principle is to automate browser actions to fetch and parse publication data.

**Note:** Due to copyright and licensing reasons, only the compiled `.pyc` files for browser handlers are provided.

## How to Use

1. **Start Chrome in Debug Mode**

   Open a terminal and run:
   
   ```sh
   "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\chrome_debug"
   ```

2. **Run the Parser**

   Navigate to the desired publisher's folder (e.g., `ACS`, `Elsevier`, etc.) and run the `main.py` script:

   ```sh
   python main.py
   ```

This will launch the automated browser and begin the parsing process. 