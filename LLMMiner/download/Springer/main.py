import json
import time
import random
from typing import List, Dict
from parser import ContentParser
from browser_handler import BrowserHandler
from bs4 import BeautifulSoup

class WebScraper:
    """Springer网页抓取与解析器"""
    def __init__(self, headless: bool = False):
        self.parser = ContentParser()
        self.browser = BrowserHandler(headless=headless)
        self.final_data = []
        self.bad_data = []

    def process_single_item(self, item: Dict) -> bool:
        tab_opened = False
        try:
            # 1. 导航到URL
            if not self.browser.navigate_to_url(item["doi"]):
                return False
            tab_opened = True
            # 2. 模拟人类行为
            self.browser.simulate_human_scroll()
            # self.browser.click_random_blank_area()
            # 3. 获取页面内容
            page_source = self.browser.get_page_source()
            if not page_source:
                print(f"❌ 无法获取页面内容: {item['doi']}")
                return True
            # 4. 解析内容
            extracted_data = self.parser.parse_html(page_source, file_name=item.get("title", ""))
            item.update(extracted_data)

            # 5. Springer特殊表格处理
            real_url = self.browser.get_current_url()
            if real_url:
                table_index = 1
                while True:
                    table_url = real_url.rstrip('/') + f'/tables/{table_index}'
                    if not self.browser.navigate_to_url(table_url):
                        break
                    table_page = self.browser.get_page_source()
                    if not table_page or '<main' not in table_page:
                        break
                    # 判断<main>是否有内容
                    soup = BeautifulSoup(table_page, 'lxml')
                    main = soup.find('main')
                    if not main or (not main.get_text(strip=True) and not main.find('table')):
                        break
                    extra_tables = self.parser.extract_tables_from_main(table_page)
                    if extra_tables:
                        if 'table' in item and isinstance(item['table'], list):
                            item['table'].extend(extra_tables)
                        else:
                            item['table'] = extra_tables
                    table_index += 1
            self.final_data.append(item)
            return True
        except Exception as e:
            print(f"❌ 处理失败: {item['doi']}，原因: {e}")
            return False
        finally:
            if tab_opened:
                try:
                    self.browser.close_current_tab()
                except Exception:
                    pass

    def process_items(self, items: List[Dict]) -> None:
        # 1. 筛选出所有Springer的条目
        springer_items = [item for item in items if item.get("publisher") == "Springer"]
        print(f"find {len(springer_items)} Springer items.")
        # 2. 加载已处理的DOI列表
        processed_dois = set()
        extracted_data = []
        extracted_path = "./Data/springer_extracted.json"
        # 检查并清理text为空的条目
        try:
            with open(extracted_path, "r", encoding="utf-8") as f:
                extracted_data = json.load(f)
            # # 过滤掉text为空的条目
            # filtered_data = [item for item in extracted_data if item.get("text") and len(item["text"]) > 0]
            # if len(filtered_data) != len(extracted_data):
            #     print(f"发现text为空的条目，已自动清理 {len(extracted_data) - len(filtered_data)} 条。")
            #     with open(extracted_path, "w", encoding="utf-8") as f:
            #         json.dump(filtered_data, f, ensure_ascii=False, indent=2)
            #     extracted_data = filtered_data
            for item in extracted_data:
                if "doi" in item and item["doi"]:
                    processed_dois.add(item["doi"])
        except FileNotFoundError:
            print("📋 springer_extracted.json is not found, start from empty.")
            pass
        # 3. 筛选出未处理的条目
        unprocessed_items = []
        for item in springer_items:
            if item.get("doi") not in processed_dois:
                unprocessed_items.append(item)
        total_processed = len(processed_dois)
        total_unprocessed = len(unprocessed_items)
        print(f"Statistics:")
        print(f"   - Total Springer items: {len(springer_items)}")
        print(f"   - Processed items: {total_processed}")
        print(f"   - Unprocessed items: {total_unprocessed}")
        if total_unprocessed == 0:
            print("✅ All Springer items are processed.")
            return
        print(f"🚀 Start processing {total_unprocessed} unprocessed Springer items...")
        for i, item in enumerate(unprocessed_items, 1):
            print(f"\nProgress [{i}/{total_unprocessed}] - DOI: {item.get('doi', 'N/A')}")
            if not self.process_single_item(item):
                print(f"❌ Encounter serious error, stop processing. Processed {i-1}/{total_unprocessed} items.")
                break
            # 等待10-20秒
            time.sleep(random.uniform(10, 20))

    def save_results(self, 
                    final_data_file: str = "./Data/springer_extracted.json",
                    bad_data_file: str = "./Data/springer_bad.json"):
        try:
            with open(final_data_file, "r", encoding="utf-8") as f:
                original_data = json.load(f)
        except FileNotFoundError:
            original_data = []
        original_data.extend(self.final_data)
        with open(final_data_file, "w", encoding="utf-8") as f:
            json.dump(original_data, f, ensure_ascii=False, indent=2)
        try:
            with open(bad_data_file, "r", encoding="utf-8") as f:
                original_bad_data = json.load(f)
        except FileNotFoundError:
            original_bad_data = []
        original_bad_data.extend(self.bad_data)
        with open(bad_data_file, "w", encoding="utf-8") as f:
            json.dump(original_bad_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Process completed: {len(self.final_data)} items processed, {len(self.bad_data)} items skipped.")

    def close(self):
        self.browser.close()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def main():
    with open("./Databases/publisher_doi.json", "r", encoding="utf-8") as f:
        input_json = json.load(f)
    with WebScraper(headless=False) as scraper:
        scraper.process_items(input_json)
        scraper.save_results()

if __name__ == "__main__":
    main() 