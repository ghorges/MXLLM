import json
import time
import random
from typing import List, Dict
from parser import ContentParser
from browser_handler import BrowserHandler
from bs4 import BeautifulSoup

class WebScraper:
    """RSC网页抓取与解析器"""
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
            # 3. 获取页面内容
            page_source = self.browser.get_page_source()
            if not page_source:
                print(f"❌ 无法获取页面内容: {item['doi']}")
                return True
            # 4. 解析内容
            extracted_data = self.parser.parse_html(page_source, file_name=item.get("title", ""))
            item.update(extracted_data)

            # RSC特殊处理，检查是否有多页内容
            real_url = self.browser.get_current_url()
            if real_url:
                # 处理可能的分页内容，如果有"下一页"按钮
                try:
                    soup = BeautifulSoup(page_source, 'lxml')
                    next_page_links = soup.find_all('a', string=lambda s: s and ('next' in s.lower() or '下一页' in s))
                    
                    page_num = 2
                    while next_page_links and page_num < 10:  # 限制最多抓取10页
                        for link in next_page_links:
                            next_url = link.get('href')
                            if next_url:
                                if not next_url.startswith('http'):
                                    # 相对URL，需要补全
                                    base_url = '/'.join(real_url.split('/')[:3])  # 获取域名部分
                                    next_url = base_url + next_url if next_url.startswith('/') else base_url + '/' + next_url
                                
                                if self.browser.navigate_to_url(next_url):
                                    self.browser.simulate_human_scroll()
                                    next_page_source = self.browser.get_page_source()
                                    if next_page_source:
                                        next_soup = BeautifulSoup(next_page_source, 'lxml')
                                        # 提取当前页的内容并添加到已有内容中
                                        article_content = next_soup.find('div', id='pnlArticleContent')
                                        if article_content:
                                            for p in article_content.find_all('p'):
                                                paragraph_text = p.get_text().strip()
                                                if paragraph_text:
                                                    item['text'].append(paragraph_text)
                                            
                                            # 提取表格
                                            new_tables = self.parser.extract_tables(next_soup)
                                            if new_tables:
                                                if 'table' in item and isinstance(item['table'], list):
                                                    item['table'].extend(new_tables)
                                                else:
                                                    item['table'] = new_tables
                                        
                                        # 查找下一页的链接
                                        next_page_links = next_soup.find_all('a', string=lambda s: s and ('next' in s.lower() or '下一页' in s))
                                        page_num += 1
                                        # 随机等待3-8秒
                                        time.sleep(random.uniform(3, 8))
                                    else:
                                        next_page_links = []
                                else:
                                    next_page_links = []
                except Exception as e:
                    print(f"⚠️ 处理分页内容时出错: {e}")

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
        # 1. 筛选出所有RSC的条目
        rsc_items = [item for item in items if item.get("publisher") == "RSC"]
        print(f"find {len(rsc_items)} RSC items.")
        # 2. 加载已处理的DOI列表
        processed_dois = set()
        extracted_data = []
        extracted_path = "./Data/rsc_extracted.json"
        # 检查并加载已处理数据
        try:
            with open(extracted_path, "r", encoding="utf-8") as f:
                extracted_data = json.load(f)
                
            # 检查并清理text为空的条目
            filtered_data = [item for item in extracted_data if item.get("text") and len(item["text"]) > 0]
            if len(filtered_data) != len(extracted_data):
                print(f"发现text为空的条目，已自动清理 {len(extracted_data) - len(filtered_data)} 条。")
                with open(extracted_path, "w", encoding="utf-8") as f:
                    json.dump(filtered_data, f, ensure_ascii=False, indent=2)
                extracted_data = filtered_data
                
            for item in extracted_data:
                if "doi" in item and item["doi"]:
                    processed_dois.add(item["doi"])
        except FileNotFoundError:
            print("📋 rsc_extracted.json is not found, start from empty.")
            pass
        # 3. 筛选出未处理的条目
        unprocessed_items = []
        for item in rsc_items:
            if item.get("doi") not in processed_dois:
                unprocessed_items.append(item)
        total_processed = len(processed_dois)
        total_unprocessed = len(unprocessed_items)
        print(f"Statistics:")
        print(f"   - Total RSC items: {len(rsc_items)}")
        print(f"   - Processed items: {total_processed}")
        print(f"   - Unprocessed items: {total_unprocessed}")
        if total_unprocessed == 0:
            print("✅ All RSC items are processed.")
            return
        print(f"🚀 Start processing {total_unprocessed} unprocessed RSC items...")
        for i, item in enumerate(unprocessed_items, 1):
            print(f"\nProgress [{i}/{total_unprocessed}] - DOI: {item.get('doi', 'N/A')}")
            if not self.process_single_item(item):
                print(f"❌ Encounter serious error, stop processing. Processed {i-1}/{total_unprocessed} items.")
                break
            # 等待10-20秒
            time.sleep(random.uniform(10, 20))

    def save_results(self, 
                    final_data_file: str = "./Data/rsc_extracted.json",
                    bad_data_file: str = "./Data/rsc_bad.json"):
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