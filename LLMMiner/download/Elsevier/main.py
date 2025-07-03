import json
from typing import List, Dict
from parser import ContentParser
from browser_handler import BrowserHandler

class WebScraper:
    """主要的网页抓取器"""
    
    def __init__(self, headless: bool = False):
        self.parser = ContentParser()
        self.browser = BrowserHandler(headless=headless)
        self.final_data = []
        self.bad_data = []
    
    def process_single_item(self, item: Dict) -> bool:
        """处理单个项目"""
        try:
            # 1. 导航到URL
            if not self.browser.navigate_to_url(item["doi"]):
                return False
            
            # 2. 模拟人类行为
            self.browser.simulate_human_scroll()
            self.browser.click_random_blank_area()

            # 3. 检查URL重定向
            redirect_info = self.browser.check_url_redirect(item["doi"])
            if redirect_info["redirected"]:
                if redirect_info["issue"] == "ip_problem":
                    print(f"❌ IP问题: {self.browser.get_current_url()}")
                    return False  # 严重问题，停止处理
                else:
                    print(f"⚠️ {redirect_info['issue']}: {self.browser.get_current_url()}")
                    item["skip_reason"] = redirect_info["issue"]
                    self.bad_data.append(item)
                    return True  # 继续处理下一个
            
            
            # 4. 获取页面内容
            page_source = self.browser.get_page_source()
            if not page_source:
                print(f"❌ 无法获取页面内容: {item['doi']}")
                return True
            
            # 5. 验证页面内容
            validation = self.parser.validate_page_content(page_source)
            if not validation["is_valid"]:
                print(f"⚠️ Skip: {validation['skip_reason']} -> {item['title']}")
                item["skip_reason"] = validation["skip_reason"]
                self.bad_data.append(item)
                return True
            
            # 6. 解析内容
            extracted_data = self.parser.parse_html(page_source)
            item.update(extracted_data)
            self.final_data.append(item)
            
            return True
            
        except Exception as e:
            print(f"❌ 处理失败: {item['doi']}，原因: {e}")
            return False
    
    def process_items(self, items: List[Dict]) -> None:
        # 1. 筛选出所有Elsevier的条目
        elsevier_items = [item for item in items if item.get("publisher") == "Elsevier"]
        print(f"find {len(elsevier_items)} Elsevier items.")
        
        # 2. 加载已处理的DOI列表
        processed_dois = set()
        
        # 从成功处理的文件中加载DOI
        try:
            with open("./Data/elsevier_extracted.json", "r", encoding="utf-8") as f:
                extracted_data = json.load(f)
                for item in extracted_data:
                    if "doi" in item and item["doi"]:
                        processed_dois.add(item["doi"])
        except FileNotFoundError:
            print("📋 elsevier_extracted.json is not found, start from empty.")
            pass
        
        # 从失败处理的文件中加载DOI
        try:
            with open("./Data/elsevier_bad.json", "r", encoding="utf-8") as f:
                bad_data = json.load(f)
                bad_dois_count = 0
                for item in bad_data:
                    if "doi" in item and item["doi"]:
                        processed_dois.add(item["doi"])
                        bad_dois_count += 1
        except FileNotFoundError:
            print("📋 bad.json is not found, start from empty.")
            pass
        
        # 3. 筛选出未处理的条目
        unprocessed_items = []
        for item in elsevier_items:
            if item.get("doi") not in processed_dois:
                unprocessed_items.append(item)
        
        total_processed = len(processed_dois)
        total_unprocessed = len(unprocessed_items)
        
        print(f"Statistics:")
        print(f"   - Total Elsevier items: {len(elsevier_items)}")
        print(f"   - Processed items: {total_processed}")
        print(f"   - Unprocessed items: {total_unprocessed}")
        
        if total_unprocessed == 0:
            print("✅ All Elsevier items are processed.")
            return
        
        # 4. 处理未处理的条目
        print(f"🚀 Start processing {total_unprocessed} unprocessed Elsevier items...")
        
        for i, item in enumerate(unprocessed_items, 1):
            print(f"\nProgress [{i}/{total_unprocessed}] - DOI: {item.get('doi', 'N/A')}")
            
            if not self.process_single_item(item):
                print(f"❌ Encounter serious error, stop processing. Processed {i-1}/{total_unprocessed} items.")
                break  # 遇到严重错误时停止
    
    def save_results(self, 
                    final_data_file: str = "./Data/elsevier_extracted.json",
                    bad_data_file: str = "./Data/elsevier_bad.json"):
        """保存处理结果"""
        # 保存成功处理的数据
        try:
            with open(final_data_file, "r", encoding="utf-8") as f:
                original_data = json.load(f)
        except FileNotFoundError:
            original_data = []
        
        original_data.extend(self.final_data)
        with open(final_data_file, "w", encoding="utf-8") as f:
            json.dump(original_data, f, ensure_ascii=False, indent=2)
        
        # 保存跳过的数据
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
        """关闭资源"""
        self.browser.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def main():
    """主函数"""
    # 加载输入数据
    with open("./Databases/publisher_doi.json", "r", encoding="utf-8") as f:
        input_json = json.load(f)
    
    # 处理数据
    with WebScraper(headless=False) as scraper:
        scraper.process_items(input_json)
        scraper.save_results()

if __name__ == "__main__":
    main()