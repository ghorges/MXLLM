import re
from bs4 import BeautifulSoup
from typing import Dict, List, Optional

class ContentParser:
    """独立的HTML内容解析器"""
    
    def __init__(self, min_text_length: int = 40):
        self.min_text_length = min_text_length
    
    def extract_abstract(self, soup: BeautifulSoup) -> str:
        """提取摘要内容"""
        abstract = ""
        abstract_container = soup.find("div", class_="abstract author")
        if abstract_container:
            abstract_div = abstract_container.find("div", class_="u-margin-s-bottom")
            if abstract_div:
                abstract = abstract_div.get_text(separator=" ", strip=True)
        return abstract
    
    def extract_text_blocks(self, soup: BeautifulSoup) -> List[str]:
        """提取正文文本块"""
        text_blocks = []
        body_container = soup.find("div", class_="Body u-font-serif")
        if body_container:
            for div in body_container.find_all("div", class_="u-margin-s-bottom"):
                content = div.get_text(separator=" ", strip=True)
                if len(content) >= self.min_text_length:
                    text_blocks.append(content)
        return list(dict.fromkeys(text_blocks))  # 去重
    
    def extract_tables(self, soup: BeautifulSoup) -> List[str]:
        """提取表格内容"""
        table_blocks = [str(table) for table in soup.find_all("table")]
        return list(dict.fromkeys(table_blocks))  # 去重
    
    def extract_doi(self, html_content: str) -> str:
        """从HTML内容中提取DOI"""
        doi_match = re.search(r"https://doi\.org/[^\s\"'>]+", html_content)
        return doi_match.group(0) if doi_match else ""
    
    def parse_html(self, html_content: str) -> Dict:
        """解析HTML内容，返回结构化数据"""
        soup = BeautifulSoup(html_content, 'lxml')
        
        return {
            "doi": self.extract_doi(html_content),
            "abstract": self.extract_abstract(soup),
            "text": self.extract_text_blocks(soup),
            "tags": [],  # 预留字段
            "table": self.extract_tables(soup)
        }
    
    def is_pdf_embed(self, soup: BeautifulSoup) -> bool:
        """检查是否为PDF嵌入页面"""
        return bool(soup.find(class_="PdfEmbed"))
    
    def is_publisher_redirect(self, soup: BeautifulSoup) -> bool:
        """检查是否为出版商重定向页面"""
        spans = soup.find_all("span", class_="link-button-text")
        for span in spans:
            if span.get_text(strip=True).lower() == "view at publisher":
                return True
        return False
    
    def validate_page_content(self, html_content: str) -> Dict[str, any]:
        """验证页面内容是否有效"""
        soup = BeautifulSoup(html_content, 'lxml')
        
        validation_result = {
            "is_valid": True,
            "skip_reason": None,
            "issues": []
        }
        
        if self.is_pdf_embed(soup):
            validation_result.update({
                "is_valid": False,
                "skip_reason": "PdfEmbed"
            })
        elif self.is_publisher_redirect(soup):
            validation_result.update({
                "is_valid": False,
                "skip_reason": "View at publisher"
            })
        
        return validation_result