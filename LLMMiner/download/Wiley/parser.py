import os
from bs4 import BeautifulSoup
from typing import Dict, List

class ContentParser:
    """Wiley HTML内容解析器"""
    def __init__(self, min_length: int = 40):
        self.min_length = min_length

    def extract_abstract(self, soup: BeautifulSoup) -> str:
        abstract_paragraphs = []
        for div in soup.find_all("div", class_="article-section__content en main"):
            for p in div.find_all("p"):
                content = p.get_text().strip()
                if len(content) >= self.min_length:
                    abstract_paragraphs.append(content)
        return "\n".join(abstract_paragraphs)

    def extract_text_blocks(self, soup: BeautifulSoup) -> List[str]:
        text_blocks = []
        for section in soup.find_all("section"):
            class_list = section.get("class", [])
            class_str = " ".join(class_list)
            if class_str in ["article-section__content", "article-section__content en main"]:
                for p in section.find_all("p"):
                    content = p.get_text().strip()
                    if len(content) >= self.min_length:
                        text_blocks.append(content)
        return list(dict.fromkeys(text_blocks))

    def extract_tags(self, soup: BeautifulSoup) -> List[str]:
        # Wiley暂未实现标签提取
        return []

    def extract_tables(self, soup: BeautifulSoup) -> List[str]:
        table_blocks = [str(table) for table in soup.find_all("table")]
        return list(dict.fromkeys(table_blocks))

    def parse_html(self, html_content: str, file_name: str = None) -> Dict:
        soup = BeautifulSoup(html_content, 'lxml')
        return {
            "name": file_name or "",
            "abstract": self.extract_abstract(soup),
            "text": self.extract_text_blocks(soup),
            "tags": self.extract_tags(soup),
            "table": self.extract_tables(soup)
        } 