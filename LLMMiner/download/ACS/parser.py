import os
from bs4 import BeautifulSoup
from typing import Dict, List

class ContentParser:
    """ACS HTML内容解析器"""
    def extract_abstract(self, soup: BeautifulSoup) -> str:
        abstract = []
        for paragraph in soup.find_all(class_='articleBody_abstractText'):
            paragraph_text = paragraph.get_text().strip()
            if paragraph_text:
                abstract.append(paragraph_text)
        return " ".join(abstract)

    def extract_text_blocks(self, soup: BeautifulSoup) -> List[str]:
        text = []
        for paragraph in soup.find_all(class_=['NLM_p', 'NLM_p last']):
            paragraph_text = paragraph.get_text().strip()
            if paragraph_text:
                text.append(paragraph_text)
        return text

    def extract_tags(self, soup: BeautifulSoup) -> List[str]:
        tags = []
        for tag in soup.select('.article__tags__item a'):
            tag_text = tag.get_text()
            if tag_text:
                tags.append(tag_text)
        return tags

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