import os
from bs4 import BeautifulSoup
from typing import Dict, List

class ContentParser:
    """Springer HTML内容解析器"""
    
    def extract_highlights(self, soup: BeautifulSoup) -> str:
        """提取Springer文章的highlights"""
        highlights = []
        
        # 检查Abs1-section和Abs2-section，找到highlights
        for section_id in ['Abs1-section', 'Abs2-section']:
            section = soup.find('div', {'id': section_id})
            if section:
                title_element = section.find('h2', class_='c-article-section__title')
                if title_element:
                    title_text = title_element.get_text().strip().lower()
                    if 'highlights' in title_text:
                        # 提取highlights内容
                        content_div = section.find('div', {'id': f'{section_id.replace("-section", "")}-content'})
                        if content_div:
                            # 处理ul/li结构
                            for ul in content_div.find_all('ul'):
                                for li in ul.find_all('li'):
                                    for p in li.find_all('p'):
                                        paragraph_text = p.get_text().strip()
                                        if paragraph_text:
                                            highlights.append(paragraph_text)
                            # 如果没有ul/li，则处理普通段落
                            if not highlights:
                                for paragraph in content_div.find_all('p'):
                                    paragraph_text = paragraph.get_text().strip()
                                    if paragraph_text:
                                        highlights.append(paragraph_text)
                        break
        
        return " ".join(highlights)

    def extract_abstract(self, soup: BeautifulSoup) -> str:
        """提取Springer文章的正式摘要"""
        abstract = []
        
        # 检查Abs1-section和Abs2-section，找到abstract
        for section_id in ['Abs1-section', 'Abs2-section']:
            section = soup.find('div', {'id': section_id})
            if section:
                title_element = section.find('h2', class_='c-article-section__title')
                if title_element:
                    title_text = title_element.get_text().strip().lower()
                    if 'abstract' in title_text:
                        # 提取abstract内容
                        content_div = section.find('div', {'id': f'{section_id.replace("-section", "")}-content'})
                        if content_div:
                            for paragraph in content_div.find_all('p'):
                                paragraph_text = paragraph.get_text().strip()
                                if paragraph_text:
                                    abstract.append(paragraph_text)
                        break
        
        return " ".join(abstract)

    def extract_text_blocks(self, soup: BeautifulSoup) -> List[str]:
        """提取Springer文章的正文内容，从Sec1-section开始遍历到Sec50-section"""
        text_blocks = []
        
        # 从Sec1-section遍历到Sec50-section
        for section_number in range(1, 51):
            section_id = f"Sec{section_number}-section"
            section_div = soup.find('div', {'id': section_id})
            
            if not section_div:
                continue
                
            # 提取章节标题
            title_element = section_div.find('h2', class_='c-article-section__title')
            if title_element:
                title_text = title_element.get_text().strip()
                if title_text:
                    text_blocks.append(f"## {title_text}")
            
            # 提取章节内容
            content_div = section_div.find('div', {'id': f'Sec{section_number}-content'})
            if content_div:
                for paragraph in content_div.find_all('p'):
                    paragraph_text = paragraph.get_text().strip()
                    if paragraph_text:
                        text_blocks.append(paragraph_text)
        
        return text_blocks

    def extract_tags(self, soup: BeautifulSoup) -> List[str]:
        """提取Springer文章的关键词标签"""
        tags = []
        # 查找关键词部分
        keywords_section = soup.find('section', {'data-title': 'Keywords'})
        if keywords_section:
            for keyword in keywords_section.find_all('a', class_='c-article-subject__link'):
                tag_text = keyword.get_text().strip()
                if tag_text:
                    tags.append(tag_text)
        return tags

    def extract_tables(self, soup: BeautifulSoup) -> List[str]:
        """提取Springer文章的表格"""
        table_blocks = []
        for table in soup.find_all("table"):
            table_html = str(table)
            if table_html:
                table_blocks.append(table_html)
        return list(dict.fromkeys(table_blocks))

    def extract_figures(self, soup: BeautifulSoup) -> List[str]:
        pass

    def extract_tables_from_main(self, html_content: str) -> List[str]:
        """从<main>标签中提取所有table"""
        soup = BeautifulSoup(html_content, 'lxml')
        main = soup.find('main')
        if not main:
            return []
        return [str(table) for table in main.find_all('table')]

    def extract_name(self, soup: BeautifulSoup) -> str:
        """提取Springer文章的标题"""
        title_element = soup.find('h1', class_='c-article-title')
        if title_element:
            return title_element.get_text().strip()
        return ""

    def parse_html(self, html_content: str, file_name: str = None) -> Dict:
        """解析Springer HTML内容"""
        soup = BeautifulSoup(html_content, 'lxml')
        return {
            "name": self.extract_name(soup),
            "highlights": self.extract_highlights(soup),
            "abstract": self.extract_abstract(soup),
            "text": self.extract_text_blocks(soup),
            "tags": self.extract_tags(soup),
            "table": self.extract_tables(soup),
        } 