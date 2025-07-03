import os
from bs4 import BeautifulSoup
from typing import Dict, List

class ContentParser:
    """RSC HTML内容解析器"""

    def extract_name(self, soup: BeautifulSoup) -> str:
        """提取RSC文章的标题"""
        title_div = soup.find('div', class_='article__title')
        if title_div:
            title_element = title_div.find('h2', class_='capsule__title')
            if title_element:
                return title_element.get_text().strip()
        return ""

    def extract_abstract(self, soup: BeautifulSoup) -> str:
        """提取RSC文章的摘要"""
        abstract_heading = soup.find('h3', class_='h--heading3 article-abstract__heading', string='Abstract')
        if abstract_heading:
            column_wrapper = abstract_heading.find_next('div', class_='capsule__column-wrapper')
            if column_wrapper:
                capsule_text = column_wrapper.find('div', class_='capsule__text')
                if capsule_text:
                    paragraphs = capsule_text.find_all('p')
                    abstract_text = ' '.join([p.get_text().strip() for p in paragraphs])
                    return abstract_text
        return ""

    def extract_text_blocks(self, soup: BeautifulSoup) -> List[str]:
        """提取RSC文章的正文内容"""
        text_blocks = []
        article_content = soup.find('div', id='pnlArticleContent')
        if article_content:
            # 提取所有段落文本
            for paragraph in article_content.find_all('p'):
                paragraph_text = paragraph.get_text().strip()
                if paragraph_text:
                    text_blocks.append(paragraph_text)
        
        return text_blocks

    def extract_tables(self, soup: BeautifulSoup) -> List[str]:
        """提取RSC文章的表格"""
        table_blocks = []
        article_content = soup.find('div', id='pnlArticleContent')
        if article_content:
            for table in article_content.find_all("table"):
                table_html = str(table)
                if table_html:
                    table_blocks.append(table_html)
        return list(dict.fromkeys(table_blocks))  # 去重

    def extract_figures(self, soup: BeautifulSoup) -> List[str]:
        """提取RSC文章的图片"""
        figure_blocks = []
        article_content = soup.find('div', id='pnlArticleContent')
        if article_content:
            for figure in article_content.find_all(['figure', 'img']):
                figure_html = str(figure)
                if figure_html:
                    figure_blocks.append(figure_html)
        return list(dict.fromkeys(figure_blocks))  # 去重

    def extract_highlights(self, soup: BeautifulSoup) -> str:
        """提取RSC文章的highlights，通常在摘要前后的亮点区域"""
        highlights = []
        
        # 查找可能的highlights区域
        highlight_sections = soup.find_all('div', class_='capsule__column-wrapper')
        for section in highlight_sections:
            heading = section.find(['h3', 'h4'], string=lambda text: text and 'highlight' in text.lower())
            if heading:
                for p in section.find_all('p'):
                    paragraph_text = p.get_text().strip()
                    if paragraph_text:
                        highlights.append(paragraph_text)
        
        return " ".join(highlights)

    def extract_tags(self, soup: BeautifulSoup) -> List[str]:
        """提取RSC文章的关键词标签"""
        tags = []
        
        # 查找可能的关键词区域
        keyword_sections = soup.find_all(['div', 'section'], class_=lambda c: c and 'keyword' in c.lower())
        for section in keyword_sections:
            for keyword in section.find_all(['a', 'span', 'li']):
                tag_text = keyword.get_text().strip()
                if tag_text and len(tag_text) < 50:  # 过滤长文本
                    tags.append(tag_text)
        
        return tags

    def parse_html(self, html_content: str, file_name: str = None) -> Dict:
        """解析RSC HTML内容"""
        soup = BeautifulSoup(html_content, 'lxml')
        return {
            "name": self.extract_name(soup),
            "abstract": self.extract_abstract(soup),
            "text": self.extract_text_blocks(soup),
            "tags": self.extract_tags(soup),
            "table": self.extract_tables(soup)
        } 