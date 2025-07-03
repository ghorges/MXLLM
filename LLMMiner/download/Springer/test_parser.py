#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Springer解析器测试脚本
用于测试解析器是否能正确提取Springer文章的摘要和章节内容
"""

from parser import ContentParser

def test_springer_parser():
    """测试Springer解析器"""
    
    # 测试HTML内容（基于你提供的示例）
    test_html = '''
    <html>
    <body>
        <!-- 摘要部分 -->
        <div class="c-article-section__content" id="Abs1-content">
            <p>Although strong and effective absorption of electromagnetic waves in the X-band region is of utmost importance for practical applications, it still faces many challenges. In this work, we prepared Fe<sub>2</sub>O<sub>3</sub>-decorated Ti<sub>3</sub>C<sub>2</sub> composites via a facile one-step solvothermal route. Scanning electron microscopy micrographs showed that Fe<sub>2</sub>O<sub>3</sub> particles effectively cover the Ti<sub>3</sub>C<sub>2</sub> surface and insert into the Ti<sub>3</sub>C<sub>2</sub> layers. X-ray diffraction patterns demonstrated that the Fe<sub>2</sub>O<sub>3</sub> particles are beneficial for the Ti<sub>3</sub>C<sub>2</sub> delamination. A high attenuation constant and suitable impedance matching enabled wide-range microwave absorption properties; however, the minimum reflection loss was relatively high. To achieve both wide bandwidth and strong absorption, we designed a multi-layered structural absorber. When the thickness was 1.9&nbsp;mm, the fourfold-layered structural absorber exhibited the optimal absorption properties with the minimum RL value of&nbsp;&minus;&nbsp;49.68&nbsp;dB at 9.922&nbsp;GHz and RL less than &minus;&nbsp;10.82&nbsp;dB in the entire X-band region. Therefore, the synthesized multi-layered structural Ti<sub>3</sub>C<sub>2</sub>/Fe<sub>2</sub>O<sub>3</sub> composites can be used for practical applications as a high-performance microwave absorber in the X-band region.</p>
        </div>
        
        <!-- 章节部分 -->
        <div class="c-article-section" id="Sec1-section">
            <h2 class="c-article-section__title js-section-title js-c-reading-companion-sections-item" id="Sec1">
                <span class="c-article-section__title-number">1 </span>Introduction
            </h2>
            <div class="c-article-section__content" id="Sec1-content">
                <p>This is the introduction section content.</p>
                <p>Another paragraph in the introduction.</p>
            </div>
        </div>
        
        <div class="c-article-section" id="Sec2-section">
            <h2 class="c-article-section__title js-section-title js-c-reading-companion-sections-item" id="Sec2">
                <span class="c-article-section__title-number">2 </span>Methods
            </h2>
            <div class="c-article-section__content" id="Sec2-content">
                <p>This is the methods section content.</p>
                <p>Experimental procedures are described here.</p>
            </div>
        </div>
        
        <div class="c-article-section" id="Sec16-section">
            <h2 class="c-article-section__title js-section-title js-c-reading-companion-sections-item" id="Sec16">
                <span class="c-article-section__title-number">5 </span>Challenges and prospection
            </h2>
            <div class="c-article-section__content" id="Sec16-content">
                <p>This article aims to summarize the flexibility and transparency potential of mainstream EMI shielding materials and review the latest developments in flexible and transparent composite materials based on AgNW networks.</p>
                <p>Obtaining satisfactory EMI shielding performance needs to be considered first in the preparation of materials.</p>
            </div>
        </div>
        
        <!-- 关键词部分 -->
        <section data-title="Keywords">
            <a class="c-article-subject__link">MXene</a>
            <a class="c-article-subject__link">Electromagnetic interference</a>
            <a class="c-article-subject__link">Shielding materials</a>
        </section>
        
        <!-- 表格 -->
        <table>
            <tr><td>Test Table</td></tr>
        </table>
        
        <!-- 图片 -->
        <figure>
            <figcaption>Test Figure Caption</figcaption>
        </figure>
    </body>
    </html>
    '''
    
    # 创建解析器实例
    parser = ContentParser()
    
    # 解析HTML
    result = parser.parse_html(test_html, "Test Article")
    
    # 打印结果
    print("=" * 50)
    print("Springer解析器测试结果")
    print("=" * 50)
    
    print(f"\n📄 文章名称: {result['name']}")
    
    print(f"\n📝 摘要:")
    print("-" * 30)
    print(result['abstract'])
    
    print(f"\n📖 正文内容 (共{len(result['text'])}段):")
    print("-" * 30)
    for i, text in enumerate(result['text'], 1):
        print(f"{i}. {text[:100]}{'...' if len(text) > 100 else ''}")
    
    print(f"\n🏷️ 关键词 (共{len(result['tags'])}个):")
    print("-" * 30)
    for tag in result['tags']:
        print(f"- {tag}")
    
    print(f"\n📊 表格 (共{len(result['table'])}个):")
    print("-" * 30)
    for i, table in enumerate(result['table'], 1):
        print(f"Table {i}: {table[:50]}...")
    
    print(f"\n🖼️ 图片 (共{len(result['figures'])}个):")
    print("-" * 30)
    for figure in result['figures']:
        print(f"- {figure}")
    
    print("\n" + "=" * 50)
    print("测试完成!")
    print("=" * 50)

if __name__ == "__main__":
    test_springer_parser() 