import json
import os
import sys
from pathlib import Path


# 设置文件路径
file_path = os.path.join("springer_clean_extract.json")

try:
    # 读取clean_extract.json文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查是否为列表
    if isinstance(data, list):
        paper_count = len(data)
        print(f"clean_extract.json中包含 {paper_count} 篇论文")
        
        # 统计DOI相关信息
        doi_count = sum(1 for paper in data if "doi" in paper)
        print(f"其中有 {doi_count} 篇论文包含DOI信息")
        no_doi_count = paper_count - doi_count
        print(f"没有DOI信息的论文数量: {no_doi_count}")
        
        # 新增统计：只有DOI的论文和DOI+其他信息的论文
        only_doi_count = sum(1 for paper in data if "doi" in paper and len(paper) == 1)
        doi_with_other_count = doi_count - only_doi_count
        
        print(f"\n仅包含DOI字段的论文数量: {only_doi_count}")
        print(f"同时包含DOI和其他字段的论文数量: {doi_with_other_count}")
        
        # 显示一些DOI示例
        if doi_count > 0:
            # 找一个只有DOI的例子
            only_doi_example = next((paper for paper in data if "doi" in paper and len(paper) == 1), None)
            if only_doi_example:
                print("\n只有DOI的论文示例:")
                # print(json.dumps(only_doi_example, indent=2, ensure_ascii=False))
            
            # 找一个有DOI和其他字段的例子
            rich_example = next((paper for paper in data if "doi" in paper and len(paper) > 1), None)
            if rich_example:
                print("\n包含DOI和其他字段的论文示例:")
                # print(json.dumps(rich_example, indent=2, ensure_ascii=False))
            
            # 显示一些论文的DOI示例
            print("\n论文DOI示例:")
            for i in range(min(3, paper_count)):
                print(f"论文 {i+1}: {data[i].get('doi', '无DOI')}")
            
            # 显示最后一篇论文的DOI
            if paper_count > 3:
                print(f"最后一篇论文: {data[-1].get('doi', '无DOI')}")
    else:
        print("警告: clean_extract.json不是一个列表格式")
        
except FileNotFoundError:
    print(f"错误: 文件 {file_path} 不存在")
except json.JSONDecodeError:
    print(f"错误: {file_path} 不是有效的JSON文件")
except Exception as e:
    print(f"发生错误: {e}") 