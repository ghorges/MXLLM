CLASSIFICATION_PROMPT = """你是一个专业的文本分类助手。请根据以下文本内容，将其分类到最合适的类别中。

文本内容：{text}

可选类别：
{categories}

请只返回一个类别名称，不需要解释。
"""

CATEGORY_DESCRIPTION = {
    "技术文档": "包含技术细节、API文档、代码示例等技术相关内容",
    "新闻资讯": "时事新闻、行业动态、市场信息等新闻类内容",
    "学术论文": "研究论文、学术报告、实验数据等学术类内容",
    "产品介绍": "产品说明、功能描述、使用教程等产品相关内容",
    "其他": "不属于以上类别的其他内容"
}

DEFAULT_CATEGORIES = list(CATEGORY_DESCRIPTION.keys()) 