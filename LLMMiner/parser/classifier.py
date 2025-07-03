from typing import List, Optional, Dict
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from ..static.classification_prompts import (
    CLASSIFICATION_PROMPT,
    CATEGORY_DESCRIPTION,
    DEFAULT_CATEGORIES
)

class TextClassifier:
    """文本分类器类，使用LLM进行文本分类"""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        categories: Optional[List[str]] = None
    ):
        """
        初始化分类器
        
        Args:
            model_name: OpenAI模型名称
            temperature: 模型温度参数
            categories: 自定义类别列表，如果为None则使用默认类别
        """
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
        
        self.categories = categories if categories is not None else DEFAULT_CATEGORIES
        self.prompt = ChatPromptTemplate.from_template(CLASSIFICATION_PROMPT)
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        
    def classify(self, text: str) -> str:
        """
        对输入文本进行分类
        
        Args:
            text: 待分类的文本内容
            
        Returns:
            str: 分类结果
        """
        result = self.chain.run({
            "text": text,
            "categories": "\n".join(f"- {cat}: {CATEGORY_DESCRIPTION.get(cat, '')}" 
                                  for cat in self.categories)
        })
        return result.strip()
    
    def batch_classify(self, texts: List[str]) -> List[str]:
        """
        批量对文本进行分类
        
        Args:
            texts: 待分类的文本列表
            
        Returns:
            List[str]: 分类结果列表
        """
        return [self.classify(text) for text in texts] 