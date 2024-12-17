import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache
from lightrag.llm import openai_embedding   
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
import os
import numpy as np
import textract

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "glm-4-flash",                                        # LLM模型名称
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="15c105fd5e2f6aa87e4dda517567ed79.fQyPrc5Tgs3PoAg0",                                  # LLM_api_key
        base_url="https://open.bigmodel.cn/api/paas/v4",      # LLM_url
        **kwargs
    )

async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model="BAAI/bge-m3",                      # Embedding模型名称
        api_key="sk-gvbvidszvlllrpjpqzagcsukujykjbzdsgardrrdlfijepqw",                      # Embedding API KEY
        base_url="https://api.siliconflow.cn/v1"  # Embedding url
    )

WORKING_DIR = "./dickens"           # 定义存放结果的文件夹
 
if not os.path.exists(WORKING_DIR): # 如果不存在就新建一个
    os.mkdir(WORKING_DIR)
 
rag = LightRAG(
    working_dir=WORKING_DIR,         # 存放数据、知识图谱的文件夹
    llm_model_func=llm_model_func,   # 自定义的LLM模型
    embedding_func=EmbeddingFunc(
        embedding_dim = 1024,
        max_token_size = 8192,
        func = embedding_func        # 自定义的Embedding模型
    )
   
)


# file_path = 'data/公明镇工业.docx'
# text_content = textract.process(file_path)

# data_folder = r'/home/xyd/test/LightRAG-main/data/公明镇去图片版本'
# text_content = []
# for root, dirs, files in os.walk(data_folder):
#     for file in files:
#         if file.endswith('.docx'):
#             file_path = os.path.join(root, file)
#             text_content.append(textract.process(file_path))
#
# rag.insert([t.decode('utf-8') for t in text_content])

# print(
#     rag.query("这篇文章包含了哪些主题", param=QueryParam(mode="local"))
# )
#
# # global模式
# print(
#     rag.query("这篇文章包含了哪些主题", param=QueryParam(mode="global"))
# )
 
# hybrid模式
print(
    rag.query("公明镇如何实践“一带三区五组团”的园区经济发展战略？", param=QueryParam(mode="hybrid"))
)

