import os
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding, hf_embedding, hf_model_complete
from lightrag.utils import EmbeddingFunc
import textract
from transformers import AutoTokenizer, AutoModel
import re
import json

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['no_proxy'] = '127.0.0.1,localhost'

WORKING_DIR = "./test2"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,
    llm_model_name="THUDM/glm-4-9b-chat",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=512,
        func=lambda texts: hf_embedding(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(
                "BAAI/bge-large-zh-v1.5",
                cache_dir=r'/media/sata1/swx/models',
                model_max_length=512,
            ),
            embed_model=AutoModel.from_pretrained(
                "BAAI/bge-large-zh-v1.5",
                cache_dir=r'/media/sata1/swx/models'
            ),
        ),
    ),
)

# data_folder = r'/media/sata1/swx/LightRAG-main/data/test'
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
tem = {"question": "", "ans": "", "source": []}
question = "公明区对外经济办公室有哪些曾用名？"
ans, sources = rag.query(question, param=QueryParam(mode="hybrid"))

tem["question"] = question
tem["ans"] = ans

for source_ in sources:
    print(source_)
    ans1 = rag.query1(ans, source_, question)
    tem["source"].append(ans1)

output_file_path = 'other.json'
with open(output_file_path, 'a', encoding='utf-8') as f:
    f.write(json.dumps(tem, ensure_ascii=False) + "\n")
