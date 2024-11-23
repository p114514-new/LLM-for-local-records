import os

from pymilvus import Collection, utility, FieldSchema, CollectionSchema, DataType, MilvusClient
import numpy as np
from rank_bm25 import BM25Okapi
from FlagEmbedding import FlagModel

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['no_proxy'] = '127.0.0.1,localhost'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['NO_PROXY'] = '127.0.0.1,localhost'


# Connect to Milvus server
def connect_milvus():
    client = MilvusClient("milvus_demo.db")
    print("Client connected.")
    return client


# Define schema and create collection
def create_collection(client, collection_name, dim):
    if client.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists.")
        return

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]

    schema = CollectionSchema(fields, description="Document retrieval collection")
    client.create_collection(collection_name, schema=schema, dimension=dim)
    print(f"Collection '{collection_name}' created.")


# Insert data into the collection
# Insert data into the collection
def insert_data(client, collection_name, texts, model, dim):
    # Encode texts to get embeddings
    embeddings = model.encode(texts, convert_to_numpy=True)

    # Ensure embeddings have the correct shape
    if embeddings.shape[1] != dim:
        raise ValueError(f"Embedding dimension mismatch: expected {dim}, got {embeddings.shape[1]}")

    # Prepare data without IDs (auto_id=True)
    data = [
        texts,  # Raw texts
        embeddings.tolist()  # Embeddings
    ]

    # transform data to dict
    data = [{"text": text, "embedding": embedding} for text, embedding in zip(*data)]

    # Insert data
    client.insert(collection_name, data)
    print("Data inserted successfully.")

    # Create index
    # Define the index parameters
    index_params = [{
        "field_name": "embedding",  # Define the field name to create the index
        "index_type": "HNSW",  # Choose index type (e.g., IVF_FLAT, IVF_SQ8)
        "metric_type": "IP",  # Choose the metric type (e.g., IP, L2, COSINE)
        "params": {"nlist": 128}  # Number of clusters (nlist) for the index
    }]

    # Create the index on the "embedding" field
    client.create_index(
        collection_name="document_retriever",
        index_params=index_params
    )

    print("Index created successfully.")


def clear_database(client, collection_name):
    client.drop_collection(collection_name)


def prepare_bm25(texts, language='chinese'):
    if language.lower() == 'chinese':
        import jieba
        jieba.enable_parallel()
        tokenized_corpus = [list(jieba.cut(doc)) for doc in texts]
    elif language.lower() == 'english':
        from nltk.tokenize import word_tokenize
        tokenized_corpus = [word_tokenize(doc) for doc in texts]
    else:
        raise AttributeError("Language not supported. Please use 'chinese' or 'english'.")
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def bm25_search(bm25, query, texts, language='chinese', top_k=5):
    if language.lower() == 'chinese':
        import jieba
        tokenized_query = list(jieba.cut(query))
    elif language.lower() == 'english':
        from nltk.tokenize import word_tokenize
        tokenized_query = word_tokenize(query)
    else:
        raise AttributeError("Language not supported. Please use 'chinese' or 'english'.")
    scores = bm25.get_scores(tokenized_query)
    sorted_indices = np.argsort(scores)[::-1][:top_k]
    results = [(texts[i], scores[i]) for i in sorted_indices]
    return results


# Perform embedding-based search
def embedding_search(client, collection_name, query, model, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True).tolist()
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    results = client.search(
        collection_name=collection_name,
        data=query_embedding,
        anns_field="embedding",
        search_params=search_params,
        limit=top_k,
        output_fields=["text"]
    )
    return [(hit['entity']['text'], hit['distance']) for hit in results[0]]


# Main driver function
def main():
    collection_name = "document_retriever"

    # Initialize connections and model
    client = connect_milvus()
    clear_database(client, collection_name)
    model = FlagModel('BAAI/bge-large-zh-v1.5',
                      query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                      use_fp16=True,
                      cache_dir=r'/media/sata8t/swx/models')

    # Example document corpus
    texts1 = [
        "2021 年1 月26 日，光明街道召开党工委办事处2021 年工作会议",
        "2021 年2 月25 日，光明街道召开高质量高颜值发展大会",
        "2021 年3 月2 日， 光明区区长蔡颖（中）率队到光明街道开展调研活动",
        "2021 年3 月16 日，光明街道召开2021 年党管武装工作专题会议",
        "2021 年4 月13 日，光明区委书记刘胜（右三）到光明回归亭公园开展调研活动",
        "2021 年5 月26 日，光明街道党工委副书记顾成军（中）代表光明街道与白花社区代表签订高风险区域责任书",
        "2021 年7 月22 日，光明人大工委主任周荣生（中）到翠湖社区开展调研活动",
        "2021 年8 月4 日，光明街道组织召开文明城市创建专题会议",
        "2021 年8 月19 日，光明街道党工委书记黄文胜（左五）率队开展文明创建督导活动，现场解决文明城市创建难题",
        "2021 年8 月24 日，光明区区长蔡颖（中）到光明街道开展文明城市创建工作调研",
        "2021 年8 月24 日，光明区副区长梁非凡（右）到光明街道开展文明创建督导活动，对各社区文明创建工作进行检查",
        "2021 年8 月31 日，深圳市党代表、市政府秘书长高圣元（前排正中）带队到光明街道迳口社区开展“市党代表进社区”暨“我为群众办实事”调研",
        "2021 年10 月19 日，光明街道光明社区开展“行走光明·齐抓文明”活动，社区书记林海东（右三）带领“两委”班子进行文明城市创建巡查督导整治“回头看”",
        "2021 年11 月4 日，光明街道举行2021 年“11·9”消防安全宣传月启动仪式暨义务消防队技能大比武活动",
    ]

    texts2 = [
        "2021 年11 月4 日，马田街道传达学习贯彻党的十九届六中全会精神干部大会召开",
        "2021 年11 月4 日，马田街道安全生产重难点隐患整治工作推进会召开, 针对辖区内重点单位、重点工程、重点场所、重点行业等重难点隐患进行研讨",
        "2021 年11 月4 日，光明街道举行2021 年“11·9”消防安全宣传月启动仪式暨义务消防队技能大比武活动",
        "2021 年11 月4 日，白田街道迎接中央第四生态环境保护督察组工作再动员再部署会议召开",
        "2021 年11 月4 日，白田街道党工委、办事处2021年工作会议召开",
    ]

    using_text = texts1

    # prepare document embeddings
    bm25 = prepare_bm25(using_text, 'chinese')

    # Insert data into Milvus
    dim = 1024
    create_collection(client, collection_name, dim)
    insert_data(client, collection_name, using_text, model, dim)

    # Query examples
    # 时间：t1q1, 地点：t2q2, 职位：t1q4
    query1 = "光明街道在2021年8月24日进行了哪些活动？"
    query2 = "2021年11月4日，马田街道政府部门进行了哪些工作？"
    query3 = "2021年11月4日，马田街道召开了哪些会议？"
    query4 = "光明区的副区长是谁？"

    using_query = query1

    print("BM25 Results:")
    bm25_results = bm25_search(bm25, using_query, using_text, 'chinese')
    for text, score in bm25_results:
        print(f"Text: {text}, Score: {score}")

    print("\nEmbedding-Based Results:")
    embedding_results = embedding_search(client, collection_name, using_query, model)
    for text, distance in embedding_results:
        print(f"Text: {text}, Distance: {distance}")

    client.close()


if __name__ == "__main__":
    main()
