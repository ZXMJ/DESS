import os
import re
from datetime import date
import pandas as pd
import json
from datetime import datetime
import requests
from datasets import load_dataset
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError
import hashlib
from elasticsearch.exceptions import NotFoundError


## Replace elastic instance here
es_client = Elasticsearch(
    ["http://10.224.164.95:9200"]
)
es_client.info()

## Download model for KNN
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

## get the files from specific folder
def get_all_files(folder_name):
    # Change the directory
    os.chdir(folder_name)
    # iterate through all file
    file_path_list =[]
    for file in os.listdir():
        print(file)
        file_path = f"{folder_name}/{file}"
        file_path_list.append(file_path)
    return file_path_list


## create the index
def create_index(index_name,mapping):
    try:
        es_client.indices.create(index=index_name,body = mapping)
        print(f"Index '{index_name}' created successfully.")
    except RequestError as e:
        if e.error == 'resource_already_exists_exception':
            print(f"Index '{index_name}' already exists.")
        else:
            print(f"An error occurred while creating index '{index_name}': {e}")
# 根据 question 列生成唯一 ID
def generate_id(question):
    return hashlib.md5(question.encode()).hexdigest()

from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords(text, max_features=20):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_array = vectorizer.get_feature_names_out()
    return " ".join(feature_array)  # 提取前 max_features 个关键词


def index_data(df_docs,source,index_name,index_name_knn,index_name_elser):
    i=0
    for index, row in df_docs.iterrows():
        i=i+1
        print("Processing i",i)
        id = generate_id(row["question"])
        text = row['context']
        # title = row['title']
        source = source
        text_embedding = model.encode(text)
        # 提取 story 的关键词，作为 ml.tokens 的内容
        ml_tokens = extract_keywords(text)
        doc ={
                        "id": id,
                        "source": source,
                        "story": text
            }
        doc_knn = {
                        "id": id,
                        "source": source,
                        "story": text,
                        "story_embedding": text_embedding
                    }
        doc_elser = {
                    
                        "id": id,
                        "source": source,
                        "story": text,
                        "ml.tokens": ml_tokens  
                    }

        # 尝试获取文档以检查是否已存在
        try:
            es_client.get(index=index_name, id=id)
            # 如果文档已存在，则执行更新操作
            response = es_client.update(index=index_name, id=id, body={"doc": doc})
            print(f"Updated document with ID {id}: {response}")
        except NotFoundError:
            # 如果文档不存在，则执行插入操作
            response = es_client.index(index=index_name, id=id, body=doc)
            print(f"Indexed new document with ID {id}: {response}")

        # 处理 KNN 索引
        try:
            es_client.get(index=index_name_knn, id=id)
            # 如果文档已存在，则执行更新操作
            response = es_client.update(index=index_name_knn, id=id, body={"doc": doc_knn})
            print(f"Updated KNN document with ID {id}: {response}")
        except NotFoundError:
            # 如果文档不存在，则执行插入操作
            response = es_client.index(index=index_name_knn, id=id, body=doc_knn)
            print(f"Indexed new KNN document with ID {id}: {response}")
            
        # 处理 index_name_elser 索引
        try:
            es_client.get(index=index_name_elser, id=id)
            # 如果文档已存在，则执行更新操作
            response = es_client.update(index=index_name_elser, id=id, body={"doc": doc_elser})
            print(f"Updated elser document with ID {id}: {response}")
        except NotFoundError:
            # 如果文档不存在，则执行插入操作
            response = es_client.index(index=index_name_elser, id=id, body=doc_elser)
            print(f"Indexed new elser document with ID {id}: {response}")
def delete_index(index_name):
    # 1. 删除现有索引
    try:
        if es_client.indices.exists(index=index_name):
            es_client.indices.delete(index=index_name)
            print(f"Index '{index_name}' has been deleted.")
        else:
            print(f"Index '{index_name}' does not exist.")
    except Exception as e:
        print(f"Error deleting index: {e}")




## Example Index name
index_name_knn = "research_index_knn_ms"
index_name = "research_index_bm25_ms"
index_name_elser = "research_index_elser_ms"
delete_index(index_name)
delete_index(index_name_knn)
delete_index(index_name_elser)

## Create Index BM25
with open('/mnt/workspace/rag/RAG/Blended-RAG/input/mapping/bm25.txt', 'r') as file:
    mapping_str = file.read().rstrip()
        # 将 JSON 字符串解析为 Python 字典
# print("Mapping string:", mapping_str)

mapping = json.loads(mapping_str)
create_index(index_name,mapping)

# Create Index Knn
with open('/mnt/workspace/rag/RAG/Blended-RAG/input/mapping/knn.txt', 'r') as file:
    mapping_str = file.read().rstrip()
mapping = json.loads(mapping_str)
create_index(index_name_knn,mapping)

# Create Index Sparse Encoder for ELSER V1 model
with open('/mnt/workspace/rag/RAG/Blended-RAG/input/mapping/sparse_encoder.txt', 'r') as file:
    mapping_str = file.read().rstrip()
mapping = json.loads(mapping_str)
create_index(index_name_elser,mapping)

df_docs = pd.read_csv('/mnt/workspace/rag/RAG/date/ms_context_sampled.csv')

source ="msmarco"
index_data(df_docs,source,index_name,index_name_knn,index_name_elser)