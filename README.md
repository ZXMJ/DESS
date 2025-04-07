# 大语言模型赋能动态评估与筛选策略的检索优化
我们将实验分为 **实验1**、**实验2** 两部分进行

## 实验1
### 创建实验1 conda 环境
```shell
conda activate dessR1 python=3.10
```
### 下载实验1 dessR1 requirements
切换到dessR1目录下的 requirement.txt
```shell
pip install -r requirements.txt
```
### 下载elasticsearch-8.9.0
切换到dessR1环境下下载
这是下载的地址 [elasticsearch-8.9.0](https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.17.3-windows-x86_64.zip)). 
也可以选择各大云服务器平台提供的elasticsearch，根据自己的需求去进行选择
也可以参考我基准论文[Blended-RAG项目](https://github.com/ibm-ecosystem-engineering/Blended-RAG)). 

### Code
- **rag_msmarco.py**: 这是Msmarco数据集评估的文件.
- **rag_nq.py**: 这是NQ数据集评估的文件.
- **rag_squad.py**: 这是squad数据集评估的文件.
- **rag_trivia.py**: 这是trivia数据集评估的文件.
- **index**：这个目录下的python文件是上面提到的各个数据集创建索引的文件


### Input 
This module uses various inputs, such as mapping and search_query, to index and search the queries at the index.

- **mapping/**: Contains mapping files with respective BM25, KNN and Sparse_Encoder.
- **search_query/**: A collection of search_queries used across different evaluation tasks.


## 实验2
### 创建实验2 conda 环境
```shell
conda activate dessR2 python=3.10
```
### 下载实验2 dessR2 requirements
切换到dessR2目录下的 requirement.txt
```shell
pip install -r requirements.txt
```
### 下载pyserini
```shell
pip install -U openai pyserini
```
或者
Install `pyserini` by following the [guide](https://github.com/castorini/pyserini#-installation). We use pyserini to conduct dense retrieval and evaluation.
### 2. 下载及加载数据 
```shell
mkdir ./indexes/
wget https://www.dropbox.com/s/rf24cgsqetwbykr/lucene-index-msmarco-passage.tgz?dl=0
wget https://www.dropbox.com/s/5vhl1aynl0kg3rj/contriever_msmarco_index.tar.gz?dl=0
# then unzip two tgz files into ./indexes/

mkdir ./data_msmarco/
wget https://www.dropbox.com/s/yms13b9k850b3vt/collection.tsv?dl=0
# then put tsv file into ./data_msmarco/
```
### 3. Code.
- **bm25Contriever.py**: 这是Bm25跟Contriever相结合索引进行评估的文件.
- **run_dl19.sh** **run_dl20.sh**: 这是两个执行运行命令的脚本文件.
### 4. Run.
```shell
mkdir ./runs_inter
./run_dl19.sh  # for DL'19
./run_dl20.sh  # for DL'20
```

