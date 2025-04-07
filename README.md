#   DESS : Retrieval Optimization Through Dynamic Evaluation and Selection Strategies Empowered by Large Language Models

We divided the experiment into two parts **Experiment 1** and **Experiment 2**.

## Experiment 1 ： dessR1
### Create the experiment 1 conda environment
```shell
conda activate dessR1 python=3.10
```
### Download experiment 1 dessR1 requirements
Switch to requirement.txt in the dessR1 directory.
```shell
pip install -r requirements.txt
```
### Download elasticsearch-8.9.0
Switch to the dessR1 environment to download
Here is the download address [elasticsearch-8.9.0](https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.17.3-windows-x86_64.zip)). 
You can also choose the elasticsearch provided by the major cloud server platforms, according to your own needs to choose!

You can also refer to my benchmark paper [Blended-RAG project](https://github.com/ibm-ecosystem-engineering/Blended-RAG)). 

All datasets used in the experiments are publicly available and can be found and downloaded from their respective official websites.

### Code
- **rag_msmarco.py**: This is the file for the Msmarco dataset evaluation.
- **rag_nq.py**: This is the file for evaluating the NQ dataset.
- **rag_squad.py**: This is the file for evaluating the squad dataset.
- **rag_trivia.py**: This is the file for trivia dataset evaluation.
- **index**: The python files in this directory are the files that create the indexes for each of the datasets mentioned above.

### Input 
This module uses various inputs, such as mapping and search_query, to index and search the queries at the index.

- **mapping/**: Contains mapping files with respective BM25, KNN and Sparse_Encoder.
- **search_query/**: A collection of search_queries used across different evaluation tasks.

## Experiment 2: dessR2
### Create the experiment 2 conda environment
```shell
conda activate dessR2 python=3.10
```
### Download experiment 2 dessR2 requirements
Switch to requirement.txt in the dessR2 directory.
```shell
pip install -r requirements.txt
```
### Download pyserini
```shell
pip install -U openai pyserini
```
Or
Install `pyserini` by following the [guide](https://github.com/castorini/pyserini#-installation). We use pyserini to conduct dense retrieval and evaluation.
### 2. Downloading and loading data
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
- **valuation.py**: This is the file that evaluates the indexing of Bm25 in combination with Contriever.
- **run_dl19.sh** **run_dl20.sh**: These are the two script files that execute the run command.
### 4. Run.
```shell
mkdir ./runs_inter
./run_dl19.sh  # for DL'19
./run_dl20.sh  # for DL'20
```

