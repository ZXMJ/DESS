import os
import sys
from tqdm import tqdm
import numpy as np
from time import sleep
import time
import json
import csv
import argparse

from pyserini.search.lucene import LuceneSearcher

print(LuceneSearcher)

from pyserini.search import get_topics, get_qrels

import openai

csv.field_size_limit(sys.maxsize)
number2word = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
from collections import Counter

from utils import *

run_path = './runs_inter'
print(f'The working directory is {run_path}')


def model_init(comode):
    # model and index initilization

    print(f'{format_time()} initial lucene searcher ...')
    bm25_searcher = LuceneSearcher('indexes/lucene-index-msmarco-passage/')


    if comode == 'rmdemo':
        print(f'{format_time()} load trec collections ...')
        psg_collections = load_tsv('data_msmarco/collection.tsv')
    else:
        psg_collections = None

    return bm25_searcher, psg_collections


def load_topics_qrels(eval_trec_mode):
    if 'dl20' in eval_trec_mode:
        topics = get_topics('dl20')
    else:
        topics = get_topics(f'{eval_trec_mode}')

    qrels = get_qrels(f'{eval_trec_mode}')
    return topics, qrels


def eval_trec(all_qids, all_hits, eval_trec_mode, eval_prefix='bm25'):
    with open(f'{run_path}/{eval_trec_mode}-{eval_prefix}-top1000-trec', 'w') as f:
        for qid, hits in zip(all_qids, all_hits):
            rank = 0
            for hit in hits:
                rank += 1
                f.write(f'{qid} Q0 {hit.docid} {rank} {hit.score} rank\n')

    os.system(
        f'python -m pyserini.eval.trec_eval -c -l 2 -m map {eval_trec_mode} {run_path}/{eval_trec_mode}-{eval_prefix}-top1000-trec')
    os.system(
        f'python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 {eval_trec_mode} {run_path}/{eval_trec_mode}-{eval_prefix}-top1000-trec')
    os.system(
        f'python -m pyserini.eval.trec_eval -c -l 2 -m recall.1000 {eval_trec_mode} {run_path}/{eval_trec_mode}-{eval_prefix}-top1000-trec')

import subprocess

def eval_trec_single_qid(qid, hits, eval_trec_mode, eval_prefix='bm25'):

    with open(f'{run_path}/{eval_trec_mode}-{eval_prefix}-top10-trec', 'w') as f:
        rank = 0
        for hit in hits:
            rank += 1
            f.write(f'{qid} Q0 {hit.docid} {rank} {hit.score} rank\n')


    result = subprocess.run(
    [
        'python', '-m', 'pyserini.eval.trec_eval', 
        '-c', '-l', '2', '-m', 'ndcg_cut.10', 
        eval_trec_mode, 
        f'{run_path}/{eval_trec_mode}-{eval_prefix}-top10-trec'
    ], 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )


    if result.returncode == 0:

        lines = result.stdout.splitlines()
        for line in lines:
            if line.startswith("ndcg_cut_10"):

                ndcg_value = line.split()[-1]
                return float(ndcg_value) 
    else:
        print(f"Error during evaluation for qid {qid}:")
        print(result.stderr) 

    return None  

def get_llm_prompt(query, stories_list):
#     text = f"""
# You are given a query and a list of candidate documents. 
# Your task is to identify the 10 documents that are most relevant to the query, ranking them from most to least relevant. 
# Consider direct, indirect, and implicit information when assessing relevance.

# Query: "{query}"

# Candidate documents:
# {chr(10).join(stories_list)}

# Instructions:
# - Select the 10 documents that best match the query.
# - Rank them in order of relevance, with the most relevant document first.
# - Only select document numbers between 1 and {len(stories_list)}.
# - If two or more documents have very similar relevance, choose the document with the smaller number.
# - **Ensure there are no duplicate document numbers in your answer.** Each document number should appear only once.
# - Do not provide explanations or additional information. Only output the document numbers, ordered by relevance, separated by spaces.


# Provide only the document numbers (1-{len(stories_list)}), separated by spaces.
# """
    text = f"""
You are given a query and a list of candidate documents. 
Your task is to identify the 10 documents that are most relevant to the query, ranking them from most to least relevant. 
Consider direct, indirect, and implicit information when assessing relevance.

Query: "{query}"

Candidate documents:
{chr(10).join(stories_list)}

Instructions:
- Select the 10 documents that best match the query.
- Rank them in order of relevance, with the most relevant document first.
- If two or more documents have very similar relevance, choose the document with the smaller number.
- Do not provide explanations or additional information. Only output the document numbers, ordered by relevance, separated by spaces.

Provide only the document numbers (1-{len(stories_list)}), separated by spaces.
"""
    return text


import torch

from langchain_community.chat_models import ChatTongyi



def qwen_Generator(prompt):
    tongyi_chat = ChatTongyi(
        model="qwen2-72b-instruct",
        api_key="XXXXXXXX",
        max_new_tokens=50,
        min_new_tokens=1,
        do_sample=False,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )


    response = tongyi_chat.invoke(prompt)
    content = response.content

    # print("API Response:", content)
    return content




import re



def top20(bm25_searcher, question, psg_collections):

    stories_list = []
    
    hits_20 = bm25_searcher.search(question, k=30)  # 搜索 top 20 文档
    # print("hits_20:", hits_20)
    for hit in hits:
        docid = hit.docid  
        score = hit.score   
        # print(f"DocID: {docid}, Score: {score}")

    for idx, hit in enumerate(hits_20, start=1):
        docid = hit.docid 
        score = hit.score
        
        doc = psg_collections[int(docid)]  
        story = doc[1].strip() 
      
        stories_list.append(f"{idx}. {story}")  
    

    # for story in stories_list:
    #     print(story)
    
    prompt = get_llm_prompt(question, stories_list)


    ans = qwen_Generator(prompt)
    # print("ans:",ans)


    numbers = [int(num) for num in re.findall(r"\d+", ans)]
    # print("numbers:",numbers)
    
    hits_10 = [hits_20[i - 1] for i in numbers]
    # print("hits_10:",hits_10)
    return hits_10
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--eval_bm25", action="store_true")
    parser.add_argument("--load_ctx", action="store_false")
    parser.add_argument("--num_ctx_gen", type=int, default=15)
    parser.add_argument("--num_ctx_use", type=int, default=4)
    parser.add_argument("--num_demo", type=int, default=15)
    parser.add_argument("--llm", type=str, default="chatgpt", choices=["chatgpt", "gpt3"])
    parser.add_argument("--prompt_prefix", type=str, default="pro2")
    parser.add_argument("--comode", type=str, default="rmdemo")
    parser.add_argument("--demo_type", type=str, default="")
    parser.add_argument("--eval_trec_mode", type=str, default='dl19-passage')

    args = parser.parse_args()

    model_prefix = f'{args.num_ctx_use}ctx-{args.comode}'
    demo_type = f'-{args.num_demo}bm2-p1' if 'demo' in args.comode else ''

    # -------------------------------------------------------------------------------------------------------------------------
    setup_seed(args.seed)


    bm25_searcher, psg_collections = model_init(args.comode)
    topics, qrels = load_topics_qrels(args.eval_trec_mode)
    all_qids = [qid for qid in tqdm(topics) if qid in qrels]
    print("all_qids的数量:", len(all_qids))

    import random


    random.seed(args.seed) 

    if args.eval_bm25:
        all_hits = []

        for qid in tqdm(all_qids):

            hits = bm25_searcher.search(topics[qid]['title'], k=10)
            # print("hits:",hits)
            # for hit in hits:
            #     docid = hit.docid   
            #     score = hit.score 
                # print(f"DocID: {docid}, Score: {score}")
            time.sleep(10)
            processed_hits = top20(bm25_searcher, topics[qid]['title'], psg_collections)

            all_hits.append(processed_hits)
            

        eval_trec(all_qids, all_hits, args.eval_trec_mode, eval_prefix='bm25')
    # -----------------------------------------------------------------------------------------------------------------------
