import requests
import json
import pandas as pd
import argparse
import collections
import numpy as np
import re
import string
import sys
import math
import os
import time
from textwrap import dedent
from PIL import Image
from nltk.translate import meteor_score as ms
from rouge_score import rouge_scorer
from bs4 import BeautifulSoup
import requests
import nltk
import gc
import torch
from nltk.translate import bleu_score
import numpy as np
from simhash import Simhash
from bleurt import score
import string
import collections
import matplotlib
import difflib
from datasets import load_dataset
import warnings
import itertools
import ast
from elasticsearch import Elasticsearch, exceptions
warnings.filterwarnings('ignore')
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.autograph.set_verbosity(3)

import warnings

warnings.filterwarnings('ignore')


from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError

es_client = Elasticsearch(
    ["http://10.224.164.48:9200"]
)


try:
    response = es_client.ml.get_trained_model_deployment_stats(model_id=".elser_model_1")
    print(response)  
except Exception as e:
    print(f"Error retrieving model deployment stats: {e}")

try:
    es_client.ml.start_trained_model_deployment(model_id=".elser_model_1")
    print("Model started successfully.")
except exceptions.ConflictError:
    print("Model is already running.")
except Exception as e:
    print(f"An error occurred while starting the model: {e}")



try:

    response = es_client.info()
    print(response)
except RequestError as e:
    print(f"RequestError: {e.info}")
except Exception as e:
    print(f"An error occurred: {e}")

## Downloading methods

from sentence_transformers import SentenceTransformer, util

model1 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model3 = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
from transformers import BertTokenizer, BertModel


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model2 = BertModel.from_pretrained("bert-base-uncased")

from transformers import AutoTokenizer, AutoModel

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer1 = AutoTokenizer.from_pretrained("google-t5/t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def bert_score(reference, candidate, return_similarity_matrix=False):
    # Load the BERT tokenizer and model
    # Tokenize the input text
    ref_tokens = tokenizer(reference, return_tensors="pt", add_special_tokens=False)
    can_tokens = tokenizer(candidate, return_tensors="pt", add_special_tokens=False)
    # Get the BERT embeddings
    model2.eval()
    with torch.no_grad():
        ref_outputs = model2(**ref_tokens)
        ref_embeddings = ref_outputs.last_hidden_state[0]
        can_outputs = model2(**can_tokens)
        can_embeddings = can_outputs.last_hidden_state[0]
    # Compute cosine similarities
    cosine_similarities = np.zeros((can_embeddings.shape[0], ref_embeddings.shape[0]))
    for i, c in enumerate(can_embeddings):
        for j, r in enumerate(ref_embeddings):
            cosine_similarities[i, j] = cosine_similarity(c, r)
    # Align cosine similarities
    max_similarities = cosine_similarities.max(axis=1)
    # Average similarity scores
    bertscore = max_similarities.mean()
    if return_similarity_matrix:
        return bertscore, cosine_similarities
    else:
        return bertscore




def sentence_similarity(ideal_answer, generated_answer):
    embedding_1 = model1.encode(ideal_answer, convert_to_tensor=True)
    embedding_2 = model1.encode(generated_answer, convert_to_tensor=True)
    sim_score = util.pytorch_cos_sim(embedding_1, embedding_2)
    sim_score = sim_score.cpu().numpy()[0][0]  # Move tensor to CPU and then convert to NumPy array
    return sim_score



def sentence_pSimilarity(ideal_answer, generated_answer):
    embedding_1 = model3.encode(ideal_answer, convert_to_tensor=True)
    embedding_2 = model3.encode(generated_answer, convert_to_tensor=True)
    sim_score = util.pytorch_cos_sim(embedding_1, embedding_2)
    sim_score = sim_score.cpu().numpy()[0][0]  # Move tensor to CPU and then convert to NumPy array
    return sim_score

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def Sim_hash(ideal_answer, generated_answer):
    return Simhash(generated_answer).distance(Simhash(ideal_answer))


def calculate_perplexity(ideal_answer, answer):
    answer_tokens = answer.strip().split()
    ideal_tokens = ideal_answer.strip().split()

    # Build a frequency distribution of ideal tokens
    token_frequency = {}
    total_tokens = 0
    for token in ideal_tokens:
        token_frequency[token] = token_frequency.get(token, 0) + 1
        total_tokens += 1

    # Calculate perplexity
    log_sum = 0
    for token in answer_tokens:
        frequency = token_frequency.get(token, 0)
        if frequency == 0:
            # Set a small probability for unseen tokens
            probability = 1 / (total_tokens + 1)
        else:
            probability = frequency / total_tokens
        log_sum += math.log2(probability)
    if len(answer_tokens) > 0:
        perplexity = 2 ** (-log_sum / len(answer_tokens))
    else:
        perplexity = 0
    return perplexity


def bleurt_score(ideal_answer, generated_answer):
    checkpoint = "/mnt/workspace/rag/RAG/Blended-RAG/code/checkpoints/bleurt-large-128"
    scorer = score.BleurtScorer(checkpoint)
    scores = scorer.score(references=[generated_answer], candidates=[ideal_answer])
    assert isinstance(scores, list) and len(scores) == 1
    return scores[0]


def blue(answer, ideal_answer):
    generated_tokens = nltk.word_tokenize(answer)
    reference_token_lists = [nltk.word_tokenize(answer) for answer in [ideal_answer]]
    bleu_score = nltk.translate.bleu_score.sentence_bleu(reference_token_lists, generated_tokens)
    return bleu_score


def meteor(answer, ideal_answer):
    generated_tokens = nltk.word_tokenize(answer)
    reference_token_lists = [nltk.word_tokenize(answer) for answer in [ideal_answer]]
    # Calculate the METEOR score
    meteor_score = ms.meteor_score(reference_token_lists, generated_tokens)
    # Instantiate a ROUGE scorer
    return meteor_score


def rouge(answer, ideal_answer):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    # Calculate the ROUGE score
    score = scorer.score(answer, ideal_answer)
    # Extract the F1 score for ROUGE-1
    rouge_score = score['rouge1'].fmeasure
    return rouge_score


def compute_exact_match_ratio(output_text, gen_query):
    matcher = difflib.SequenceMatcher(None, output_text, gen_query)
    return matcher.ratio()


import re



def remove_numbering(question):

    return re.sub(r'^\d+\.\s*', '', question)

def get_llm_prompt(query, stories_list):

    text = f"""You are given a query and a list of candidate documents. 
Determine which document best answers the query, considering direct, indirect, and implicit information.

Query: "{query}"

Candidate documents:
{chr(10).join(stories_list)}

Select the document that best matches the query.If you absolutely cannot determine a best match, return document number 1 .

Provide only the document number (1-{len(stories_list)}). No explanations, no other output, just the number."""

    return text




def get_prompt(context, question):
    text = f"""    
    Answer the following question based on the information given in the article.Respond as concisely as possible.If there is no good answer, respond with "I don't know."\n\nArticle: {context} \n\nQuestion: {question}\nAnswer:"""

    return text




def get_quePrompt(sentence):
    
    text = f""" 
Generate as many simple, direct, and relevant questions as possible based on the following paragraph. The number of questions should not be limited—each question should focus on key information such as location, time, people, or events. Ensure that the answers are explicitly present in the paragraph, and keep the questions as concise as possible.

Examples:
1. Who was the Norse leader?
2. In what year did the battle take place?
3. Who attended the meeting?

Use these examples as a reference to generate as many short, direct questions as possible where the answers can be easily found within the paragraph.

Paragraph: "{sentence}"
"""
    return text


import spacy


nlp = spacy.load("en_core_web_sm")



def segmentation(text):

    doc = nlp(text)

    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences



import torch

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
import re
from langchain import PromptTemplate, LLMChain

from langchain_community.chat_models import ChatTongyi





def qwen_Generator(prompt):
    tongyi_chat = ChatTongyi(
        model="llama3.1-405b-instruct",
        api_key="xxxxxxxxxxxxxxx",
        max_new_tokens=50,
        min_new_tokens=1,
        do_sample=False,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )


    response = tongyi_chat.invoke(prompt)
    content = response.content

    print("API Response:", content)
    return content



def qwen_queGenerator(paragraph):
    tongyi_chat = ChatTongyi(
        model="llama3.1-405b-instruct",
        api_key="xxxxxxxxxxxxxxx",
        max_new_tokens=50,
        min_new_tokens=1,
        do_sample=False,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )


    prompt = get_quePrompt(paragraph)

    try:

        response = tongyi_chat.invoke(prompt)
        content = response.content

        # print("API Response:", content)

        # 确保 response 是字符串并处理
        if isinstance(content, str):

            result = [q.strip() for q in content.split('\n') if q.strip()]
        else:

            result = [content]

        return result

    except Exception as e:
        print(f"Error in qwen_queGenerator: {e}")
        return []




def final_generator(context, question): 
    tongyi_chat = ChatTongyi(
        model="llama3.1-70b-instruct",
        api_key=" xxxxxxxxxxxxxxx ",
        max_new_tokens=50,
        min_new_tokens=1,
        do_sample=False,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )
    

    prompt = get_prompt(context.replace('`', ''), question)
    
    try:

        response = tongyi_chat.invoke(prompt)
        

        print("API Response:", response)
        

        if isinstance(response, str):
            result = re.sub(r"\s+", " ", response).strip()
        else:
            result = response 

        return result
    
    except Exception as e:

        print("Error occurred during API call:", str(e))
        if 'response' in locals():
            print("Raw Response:", response)  
        return None


def top5(es_client, index, question):

    stories_list = []
    query_embedding = model1.encode(question)
    response = es_client.search(
                        index=index,
                
#                         body={
#                   "query": {
#                    "multi_match" : {
#                       "query":question,
#                         "type":"best_fields",
#                         "fields":[ "story", "title"],
#                         "tie_breaker": 0.3

#                     }
#                   },
#                   "knn": {
#                     "field": "story_embedding",
#                     "query_vector": query_embedding,
#                     "k": 10,
#                     "num_candidates": 100,
#                      "boost": 10
#                   },
#                   #"size": 5
#                 },
            
    return match, stories_list
    
def jugement(question, heuristic_answer):

    prompt = (f"<s>[INST] <<SYS>>\nYou are a helpful assistant. Your task is to parse user input into"
              f" structured formats according to the coarse answer. Current datetime is 2023-12-20 9:47:28"
              f" <</SYS>>\nCoarse answer: (({heuristic_answer}))\nQuestion: (({question})) [/INST]")
    resp = requests.post(url="XXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
                     headers={"Authorization": "XXXXXXXXXXXXXXXXXXXX"},
                     json={"data": [prompt]})
    generated_text = "No Known state found in the output"
    print(generated_text)
    return generated_text    

def ragResult(context,question,gold_answer):
    count_value = 0
    count_value_f1 = 0
    blue_score_count = 0
    meteor_score_count = 0
    rouge_score_count = 0
    sentence_similarity_score_count = 0
    Sim_hash_score_count = 0
    bleurt_score1_count = 0
    bert_score1_count = 0
    perplexity_score_count = 0
    # context = doc['_source']['story']
    time.sleep(3) 
    ans = final_generator(context, question).content
    print("gold_answer:",gold_answer)
    print("ans:",ans)
    value = compute_exact_match_ratio(gold_answer,ans)
    count_value = count_value+value
        
    value_f1 = compute_f1(gold_answer,ans)
    count_value_f1 = count_value_f1+value_f1
    blue_score =blue(gold_answer, ans)
    blue_score_count = blue_score+blue_score_count
                
    meteor_score =meteor(gold_answer, ans)
    meteor_score_count = meteor_score+meteor_score_count
                
    rouge_score =rouge(gold_answer, ans)
    rouge_score_count = rouge_score+rouge_score_count
                
    sentence_similarity_score =sentence_similarity(gold_answer, ans)
    sentence_similarity_score_count = sentence_similarity_score+sentence_similarity_score_count
                
    Sim_hash_score =Sim_hash(gold_answer, ans)
    Sim_hash_score_count = Sim_hash_score+Sim_hash_score_count
                
    perplexity_score =calculate_perplexity(gold_answer, ans)
    perplexity_score_count = perplexity_score+perplexity_score_count
                
    bleurt_score1 =bleurt_score(gold_answer, ans)
    bleurt_score1_count = bleurt_score1+bleurt_score1_count
    try:
        bert_score1 =bert_score(gold_answer, ans)
        bert_score1_count = bert_score1+bert_score1_count
    except Exception as e:
        print(f"Error calculating BERT score: {e}")
                
    return count_value,count_value_f1,blue_score_count,meteor_score_count,rouge_score_count,sentence_similarity_score_count,Sim_hash_score_count,perplexity_score_count,bleurt_score1_count,bert_score1_count
    


def processESIndex_RAG(df_squad, index):

    df_squad_sampled = df_squad.sample(n=400, random_state=202411)  # 设置随机种子 2024 保持可重复性

    count = 0
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    true_llm_count = 0
    true_similarity_count = 0

    stories_list = []
    squad_count = 0
    question_count = 0
    
    total_count_value = 0
    total_count_value_f1 = 0
    total_blue_score_count = 0
    total_meteor_score_count = 0
    total_rouge_score_count = 0
    total_sentence_similarity_score_count = 0
    total_Sim_hash_score_count = 0
    total_perplexity_score_count = 0
    total_bleurt_score1_count = 0
    total_bert_score1_count = 0


    for ind in df_squad.index:

        data = df_squad.loc[ind]

        dict_obj = ast.literal_eval(data['answers'])

        gold_answer = dict_obj['text'][0]
        question = data['question']
        gold_doc = data['context']
        query_embedding = model1.encode(question)

        question_count += 1
        if 1 <= question_count <= 100:
            print("Questions----", question_count)
            response = es_client.search(
                index=index,
  
#                 body={
#           "query": {
#            "multi_match" : {
#               "query":question,
#                 "type":"best_fields",
#                 "fields":[ "story", "title"],
#                 "tie_breaker": 0.3

#             }
#           },
#           "knn": {
#             "field": "story_embedding",
#             "query_vector": query_embedding,
#             "k": 10,
#             "num_candidates": 100,
#              "boost": 10
#           },
#           #"size": 5
#         },
                # "size": 5,
                size=2  
            )
      
            all_hits = response['hits']['hits']
            # print(all_hits)
            for num, doc in enumerate(all_hits):
                # print("question:",question)
                # print("gold_answer:",gold_answer)
                # print("gold_doc:",gold_doc)
                context = doc['_source']['story']
                # print("doc['_source']['story']:",doc['_source']['story'])

                time.sleep(5)

                # questions = qwen_queGenerator(context)
                if gold_answer in doc['_source']['story']:
                    count += 1
                generated_text = jugement(question, context)


                if "<Known(True)>" in generated_text:
 
                    count1 += 1
                    if gold_answer in doc['_source']['story']:
        
                        true_llm_count += 1
      
                    # time.sleep(6) 
                    questions = qwen_queGenerator(context)
                    questions_cleaned = [remove_numbering(q) for q in questions]
                    # print(questions_cleaned)

                    c_similarity = [sentence_similarity(question, q) for q in questions_cleaned]
                    # print("A语义相似度:",c_similarity)
          

                    s_similarity = [sentence_pSimilarity(question,q) for q in questions_cleaned]
                    # print("P语义相似度:",s_similarity)
                    for num1, num2 in zip(c_similarity, s_similarity):
                        if num1 > 0.4 or num2 > 0.4:
             的
                            count2 += 1
                            if gold_answer in doc['_source']['story']:
                           
                                true_similarity_count += 1
                                context = doc['_source']['story']
                                (count_value, count_value_f1, blue_score_count, meteor_score_count, rouge_score_count, sentence_similarity_score_count, Sim_hash_score_count, perplexity_score_count, bleurt_score1_count, bert_score1_count) = ragResult(context,question,gold_answer)
                       
                                total_count_value += count_value
                                total_count_value_f1 += count_value_f1
                                total_blue_score_count += blue_score_count
                                total_meteor_score_count += meteor_score_count
                                total_rouge_score_count += rouge_score_count
                                total_sentence_similarity_score_count += sentence_similarity_score_count
                                total_Sim_hash_score_count += Sim_hash_score_count
                                total_perplexity_score_count += perplexity_score_count
                                total_bleurt_score1_count += bleurt_score1_count
                                total_bert_score1_count += bert_score1_count
                            break
                    else:
            
                        match, stories_list = top5(es_client, index, question)  
                        if match:
                         引
                            ans_index = int(match.group(1)) - 1
                            # print("ans_index:", ans_index)


                            if 0 <= ans_index < len(stories_list):
                                if gold_answer in stories_list[ans_index]:
                                  
                                    count3 += 1
                                    context = stories_list[ans_index]
                                    (count_value, count_value_f1, blue_score_count, meteor_score_count, rouge_score_count, sentence_similarity_score_count, Sim_hash_score_count, perplexity_score_count, bleurt_score1_count, bert_score1_count) = ragResult(context,question,gold_answer)
                          
                                    total_count_value += count_value
                                    total_count_value_f1 += count_value_f1
                                    total_blue_score_count += blue_score_count
                                    total_meteor_score_count += meteor_score_count
                                    total_rouge_score_count += rouge_score_count
                                    total_sentence_similarity_score_count += sentence_similarity_score_count
                                    total_Sim_hash_score_count += Sim_hash_score_count
                                    total_perplexity_score_count += perplexity_score_count
                                    total_bleurt_score1_count += bleurt_score1_count
                                    total_bert_score1_count += bert_score1_count
                            else:
                                print(f"Warning: The answer index {ans_index + 1} is out of range.")
                        else:
                            print(f"Error: The generated answer '{ans_index}' does not contain a valid integer.")
                            

                elif "<Known(False)>" in generated_text:
              
                    match, stories_list = top5(es_client, index, question)  
                    if match:
                
                        ans_index = int(match.group(1)) - 1
                        # print("ans_index:", ans_index)

                     
                        if 0 <= ans_index < len(stories_list):
                            if gold_answer in stories_list[ans_index]:
                             
                                count4 += 1
                                context = stories_list[ans_index]
                                (count_value, count_value_f1, blue_score_count, meteor_score_count, rouge_score_count, sentence_similarity_score_count, Sim_hash_score_count, perplexity_score_count, bleurt_score1_count, bert_score1_count) = ragResult(context,question,gold_answer)
                  
                                total_count_value += count_value
                                total_count_value_f1 += count_value_f1
                                total_blue_score_count += blue_score_count
                                total_meteor_score_count += meteor_score_count
                                total_rouge_score_count += rouge_score_count
                                total_sentence_similarity_score_count += sentence_similarity_score_count
                                total_Sim_hash_score_count += Sim_hash_score_count
                                total_perplexity_score_count += perplexity_score_count
                                total_bleurt_score1_count += bleurt_score1_count
                                total_bert_score1_count += bert_score1_count
                        else:
                            print(f"Warning: The answer index {ans_index + 1} is out of range.")
                    else:
                        print(f"Error: The generated answer '{ans_index}' does not contain a valid integer.")
     question_count = 100
    print(count1)
    print( true_llm_count)
    print(count3)
    print( count4)
    print(true_similarity_count)
    print(count)
    print( count3+count4+true_similarity_count)

    print("Count value ----",total_count_value ,"F1----",total_count_value_f1)
    print("Avg EM Accuracy",total_count_value/question_count)
    print("Avg f1 Accuracy",total_count_value_f1/question_count)
    print("Avg blue_score Accuracy",total_blue_score_count/question_count)
    print("Avg meteor_score Accuracy",total_meteor_score_count/question_count)
    print("Avg rouge_score Accuracy",total_rouge_score_count/question_count)
    print("Avg sentence_similarity_score Accuracy",total_sentence_similarity_score_count/question_count)
    print("Avg Sim_hash_score_count Accuracy",total_Sim_hash_score_count/question_count)
    print("Avg perplexity_score_count Accuracy",total_perplexity_score_count/question_count)
    print("Avg bleurt_score1_count Accuracy",total_bleurt_score1_count/question_count)
    print("Avg bert_score1_count Accuracy",total_bert_score1_count/question_count)

index_name_knn = "research_index_knn_squad"
index_name = "research_index_bm25_squad"
index_name_elser = "research_index_elser_squad"

validation_dataset = pd.read_csv("/mnt/workspace/rag/RAG/date/filtered_data.csv")
df_questions = pd.DataFrame(validation_dataset)

processESIndex_RAG(df_questions, index_name_knn)

try:
    es_client.ml.stop_trained_model_deployment(model_id=".elser_model_1")
    print("Model stopped successfully.")
except exceptions.NotFoundError:
    print("Model is not found or is already stopped.")
except Exception as e:
    print(f"An error occurred while stopping the model: {e}")