import time
import numpy as np
import pandas as pd
import torch
import faiss
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
from unstructured.partition.html import partition_html

class WebQA:
    
    def __init__(self, model=None, tokenizer=None, embedding_model_name=None, url=None,):
        self.model = None
        self.tokenizer = None
        self.embedding_model = None
        self.context = None
        self.index = None
        self.url = None
        
        if all(arg is not None for arg in (model, tokenizer, embedding_model_name,url)):
            self.set_model(model)
            self.set_tokenizer(tokenizer)
            self.set_embedding_model(embedding_model_name)
            self.load_context(url)
            self.set_index(self.prepare_sentences_vector(self.get_embeddings(self.embedding_model)))
        
    def set_model(self, model):
        self.model = AutoModelForQuestionAnswering.from_pretrained(model)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        
    def set_embedding_model(self, model):
        self.embedding_model = SentenceTransformer(model)

    def load_context(self, url):
        elements = partition_html(url=url)
        context = [str(element) for element in elements if len(str(element)) > 60]
        self.context = context
        print('Load context done')

    def set_index(self):
        vector = self.get_embeddings(self.context)
        index = faiss.IndexFlatL2(vector.shape[1])
        index.add(vector)
        self.index = index

    def get_embeddings(self, text_list):
        return self.embedding_model.encode(text_list)

    def prepare_sentences_vector(self, encoded_list):
        encoded_list = [i.reshape(1, -1) for i in encoded_list]
        encoded_list = np.vstack(encoded_list).astype('float32')
        encoded_list = normalize(encoded_list)
        return encoded_list
    
    def faiss_search(self, index, question_vector, k=1):
        distances, indices = index.search(question_vector, k)
        return distances, indices

    def model_pipeline(self, question, context):
        inputs = self.tokenizer(question, context, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
        Answer = self.tokenizer.decode(predict_answer_tokens)
        return Answer.replace('<unk>', '@')

    def predict_test(self, context, question, index, url):
        t = time.time()
        question = question.strip()
        question_vector = self.get_embeddings([question])
        question_vector = self.prepare_sentences_vector(question_vector)
        _, indices = self.faiss_search(index, question_vector, 3)

        most_similar_contexts = ''
        for i in range(3):
            most_sim_context = context[indices[0][i]].strip()
            answer_url = f"{url}#:~:text={most_sim_context}"
            most_similar_contexts += f'<a href="{answer_url}">[ {i+1} ]: {most_sim_context}</a>\n\n'
        return most_similar_contexts
    
    def chat_interface(self, question):
        response = self.predict_test(self.set_embedding_model, context, question, index, url)
        return response
