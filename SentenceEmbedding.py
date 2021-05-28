#!/usr/bin/env python
# coding: utf-8

# In[1]:


from splitintosentences import split_into_sentences
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np 
import time
import os, sys
import pandas as pd
PRE_TRAINED_MODEL = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL, do_lower_case= False)
model = BertModel.from_pretrained(PRE_TRAINED_MODEL)

def generate_sentence_embedding( sentence ):
  padded_sentence = '[CLF]' + sentence + '[SEP]'
  tokenized_sentence = tokenizer.tokenize( padded_sentence )
  if len( tokenized_sentence ) > 512 :
    tokenized_sentence = tokenized_sentence[:512]

  token_ids = tokenizer.convert_tokens_to_ids( tokenized_sentence )

  segment_ids = [1]* len(token_ids)

  token_tensor = torch.tensor([token_ids])
  segment_tensor = torch.tensor([segment_ids])
  with torch.no_grad():
    encoded_layers, _ = model(token_tensor, segment_tensor )

  token_embeddings = torch.stack(encoded_layers, dim= 0)
  token_embeddings = torch.squeeze(token_embeddings, dim= 1)
  token_embeddings = torch.sum(token_embeddings[-4:,:,:], dim= 0)
  sentence_embedding_sum = torch.sum(token_embeddings, dim= 0)

  del padded_sentence
  del tokenized_sentence
  del token_ids, segment_ids
  del token_tensor
  del segment_tensor
  del encoded_layers
  del token_embeddings

  return sentence_embedding_sum

def generate_embeddings(corpora, inputpath, output_path_para, output_path_docu):
    i = 0
   
    for document_path in corpora:
        with open(document_path, encoding="utf8") as file:
            document = file.read()
        document_id = document_path[len(inputpath)+9:-4]

        if not document or not document_id:
            continue
        
        document_embeddings = torch.zeros(768)
        paragraphs_embeddings = []
        sentence_count= 0
        paragraphs = document.split('\n')
        
        previous_para_embeddings = None
        previous_para_length = None

        for paragraph_index, paragraph in enumerate(paragraphs):
            sentences = split_into_sentences(paragraph)

            current_para_embeddings= torch.zeros(768)

            current_para_length= len(sentences)

            for sentence in sentences:
                sentence_count+=1 
                sentence_embedding= generate_sentence_embedding(sentence)        
                current_para_embeddings.add_(sentence_embedding)
                document_embeddings.add_(sentence_embedding)
                del sentence_embedding, sentence

            if previous_para_embeddings is not None:
                two_para_lengths= previous_para_length + current_para_length
                two_para_embeddings= (previous_para_embeddings + current_para_embeddings)/two_para_lengths
        
                paragraphs_embeddings.append(two_para_embeddings)            
            
            previous_para_embeddings = current_para_embeddings
            previous_para_length = current_para_length
            del sentences
            del paragraph

        del previous_para_embeddings, previous_para_length
        del current_para_embeddings, current_para_length
        del two_para_embeddings
            
        paragraphs_embeddings= torch.stack(paragraphs_embeddings, dim=0)
        document_embeddings= document_embeddings/sentence_count
        document_embeddings= document_embeddings.unsqueeze(0)      
      
        document_file = output_path_docu +'\\docu-embedding-'+document_id+'.pt'
        paragraph_file = output_path_para +'\\para-embedding-'+document_id+'.pt'
        torch.save(document_embeddings, document_file)
        torch.save(paragraphs_embeddings, paragraph_file)
        
        del document_embeddings
        del paragraphs_embeddings
        del document, document_id
        del paragraphs   

