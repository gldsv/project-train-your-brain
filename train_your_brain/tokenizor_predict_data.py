import json
import os
import pandas as pd
from transformers import AutoTokenizer
from chunk_text import Chunk
from create_chunk import CreateChunk


class Tokenizor_prediction():

    def __init__(self, url:str ,episode:str,chunked_text, model = "camembert-base" ):
        #load tokenizer from HF
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        # Call Function
        self.url = url
        self.episode = episode
        self.chunked_text_df , self.chunked_text_dict = chunked_text
        self.chunked_text_json=json.loads(self.chunked_text_df.to_json(force_ascii=False,orient='records'))


    def tokenizor_prediction(self,chunked_text_json,tokenizer):
        ''' Preporcess chuncked data for tokenization'''
        json_traité=[]
        for j_ in chunked_text_json:
            j=j_.copy()
            tokens_output=tokenizer(
                j['chunk'],
                return_attention_mask = True,
                return_token_type_ids = False,
                return_offsets_mapping=True,
                padding = False,
                add_special_tokens=False) ## WARNING this removes special tokens NEEDED for BERT & co (101 and 102)
            word_ids = tokens_output.word_ids()

            j['input_ids'] = tokens_output['input_ids']
            j['attention_mask'] = tokens_output['attention_mask']
            j['offset_mapping'] = [(t1+int(j['chunk_start']),t2+int(j['chunk_start'])) for t1,t2 in tokens_output['offset_mapping']]
            j['length'] = len(j['chunk'].replace('-',' ').replace("'"," ").split(' '))

            json_traité.append(j)

        return json_traité



    def prediction_data_extract(self):
        ''' Tokenize data - Return DataFrame '''
        storage = []
        chunk_mapping_list=self.chunked_text_dict
        res = self.tokenizor_prediction(self.chunked_text_json,self.tokenizer)
        for i in res :
            storage.append(CreateChunk(self.tokenizer ,512).prediction_token_output(res, chunk_mapping_list))
        df=pd.concat(storage)
        return df
