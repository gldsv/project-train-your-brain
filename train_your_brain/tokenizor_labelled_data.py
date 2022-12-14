import re
import json
import os
import pandas as pd
from transformers import AutoTokenizer
from chunk_text import Chunk
from label_extract import LabelledData
from create_chunk import CreateChunk


class Tokenizor():

    def __init__(self, url:str, model = "camembert-base"):
        #load tokenizer from HF
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        # Call Function
        self.url = url
        self.labelled = LabelledData(self.url)
        self.output , self.raw_text , self.nb_episode = self.labelled.extract_labels_from_json()
        self.labels_json=json.loads(self.output.to_json(force_ascii=False,orient='records'))
        self.mapping = {
            "B-Annonce_Question":1,
            "I-Annonce_Question":2,
            "B-Question":3,
            "I-Question":4,
            "B-Bonne_Reponse":5,
            "I-Bonne_Reponse":6,
            "B-Confirmation_Reponse":6,
            "I-Confirmation_Reponse":6,
            "B-Mauvaise_Reponse":7,
            "I-Mauvaise_Reponse":8,
            "B-To_delete":0,
            "I-To_delete":0,
            "B-Blabla":0,
            "I-Blabla":0
            }

    def tokenizor(self,labels_json,tokenizer):

        json_traité=[]
        for j_ in labels_json:
            if j_['label']in ['Episode Start','Episode End']:
                continue
            j=j_.copy()
            tokens_output=tokenizer(
                j['text'],
                return_attention_mask = True,
                return_token_type_ids = False,
                return_offsets_mapping=True,
                padding = False,
    #           max_length = 256,
    #           truncation = True, --> pas de padding à ce stade
                add_special_tokens=False) ## WARNING this removes special tokens NEEDED for BERT & co (101 and 102)
            word_ids = tokens_output.word_ids()

            j['input_ids'] = tokens_output['input_ids']
            j['attention_mask'] = tokens_output['attention_mask']
            j['offset_mapping'] = [(t1+int(j['label_start']),t2+int(j['label_start'])) for t1,t2 in tokens_output['offset_mapping']]
            j['length'] = len(j['text'].replace('-',' ').replace("'"," ").split(' '))
            j['labels'] = j['label'].replace(' ','_')
            j['labels_matched'] = []
            for i,w in enumerate(word_ids):
                if w==None: #or w==word_ids[i-1]:
                    j['labels_matched'].append(0)
                    continue
                else:
                    if w==0:
                        j['labels_matched'].append(self.mapping.get(f"B-{j['labels']}"))
                    else:
                        j['labels_matched'].append(self.mapping.get(f"I-{j['labels']}"))

    #         j['episode'] = j['']
            json_traité.append(j)
        return json_traité


    def labelled_data_extract(self):
        storage = []
        final_result = []
        for text in self.raw_text:
            chunk = Chunk(text = text[0], episode= text[1])
            final_result.append([text[1], chunk.chunk_dict()])
        for episode in final_result:
            output_ = self.output.query(f'episode == {episode[0]}')
            labels_json=json.loads(output_.to_json(force_ascii=False,orient='records'))
            res = self.tokenizor(labels_json,self.tokenizer)
            chunk_mapping_list=episode[1]
            storage.append(CreateChunk(self.tokenizer ,512).__call__(sample_full=res,chunk_mapping_list=chunk_mapping_list))
        df=pd.concat(storage)
        return df


if __name__ == "__main__":
    #Update list to_preprocess to have json output
    to_preprocess = ['train_data', 'test_data']
    for i in to_preprocess :
        url = os.path.join(f'./exchange/{i}')
        tmp = Tokenizor( url=url)
        df = tmp.labelled_data_extract()
        json_files_list = LabelledData(url).get_json_list()
        df.to_csv(f"{url}/label_{i}_tokenized.csv")
        df.to_json(f'{url}/label_{i}_tokenized.json',force_ascii=False,orient='records')
        print(f"✅ Labellised {i}, chunked and tokenized ")
        print(f"ℹ️ Json List for {i} : {json_files_list}")
