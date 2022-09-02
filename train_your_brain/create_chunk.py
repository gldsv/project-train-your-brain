#import pandas as pd
#class CreateChunk:
#    def __init__(self, tokenizer, max_len,):
#        self.tokenizer = tokenizer
#        self.max_len = max_len-2 #to account for special tokens cls and sep to be added
#
#
#    def baby_chunkator(self, batch):
#        output = dict()
#        output["ids"] = [s for sample in batch for s in sample["input_ids"]]
#        output["mask"] = [s for sample in batch for s in sample["attention_mask"]]
#        output["targets"] = [s for sample in batch for s in sample["labels_matched"]]
#        output['episode_id'] = set([sample['Episode'] for sample in batch])
#        output['chunk_id'] = set([sample['chunk_id'] for sample in batch])
#        output['offset_mapping']=[s for sample in batch for s in sample["offset_mapping"]]
#
#        #truncation couic
#        if len(output['ids']) > self.max_len :
#            output['ids'] = output['ids'][: self.max_len]
#            output['mask'] = output['mask'][: self.max_len]
#            output['targets'] = output['targets'][: self.max_len]
#            output['offset_mapping'] = output['offset_mapping'][: self.max_len]
#
#        # add special tokens
#        output['ids'] = [self.tokenizer.cls_token_id] + output["ids"] +[self.tokenizer.sep_token_id]
#        output["mask"] = [1] + output['mask'] + [1]
#        output["targets"] = [0] + output['targets'] + [0]
#        output["offset_mapping"] = [[0,0]] + output['offset_mapping'] + [[0,0]]
#
#
#        # add padding
#        if self.tokenizer.padding_side == "right":
#            output["ids"] = output["ids"] + (self.max_len+2 - len(output["ids"])) * [self.tokenizer.pad_token_id]
#            output["mask"] = output["mask"] + (self.max_len+2 - len(output["mask"])) * [0]
#            output['targets'] = output['targets'] +(self.max_len+2 - len(output["targets"])) * [0]
#            output['offset_mapping'] = output['offset_mapping'] +(self.max_len+2 - len(output["offset_mapping"])) * [[0,0]]
#        else:
#            output["ids"] = (self.max_len+2 - len(output["ids"])) * [self.tokenizer.pad_token_id] + output["ids"]
#            output["mask"] = (self.max_len+2 - len(output["mask"])) * [0] + output["mask"]
#            output['targets'] = (self.max_len+2 - len(output["targets"])) * [0] + output['targets']
#            output['offset_mapping'] = (self.max_len+2 - len(output["offset_mapping"])) * [[0,0]] + output['offset_mapping']
#
#
##          convert to tensors
##         output["ids"] = tf.convert_to_tensor(output["ids"], dtype=tf.int32)
##         output["mask"] = tf.convert_to_tensor(output["mask"], dtype=tf.int32)
##         output["targets"] = tf.convert_to_tensor(output["targets"], dtype=tf.int32)
#        return output
#
#    def __call__(self,sample_full,chunk_mapping_list):
#        sample_full_df = pd.DataFrame(sample_full).sort_values('Label_Start').reset_index()
#        chunks = []
#        for i,chunk in enumerate(chunk_mapping_list):
#            if i<len(chunk_mapping_list)-1:
#                sample_extract = sample_full_df[sample_full_df['Label_Start']<chunk['chunk_end']].copy()
#            else:
#                sample_extract = sample_full_df
#            sample_extract['chunk_id']=chunk['chunk_id']
#            sample_full_df=sample_full_df.drop(sample_extract.index)
#            chunk = self.baby_chunkator(sample_extract.to_dict(orient='records'))
#            chunks.append(chunk)
#        return pd.DataFrame(chunks).to_json(orient='records')




class CreateChunk:
    def __init__(self, tokenizer, max_len,):
        self.tokenizer = tokenizer
        self.max_len = max_len-2 #to account for special tokens cls and sep to be added


    def __call__(self, batch):
        output = dict()
        output["ids"] = [s for sample in batch for s in sample["input_ids"]]
        output["mask"] = [s for sample in batch for s in sample["attention_mask"]]
        output["targets"] = [s for sample in batch for s in sample["labels_matched"]]
        output['episode_id'] = set([sample['episode'] for sample in batch])
        output['chunk_id'] = set([sample['chunk_id'] for sample in batch])
        output['offset_mapping']=[s for sample in batch for s in sample["offset_mapping"]]

        #truncation couic
        if len(output['ids']) > self.max_len :
            output['ids'] = output['ids'][: self.max_len]
            output['mask'] = output['mask'][: self.max_len]
            output['targets'] = output['targets'][: self.max_len]
            output['offset_mapping'] = output['offset_mapping'][: self.max_len]

        # add special tokens
        output['ids'] = [self.tokenizer.cls_token_id] + output["ids"] +[self.tokenizer.sep_token_id]
        output["mask"] = [1] + output['mask'] + [1]
        output["targets"] = [0] + output['targets'] + [0]
        output["offset_mapping"] = [[0,0]] + output['offset_mapping'] + [[0,0]]


        # add padding
        if self.tokenizer.padding_side == "right":
            output["ids"] = output["ids"] + (self.max_len+2 - len(output["ids"])) * [self.tokenizer.pad_token_id]
            output["mask"] = output["mask"] + (self.max_len+2 - len(output["mask"])) * [0]
            output['targets'] = output['targets'] +(self.max_len+2 - len(output["targets"])) * [0] #needed??
            output['offset_mapping'] = output['offset_mapping'] +(self.max_len+2 - len(output["offset_mapping"])) * [[0,0]]

        else:
            output["ids"] = (self.max_len+2 - len(output["ids"])) * [self.tokenizer.pad_token_id] + output["ids"]
            output["mask"] = (self.max_len+2 - len(output["mask"])) * [0] + output["mask"]
            output['targets'] = (self.max_len+2 - len(output["targets"])) * [0] + output['targets']#needed??
            output['offset_mapping'] = (self.max_len+2 - len(output["offset_mapping"])) * [[0,0]] + output['offset_mapping']


         #convert to tensors
#         output["ids"] = torch.tensor(output["ids"], dtype=torch.long)
#         output["mask"] = torch.tensor(output["mask"], dtype=torch.long)
#         output["targets"] = torch.tensor(output["targets"], dtype=torch.float)
        return output
