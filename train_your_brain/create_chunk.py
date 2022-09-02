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

        #truncation couic
        if len(output['ids']) > self.max_len :
            output['ids'] = output['ids'][: self.max_len]
            output['mask'] = output['mask'][: self.max_len]
            output['targets'] = output['targets'][: self.max_len]

        # add special tokens
        output['ids'] = [self.tokenizer.cls_token_id] + output["ids"] +[self.tokenizer.sep_token_id]
        output["mask"] = [1] + output['mask'] + [1]
        output["targets"] = [0] + output['targets'] + [0]


        # add padding
        if self.tokenizer.padding_side == "right":
            output["ids"] = output["ids"] + (self.max_len+2 - len(output["ids"])) * [self.tokenizer.pad_token_id]
            output["mask"] = output["mask"] + (self.max_len+2 - len(output["mask"])) * [0]
            output['targets'] = output['targets'] +(self.max_len+2 - len(output["targets"])) * [0] #needed??
        else:
            output["ids"] = (self.max_len+2 - len(output["ids"])) * [self.tokenizer.pad_token_id] + output["ids"]
            output["mask"] = (self.max_len+2 - len(output["mask"])) * [0] + output["mask"]
            output['targets'] = (self.max_len+2 - len(output["targets"])) * [0] + output['targets']#needed??


         #convert to tensors
#         output["ids"] = torch.tensor(output["ids"], dtype=torch.long)
#         output["mask"] = torch.tensor(output["mask"], dtype=torch.long)
#         output["targets"] = torch.tensor(output["targets"], dtype=torch.float)
        return output
