import pandas as pd
import re

class Chunk:
    def __init__(self, text:str, episode:str):
        """Constructor for Chunck class"""
        self.text = text
        self.color = ["bleu, bleue, bleues, rouge, rouges, blanche, blanches, blanc"]
        self.space = "{200,}"
        self.regex = f"(?=(question {self.color}.{self.space}?)question)"
        self.episode = episode
        self.chunk = self.chunk_text()

    def chunk_text(self):
        """Chunck text by question blocs with regex rules"""
        indexes = [0]
        result = []
        for match in re.finditer(self.regex, self.text):
            indexes.append(match.start())
            for i in range(1,len(indexes)) :
                start = indexes[i-1]
                end = indexes[i]
                chunk = self.text[start:end]
                tmp = {"episode":self.episode,"chunk_start":start, "chunk_end":end, "chunk":chunk}
            result.append(tmp)
        final_chunk = {"episode":self.episode,"chunk_start":end, "chunk_end":len(self.text), "chunk":self.text[end:len(self.text)]}
        result.append(final_chunk)
        data = pd.DataFrame(result, columns=['episode','chunk_start','chunk_end','chunk']).reset_index().rename(columns={'index':'chunk_id'})
        return data

    def chunk_dict(self):
        data = self.chunk_text()
        chunk_info = []
        for index, row in data.iterrows():
            chunk_dict = {"chunk_id":row["chunk_id"],"chunk_start":row["chunk_start"],"chunk_end":row["chunk_end"] }
            chunk_info.append(chunk_dict)
        return chunk_info
