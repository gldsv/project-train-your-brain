import pandas as pd
import re

class Chunk:
    def __init__(self, text:str):
        """Constructor for Chunk class"""
        self.text = text
        self.color = ["bleu, bleue, bleues, rouge, rouges, blanche, blanches, blanc"]
        self.space = "{200,}"
        self.regex = f"(?=(question {self.color}.{self.space}?)question)"
        self.chunk = self.chunk_text()

    def chunk_text(self):
        """Chunk text by question blocs with regex rules"""
        questions = re.findall(self.regex, self.text, flags=re.IGNORECASE)
        return questions
