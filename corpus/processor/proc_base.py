from abc import ABC, abstractmethod
from typing import Union, List


class BasicProcessor(ABC):
    def __init__(self, path, tokenizer, pad_side='left', pad_length=None):
        self.path = path
        self.tokenizer = tokenizer
        self.pad_side = pad_side
        self.pad_length = pad_length
        self.config = self.create_config()

        with open(path, 'r') as f:
            self.signature = f"{f.read()}/{tokenizer.__class__.__name__}/{pad_side}/{pad_length}"


    @abstractmethod
    def create_config(self):
        pass


    @abstractmethod
    def process(self, instance: dict) -> Union[List, List, List]:
        pass


    def padding(self, input_ids, labels, attention_mask):        
        if self.pad_length is not None:
            remain = self.pad_length - len(input_ids)
            if remain < 0:
                raise ValueError

            if self.pad_side == 'left':
                input_ids = [self.tokenizer.pad_token_id] * remain + input_ids
                labels = [-100] * remain + labels
                attention_mask = [1] * remain + attention_mask
            elif self.pad_side == 'right':
                input_ids = input_ids + [self.tokenizer.pad_token_id] * remain
                labels = labels + [-100] * remain
                attention_mask = attention_mask + [1] * remain
            else: raise NotImplementedError

        return input_ids, labels, attention_mask
