from dataclasses import dataclass
from collections import OrderedDict
import copy
from typing import Optional, List, Dict
import json
from .proc_base import BasicProcessor


"""
{
    "concat": {
        "input": {
            "trunc_rear": false,
            "trunc_txt": 96,
            "train": false
        },
        "output": {
            "trunc_rear": true,
            "trunc_txt": 96,
            "train": true
        }
    },
    "truncation": {
        "enable": true,
        "max_tokens": 16384,
        "order": ["input", "output"]
    }
}
"""


@dataclass
class Keyword:
    trunc_rear: bool
    trunc_txt: Optional[int]
    train: bool


@dataclass
class TruncationConfig:
    enable: bool
    max_tokens: Optional[int]
    order: Optional[List[str]]


@dataclass
class ConcatProcessorConfig:
    concat: Dict[str, Keyword]
    truncation: TruncationConfig


class ConcatProcessor(BasicProcessor):
    def create_config(self):
        with open(self.path, 'r') as f:
            config = json.load(f)

        keyword_config = {k: Keyword(**v) for k, v in config['concat'].items()}
        truncation_config = TruncationConfig(**config['truncation'])
        return ConcatProcessorConfig(
            concat=keyword_config, 
            truncation=truncation_config)


    def process(self, instance):
        result = OrderedDict()
        num_tokens = 0

        for key in self.config.concat.keys():

            if key not in instance.keys():
                raise ValueError

            concat = self.config.concat[key]

            # text level truncation
            text = instance[key]
            if concat.trunc_txt is not None:
                text = (
                    text[:concat.trunc_txt * 1024] 
                    if concat.trunc_rear 
                    else text[-concat.trunc_txt * 1024:])
            
            # convert to tokens
            input_ids = self.tokenizer(text, add_special_tokens=False).input_ids

            labels = copy.deepcopy(input_ids) if concat.train else [-100] * len(input_ids)
            result[key] = {
                "input_ids": input_ids,
                "labels": labels,
                "trunc_rear": concat.trunc_rear}
            num_tokens += len(input_ids)

        # final truncation
        if self.config.truncation.enable:
            if (exceed := (num_tokens - self.config.truncation.max_tokens)) > 0:

                for key in self.config.truncation.order:
                    num_prune = min(len(result[key]['input_ids']), exceed)
                    num_remain = len(result[key]['input_ids']) - num_prune

                    if num_prune == len(result[key]['input_ids']):
                        result[key]['input_ids'] = []
                        result[key]['input_ids'] = []
                        continue

                    if result[key]['trunc_rear']:
                        result[key]['input_ids'] = result[key]['input_ids'][:num_remain]
                        result[key]['labels'] = result[key]['labels'][:num_remain]
                    else:
                        result[key]['input_ids'] = result[key]['input_ids'][-num_remain:]
                        result[key]['labels'] = result[key]['labels'][-num_remain:]
                    exceed -= num_prune

                    if exceed == 0:
                        break

        input_ids = []
        labels = []
        for _, value in result.items():
            input_ids += value['input_ids']
            labels += value['labels']
        attention_mask = [0] * len(input_ids)

        # padding
        input_ids, labels, attention_mask = self.padding(
            input_ids, labels, attention_mask)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }
