from .conversations import get_conv_template, SeparatorStyle
from .proc_base import BasicProcessor
from ..utils import corpus_log

from dataclasses import dataclass, field
from typing import List, Dict
import json
import copy


"""
{
    "conversation": {
        "conv_template": "vicuna_v1.1"
        "conv_keyword": "conversations",
        "role_keyword": "role",
        "cont_keyword": "content",

        "roles": {
            "user": 0,      # 0 不可训练
            "assistant": 1  # 1 可训练
        }
    },
    "truncation": {
        "enable": false,
        "max_tokens": 16384
    }
}
"""


@dataclass
class ConversationConfig:
    conv_template: str = field(default=None)
    conv_keyword: str = field(default=None)
    role_keyword: str = field(default=None)
    cont_keyword: str = field(default=None)
    roles: Dict[str, int] = field(default=None) 


@dataclass
class TruncationConfig:
    enable: bool = field(default=False)
    max_tokens: int = field(default=None)


@dataclass
class ConversationProcessorConfig:
    conversation: ConversationConfig = field(default=None)
    truncation: TruncationConfig = field(default=None)


class ConversationProcessor(BasicProcessor):
    def create_config(self):
        with open(self.path, 'r') as f:
            config = json.load(f)

        roles = config["conversation"]["roles"]
        conversation = ConversationConfig(
            conv_template=config["conversation"]["conv_template"],
            conv_keyword=config["conversation"]["conv_keyword"],
            role_keyword=config["conversation"]["role_keyword"],
            cont_keyword=config["conversation"]["cont_keyword"],
            roles=roles)
        truncation = TruncationConfig(
            enable=config["truncation"]["enable"],
            max_tokens=config["truncation"]["max_tokens"])

        config = ConversationProcessorConfig(
            conversation=conversation,
            truncation=truncation)

        return config
    

    def process(self, instance):
        conv_keyword = self.config.conversation.conv_keyword
        role_keyword = self.config.conversation.role_keyword
        cont_keyword = self.config.conversation.cont_keyword
        conv = get_conv_template(self.config.conversation.conv_template)
        source = instance[conv_keyword]
        
        roles = {}
        for role, id in self.config.conversation.roles.items():
            roles.update({role: conv.roles[id]})

        if roles[source[0][role_keyword]] != conv.roles[0]:
            instance = instance[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence[role_keyword]]
            assert role == conv.roles[j % 2]
            conv.append_message(role, sentence[cont_keyword])
        conversation = conv.get_prompt()

        # Tokenize conversations
        input_ids = self.tokenizer(
            conversation,
            max_length=self.config.truncation.max_tokens,
            truncation=self.config.truncation.enable,
        ).input_ids
        target = copy.deepcopy(input_ids)

        assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

        # Mask targets. Only compute loss on the assistant outputs.
        sep = conv.sep + conv.roles[1] + ": "
        total_len = len(target)

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = [-100] * len(target[:cur_len])
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(self.tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2

            # if i != 0 and not self.tokenizer.legacy:
            #     # The legacy and non-legacy modes handle special tokens differently
            #     instruction_len -= 1

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = [-100] * len(target[cur_len : cur_len + instruction_len])
            cur_len += turn_len

            # if i != 0 and not self.tokenizer.legacy:
            #     # The legacy and non-legacy modes handle special tokens differently
            #     cur_len -= 1

        target[cur_len:] = [-100] * len(target[cur_len:])

        if cur_len < self.tokenizer.model_max_length:
            if cur_len != total_len:
                target = [-100] * len(target)
                corpus_log(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)")

        attention_mask = [0] * len(input_ids)                
        input_ids, target, attention_mask = self.padding(
            input_ids, target, attention_mask)

        return dict(
            input_ids=input_ids,
            labels=target,
            attention_mask=attention_mask)