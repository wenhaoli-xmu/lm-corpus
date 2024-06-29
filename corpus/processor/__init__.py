from .proc_concat import ConcatProcessor
from .proc_conv import ConversationProcessor
import json


def get_processor(path, *args, **kwargs):
    with open(path, 'r') as f:
        config = json.load(f)

    if "concat" in config:
        return ConcatProcessor(path, *args, **kwargs)
    elif "conversation" in config:
        return ConversationProcessor(path, *args, **kwargs)
    else:
        raise NotImplementedError