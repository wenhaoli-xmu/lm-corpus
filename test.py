from corpus import *
from transformers import AutoTokenizer


tok = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_fast=True)


proc = get_processor("dataconfig.json", tok)
corp = Corpus("longalpaca.json", proc)
import IPython
IPython.embed(header='debug')