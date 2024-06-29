import json
import numpy as np
from .utils import corpus_log
from .corpus import BasicCorpus



def stat_corpus(corpus):
    length = {}
    num_instance = 0
    keywords = corpus.data[0]
    for key in keywords:
        length[key] = []

    for data in corpus.data:
        num_instance += 1
        for key, value in data.items():
            if isinstance(value, (str, list)):
                length[key].append(len(value))
            elif isinstance(value, (int, float)):
                length[key].append(value)


    corpus_log(f"num_instance: {num_instance}")
    for keyword, value in keywords.items():
        corpus_log("-" * 40)
        corpus_log(f"{keyword}:")
        corpus_log(f"\ttype: {type(value)}")
        
        if isinstance(value, (str, list)):
            corpus_log(f"\tmax_length: {max(length[keyword])}")
            corpus_log(f"\tmin_length: {min(length[keyword])}")
            corpus_log(f"\tavg_length: {np.mean(length[keyword])}")
        elif isinstance(value, (int, float)):
            corpus_log(f"\tmax: {max(length[keyword])}")
            corpus_log(f"\tmin: {min(length[keyword])}")
            corpus_log(f"\tavg: {np.mean(length[keyword])}")



def stat_json_file(path):

    length = {}
    num_instance = 0

    with open(path, 'r') as f:
        line = f.readline()
        keywords = json.loads(line)

        for key in keywords:
            length[key] = []

        while line:
            if line.strip():
                num_instance += 1
                line = json.loads(line)
                for key, value in line.items():
                    if isinstance(value, (str, list)):
                        length[key].append(len(value))
                    elif isinstance(value, (int, float)):
                        length[key].append(value)

            line = f.readline()

    corpus_log(f"num_instance: {num_instance}")
    for keyword, value in keywords.items():
        corpus_log("-" * 40)
        corpus_log(f"{keyword}:")
        corpus_log(f"\ttype: {type(value)}")
        
        if isinstance(value, (str, list)):
            corpus_log(f"\tmax_length: {max(length[keyword])}")
            corpus_log(f"\tmin_length: {min(length[keyword])}")
            corpus_log(f"\tavg_length: {np.mean(length[keyword])}")
        elif isinstance(value, (int, float)):
            corpus_log(f"\tmax: {max(length[keyword])}")
            corpus_log(f"\tmin: {min(length[keyword])}")
            corpus_log(f"\tavg: {np.mean(length[keyword])}")


def stat(path_or_corpus):
    if isinstance(path_or_corpus, str):
        return stat_json_file(path_or_corpus)
    elif isinstance(path_or_corpus, BasicCorpus):
        return stat_corpus(path_or_corpus)
    else:
        raise NotImplementedError