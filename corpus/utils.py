from pygments import console


def corpus_log(info, **kwargs):
    print(console.colorize("yellow", "corpus: ") + f"{info}", **kwargs)