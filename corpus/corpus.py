from torch.utils.data import Dataset
from abc import abstractmethod, ABC
from .utils import corpus_log

from pygments import console
import hashlib
import random
import json
import time
import os


class Flag:
    def __init__(self):
        self.quit = False


class BasicCorpus(Dataset, ABC):
    def __init__(
            self, 
            json_path, 
            processor, 
            max_instance=None,
            use_cache=True,
            cache_dir='data_cache'):

        self.max_instance = self.inf if max_instance is None else max_instance
        self.json_path = json_path
        self.processor = processor
        self.cache_dir = cache_dir
        self.use_cache = use_cache

        self.signature = hashlib.sha256(
            f"{self.__class__.__name__}/{self.json_path}/{self.max_instance}/{self.processor.signature}".encode()
        ).hexdigest()
        self.data = []

        if use_cache and not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)

        # load data
        if self.use_cache and self.is_checkpoint_exists():
            self.load()
        else:
            self.sample_data()
        self.print_final_info()

        # dump data
        if self.use_cache and not self.is_checkpoint_exists():
            self.dump()


    @property
    def inf(self):
        return 1e12
    

    @property
    def checkpoint_path(self):
        return os.path.join(self.cache_dir, f"{self.signature}.json")


    @abstractmethod
    def sample_data(self):
        ...


    def is_checkpoint_exists(self):
        return os.path.exists(self.checkpoint_path)


    def load(self):
        assert os.path.isdir(self.cache_dir), f"`{self.cache_dir}` is not existing."
        assert self.is_checkpoint_exists(), f"checkpoint not exists"
        with open(self.checkpoint_path, 'r') as f:
            for line in f:
                if line and line.strip() != '':
                    data = json.loads(line)
                    self.data.append(data)
                    self.print_process_info()

    
    def dump(self):
        assert os.path.isdir(self.cache_dir), f"`{self.cache_dir}` is not existing."
        corpus_log(f"Dumping data to `{self.cache_dir}` ... ")
        with open(self.checkpoint_path, 'w') as f:
            for data in self.data:
                json_code = json.dumps(data)
                f.write(json_code + '\n')


    def print_process_info(self):
        total = self.max_instance if self.max_instance != self.inf else '?'
        corpus_log(f"{self.json_path}:\t{len(self.data)}/{total}", end='\r', flush=True)


    def print_final_info(self):
        total = self.max_instance if self.max_instance != self.inf else '?'
        corpus_log(console.colorize("green" if len(self.data) == self.max_instance else "red",
                   f"\033[K{self.json_path}:\t{len(self.data)}/{total}"),
                   flush=True)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        return self.data[index]


class Corpus(BasicCorpus):
    def sample_data(self):
        with open(self.json_path, 'r') as f:
            line = f.readline()
        
            while line:
                result = None
                if line.strip():
                    line = json.loads(line)
                    result = self.processor.process(line)

                if result is not None:
                    self.data.append(result)
                    self.print_process_info()

                if len(self.data) >= self.max_instance:
                    break

                line = f.readline()


class RandomSampleCorpus(BasicCorpus):

    @property
    def total(self):
        if not hasattr(self, "_total"):
            count = 0
            with open(self.json_path, 'r') as f:
                for _ in f:
                    count += 1
            self._total = count
        return self._total


    def sample_data(self):
        with open(self.json_path, 'r') as f:
            line = f.readline()
            i = 0
            while line:
                result = None
                if line.strip():
                    line = json.loads(line)
                    result = self.processor.process(line)

                if result is not None:
                    if len(self.data) < self.max_instance:
                        self.data.append(result)
                    else:
                        j = random.randint(0, i)
                        if j < self.max_instance:
                            self.data[j] = result
                    i += 1
                    self.print_process_info()

                line = f.readline()


    def print_process_info(self):
        steps = ['-', '/', '|', '\\']
        if not hasattr(self, '_step'):
            self._step = 0
            self._time = time.time()
            self._ema_time = None
        if self._step % 10 == 0:
            delta = (time.time() - self._time) * (self.total - self._step)
            self._time = time.time()
            self._ema_time = (delta * 10
                if self._ema_time is None 
                else self._ema_time * 0.9 + delta * 0.1)
            corpus_log(f"{self.json_path}:\t{steps[(self._step // 10) % 4]}\tETA:\t{int(self._ema_time)} sec", end='\r', flush=True)
        self._step += 1
    

class LazyBasicCorpus(BasicCorpus):
    def is_checkpoint_exists(self):
        return False
    

    def load(self):
        ...

    
    def dump(self):
        ...


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_cache:
            corpus_log("lazy corpus dose not support `use_cache=True`, please disable it.")
            self.use_cache = False


class LazyCorpus(LazyBasicCorpus):
    def sample_data(self):
        with open(self.json_path, 'r') as f:
            for line in f:
                if line.strip() == '':
                    continue
                data = json.loads(line)
                self.data.append(data)
                self.print_process_info()
                if len(self.data) >= self.max_instance:
                    break


    def __getitem__(self, index):
        return self.processor.process(self.data[index])

        
class LazyRandomSampleCorpus(LazyBasicCorpus):
    @property
    def total(self):
        if not hasattr(self, "_total"):
            count = 0
            with open(self.json_path, 'r') as f:
                for _ in f:
                    count += 1
            self._total = count
        return self._total


    def sample_data(self):
        with open(self.json_path, 'r') as f:
            for line in f:
                if line.strip() == '':
                    continue
                data = json.loads(line)
                self.data.append(data)
        self.data = random.choices(self.data, k=self.max_instance)


    def print_process_info(self):
        corpus_log(f"{self.json_path}:\t{len(self.data)}/{self.total}", end='\r', flush=True)


    def __getitem__(self, index):
        return self.processor.process(self.data[index])