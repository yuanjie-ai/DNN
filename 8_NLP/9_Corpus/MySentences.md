- LineSentence
```python
from gensim.models.word2vec import LineSentence

LineSentence('/DATA/1_DataCache/NLP/new_zhwiki/wiki_test.txt')
```


---
- MySentences
```python
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MySentences(object):
    def __init__(self, path='/DATA/1_DataCache/NLP/new_zhwiki/wiki_test.txt'):
        self.path = Path(path)

    def __iter__(self):
        if self.path.is_dir():
            files = tqdm(filter(lambda x: x.is_file(), self.path.iterdir()))
            for file in files:
                files.set_description("Processing '%s' " % file)
                with open(file) as file:
                    lines = tqdm(file)
                    for line in lines:
                        lines.set_description("Processing ")
                        yield line.lower().split()
        else:
            with open(self.path) as file:
                lines = tqdm(file)
                for line in lines:
                    lines.set_description("Processing ")
                    # assume there's one document per line, tokens separated by whitespace
                    yield line.lower().split()
```
