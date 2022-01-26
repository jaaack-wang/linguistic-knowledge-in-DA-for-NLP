import json
from tqdm import tqdm
from os.path import exists


def ngramGenerator(tokens, n):    
    start = 0
    end = len(tokens) - n + 1
    res = []
    for i in range(start, end):
        res.append(' '.join(tokens[i:i+n]))
    return res


def saveJson(dic, path):
    with open(path, 'w') as f:
        json.dump(dic, f)
    print(path + " has been saved!")
    

def readJson(path):
    return json.load(open(path))


def load_dataset(fpath, num_row_to_skip=0):
    def read(path):
        data = open(path)
        for _ in range(num_row_to_skip):
            next(data)
    
        for line in data:
            line = line.split('\t')
        
            yield line[0], line[1], int(line[2].rstrip())
    
    if isinstance(fpath, str):
        assert exists(fpath), f"{fpath} does not exist!"
        return list(read(fpath))
    
    elif isinstance(fpath, (list, tuple)):
        for fp in fpath:
            assert exists(fp), f"{fp} does not exist!"
        return [list(read(fp)) for fp in fpath]
    
    raise TypeError("Input fpath must be a (list) of valid filepath(es)")
