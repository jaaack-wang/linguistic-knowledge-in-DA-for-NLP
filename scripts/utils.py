import json


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


def dataLoader(path):
    file = open(path)
    next(file)
    for line in file:
        line = line.split('\t')
        yield line[0], line[1], int(line[2].strip())
        

def lcqmcLoader(dataset, path_tmp='../data/{}.txt'):
    path = path_tmp.format(dataset)
    return list(dataLoader(path))