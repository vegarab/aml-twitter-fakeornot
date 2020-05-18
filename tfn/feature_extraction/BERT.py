import pickle

def save(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    return 1

##-- Load obj from file    
def load(filename):
    with open(filename, 'rb') as input: 
        obj = pickle.load(input)
    return obj   


if __name__ == '__main__':
    from tfn.preprocess import Dataset
    self.sentences, self.labels = Dataset("glove")
    dataset = [sentences, labels]

    save(dataset, 'preprocess_data')
    