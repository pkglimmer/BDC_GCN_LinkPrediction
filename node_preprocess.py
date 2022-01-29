import pandas as pd
import numpy as np
import json, time
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
"""
1. Generate doc
"""

def unique(list1):
    x = np.array(list1)
    return np.unique(x)

data_dir  = './bill_challenge_datasets'
train_graph = pd.read_csv(f'{data_dir}/Training/training_graph.csv')
page_label = pd.read_csv(f'{data_dir}/Training/node_classification.csv')
iso_nodes = pd.read_csv(f'{data_dir}/Training/isolated_nodes.csv')
iso_nodes = list(iso_nodes.nodes)

f = open(f'{data_dir}/Training/node_features_text.json')
corpus = json.load(f)
f.close()
final_corpus = []
for k in corpus.keys():
    final_corpus.append(corpus[k])
tagged_data = [TaggedDocument([str(x) for x in d], [i]) for i, d in enumerate(final_corpus)]

t0 = time.time()
d2v_model = Doc2Vec(tagged_data, vector_size = 32, window = 2, min_count = 1, epochs = 200)

node_list= list(train_graph.node1)+ list(train_graph.node2)
node_list = unique(node_list)
df = pd.DataFrame()
df = pd.DataFrame(columns = ['id', 'label', 'd2v'])

for node in node_list:
    df = df.append({
        'id': node,
        'label': page_label.loc[node, 'page_type'],
        'd2v': d2v_model.infer_vector([str(x) for x in corpus[str(node)]])
    }, ignore_index = True)


print(df.head())
df.to_csv('node_feature_32.csv')
t1 = time.time()
print(f'Time elapsed: {t1-t0}s')