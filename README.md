# BDC_GCN_LinkPrediction
Datathon2022

### Result

Classification accuracy (both on train and val, True Positive and True negative rate trade-off shown in result.png.

* Data preparation: node_preprocessing.py

* Training process runable: vis.ipynb or dgl_test.py

* EDA: eda.ipynb

* trained weights: model_final.pth


### Data

- nodes (number id): webpage
    - (22470 linked, 1655 isolated)
- edge: exists if two pages link to each other (132039)
- Page’s text description
- Page type (label)

### Pre-processing

- Node features
    - labels: provided, 4 types
    - Embedding text one-hot vectors
    - Use **Doc2Vec, decide the output feature dimension based on the raw sentence length**
    
- Problem Abstraction: Link Prediction in Graph

- Future improvement: small model — room to increase complexity
    - Deeper GraphSAGE
    - higher number of channels
    - longer text embedding

### Graph

- Nodes: pages
- Edges: connectivity of pages
- Node feature: label + (embedded) text
