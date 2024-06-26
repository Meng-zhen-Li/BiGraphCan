# BiGraphCan
BiGraphCan is an algorithm that enhances the representation of input networks by integrating the (i) raw network, (ii) local node similarity, and (iii) global node similarity through a Graph Convolutional Network (GCN) to learn accurate and representative embeddings. BiGraphCan takes a bipartite graph (e.g., drug-target associations) and two networks representing side information for entities on each side (e.g., a drug-drug similarity and a protein-protein interaction network) to compute an embeddings for all nodes in the bipartite graph.
## Datasets
There are preprocessed data and the preprocessing code in the `data` folder. 
## Running BiGraphCan
Use the following command to run BiGraphCan:
```
python main.py --dataset DGI
```
Parameters:
- `--testing_nodes:` The indeces of the nodes for testing(in one side of the bipartite graph).
- `--epochs:` The number of epochs to train the model.
- `--learning_rate:` The learning rate of training.
- `--hidden1:` The number of dimensions of the first GCN hidden layer.
- `--hidden2:` The number of dimensions of the second GCN hidden layer, and also the final output dimension.
- `--noise:` The percentage of noise added to the input graph.
