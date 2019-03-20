# Dependencies
* Tensorflow
* Numpy

#Usage
The main file to run is `master.py`. This file orchestrates the entire data generation/analysis pipeline. The important parameter for this file is the `config` dictionary, which contains the following keys:

* root : The root path at which the master file is located.
* symname : The symbolic name of the NN. In other words, the name of the folder where the data should be saved.
* epochs : A list of epochs for which the NN should be trained.
* layerWidths : The width of each hidden layer.
* layerNames : Names of each layer. Should only worry about this if using .hdf5 format (disabled by default)
* numProcesses : Number of processes to use when running bulkProcess in parallel. Currently disabled by default.
* nnSaveFn : The function to use to generate save location for NN weights.
* accSaveFn : The function to use to generate save location for NN accuracy data.

These parameters should be tuned according to usage requirements. For example, if I wanted to train a network with two hidden layers of width 16 and 16, I would just set `layerWidths = [16,16]`, and master would take care of the rest.

To actually use the file to do things, there are the following three functions:

* `trainNetwork()`: This method trains a NN and stores data using the configuration provided in the `config` dictionary. 
* `bulkProcess()`: This method calls the method `runPipeline()` once on each epoch to calculate betti numbers.
* `analysis()` : This method runs any necessary analysis on the betti numbers. For example, to generate average lifetime betti graphs and save them.

There are four supporting files for master.py:

* `weightloader.py` : This file knows how to load NN weights from storage.
* `generateAdjacencyMatrix.py` : This file knows how to convert NN weights into an adjacency matrix.
* `preprocessing.py` : This file preprocesses an adjacency matrix to get it ready for a filtration. For example, this is where floyd-warshall is run.
* `filtration.py` : This file takes a matrix and computes homology. This is where ripser is run.

By adding functions to these files, we can easily change our pipeline. We can then call these new functions from `runPipeline()`, and we are assured that data is stored safely and that we can easily run things without having to mess around with paths.

Example usage to train a network and then compute Homology, take the betti numbers and make graphs.

```
python master.py
```

And in master.py:

```
if __name__ == "__main__":
  trainNetwork()
  bulkProcess()
  analysis()
```
