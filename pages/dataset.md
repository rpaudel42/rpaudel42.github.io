## A Repository of Benchmark Graph Datasets for Graph Classification

### Introduction to Graph Classification
Recent years have witnessed an increasing number of applications involving objects with structural relationships, including chemical compounds in Bioinformatics, brain networks, image structures, and academic citation networks. For these applications, graph is a natural and powerful tool for modeling and capturing dependency relationships between objects.

Unlike conventional data, where each instance is represented in a feature-value vector format, graphs exhibit node–edge structural relationships and have no natural vector representation. This challenge has motivated many graph classification algorithms in recent years. Given a set of training graphs, each associated with a class label, graph classification aims to learn a model from the training graphs to predict the unseen graphs in future.  The following picture shows the difference betweeb classification on **vector data** and **graph data**.

![(Graph Classification)](https://github.com/shiruipan/graph_datasets/blob/gh-pages/VectorVsGraph.png)


### Dataset Summaization

This repository maintains 31 benchmark graph datasets, which are widely used for graph classification. The graph datasets consist of:

- **chemical compounds**
- **citation networks**  
- **social networks** 
- **brain networks**


The chemical compound graph datasets are in “.sdf” or “.smi” format, and other graph dataset are represented as “.nel” format. All these graph datasets can be handle by frequent subgraph miner packages such as Moss [1] or other softwares. These graphs can be easily converted to other formats handled by Matlab or other softwares. 
A summarization of our graph datasets is given in [Table 1](https://github.com/shiruipan/graph_datasets/blob/gh-pages/Picture1.png).

![Fig 1 (Graph Datasets)](https://github.com/shiruipan/graph_datasets/blob/gh-pages/Picture1.png)


If you used the dataset, please cite the related papers properly.

### 1.	NCI Anti-cancer activity prediction data (NCI)
**Description:** 

The NCI graph datasets are commonly used as the benchmark for graph classification. Each NCI dataset belongs to a bioassay task for anticancer activity prediction, where each chemical compound is represented as a graph, with atoms representing nodes and bonds as edges. A chemical compound is positive if it is active against the corresponding cancer, or negative otherwise.  Table 1 summarizes the NCI graph data we download from PubChem. We have removed disconnected graphs and graphs with unexpected atoms (some graphs have atoms represented as `*`) in the original graphs. Columns 2-3 show the number of positive and total number of graphs in each dataset, and Columns 4-5 indicate the average number of nodes and edges in each dataset, respectively. 

Number of Datasets: **18 (9 imbalanced + 9 balanced data)**
 
**Full Dataset:**

The full datasets of NCI graphs can be downloaded here (**[NCI_full.zip](https://github.com/shiruipan/graph_datasets/blob/master/Graph_Repository/NCI_full.zip?raw=true)**), which are naturally imbalanced and ideal benchmark for imbalanced or cost-sensitive graph classification. We have considered cost-sensitive graph classification in [2], and graph stream classification in [3][4][5].

**Partial Dataset:**

We randomly select #Pos number of negative graphs from each original graph set to create balanced graph datasets, which are available here (**[NCI_balanced.zip](https://github.com/shiruipan/graph_datasets/blob/master/Graph_Repository/NCI_balanced.zip?raw=true)**). This dataset was used in [7] for genral graph classification and [5] for multi-task graph classification

**Citations:**

If you used this dataset, please cite 2-3 of following papers:

- _Shirui Pan, Jia Wu, and Xingquan Zhu “CogBoost: Boosting for Fast Cost-sensitive Graph Classification",  IEEE Transactions on Knowledge and Data Engineering (TKDE),  27(11): 2933-2946 (2015)_
- _Shirui Pan, Jia Wu, Xingquan Zhu, Chengqi Zhang, Philip S. Yu. "Joint Structure Feature Exploration and Regularization for Multi-Task Graph Classification." IEEE Trans. Knowl. Data Eng. 28(3): 715-728 (2016)_
- _Shirui Pan, Jia Wu, Xingquan Zhu, and Chengqi Zhang, “Graph Ensemble Boosting for Imbalanced Noisy Graph Stream Classification",  IEEE Transactions on Cybernetics (TCYB), 45(5): 940-954 (2015)._
- _Shirui Pan, Xingquan Zhu, Chengqi Zhang, and Philip S. Yu. "Graph Stream Classification using Labeled and Unlabeled Graphs", International Conference on Data Engineering (ICDE), pages 398-409, 2013_
- _Shirui Pan, Jia Wu, Xingquan Zhu, Guodong Long, and Chengqi Zhang. " Task Sensitive Feature Exploration and Learning for Multi-Task Graph Classification."  IEEE Trans. Cybernetics (TCYB) 47(3): 744-758 (2017)._
- _Shirui Pan, Jia Wu, Xingquan Zhu, Guodong Long, Chengqi Zhang. "Finding the best not the most: regularized loss minimization subgraph selection for graph classification." Pattern Recognition (PR) 48(11): 3783-3796 (2015)_

