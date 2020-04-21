---
layout: post
title: Graph Representation
tags: graph stream, graph represenatation, anomaly detection
categories: graph stream, graph represenatation, graph sketching
next: iot.html
description: Ramesh Paudel, Graph Represenatation
---

<div class="topimage">
    <img src="../assets/pics/sketching.pdf"
              title="Graph Representation" alt="Graph Representation"/>
</div>

In any application domain where the data are represents an entities and the relationship between
entities, e.g., web-page and hyperlinks; people and their friendship; papers and citations;
network devices and traffic flow; chemical compound and their bond, the graph is usually massive.
If a data have $$n$$ entities, it is essentially an $$O(n^2)$$ dimensional object.
Graph being ubiquitous standard for representing such relational data, impose a sheer challenge
to any learning algorithm because of it's size. In this work, we propose a novel unsupervised graph
representation approach in a graph stream called *SnapSketch* that can be used for anomaly detection.
It first performs a fixed-length random walk from each node in a network and constructs n-shingles
from a walk path. The top discriminative n-shingles identified using a frequency measure are projected
into a dimensional projection vector chosen uniformly at random. Finally, a graph is sketched into a
low-dimensional sketch vector using a simplified hashing of the projection vector and the cost of shingles.
Using the learned sketch vector, anomaly detection is done using the state-of-the-art anomaly detection
approach called Robust Random Cut Forest (RRCF). *SnapSketch* has several advantages, including fully unsupervised learning,
constant memory space usage, entire-graph embedding, and real-time anomaly detection.


