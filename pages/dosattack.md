---
layout: post
title: Detecting the Onset of a Network Layer DoS Attack
tags: DoS Attack, anomaly detection, graph-based anomaly
categories: Network, DoS Attack
description: Ramesh Paudel; Detecting the Onset of a Network Layer DoS Attack
---

<div class="topimage">
    <a href="../assets/pics/DoS.png">
        <img src="../assets/pics/DoS.png"
              title="DoS attack graph" alt="DoS attack graph structure"/></a>
    </div>

A denial-of-service (DoS) attack is a malicious act
with the goal of interrupting the access to a computer network.
The result of this type of attack can cause the computers on the
network to squander their resources to serve illegitimate requests
that result in a disruption of the network’s services to legitimate
users. With a sophisticated DoS attack, it becomes difficult to
distinguish malicious requests from legitimate requests. Since a
network layer DoS attack can cause interruptions to a network
while causing collateral damage, it is vital to understand the
measures to mitigate against such attacks. Generally, approaches
that implement distribution charts based on statistical analysis or
honeypots have been applied to detect a DoS attack. However, this
is usually too late, as the damage is already done. We hypothesize
in this work that a graph-based approach can provide the
capability to identify a DoS attack at its inception. A graph-based
approach will also allow us to not only focus on anomalies within
an entity (like a computer), but also allow us to analyze the
anomalies that exist in an entity’s relationship with other entities,
thus providing a rich source of contextual analysis. We
demonstrate our proposed approach using a publicly-available
data set. 

In this research, the early detection of the anomaly
was possible because the graph-based approach was able to
represent the direct repercussions of the attack (e.g., calls to
the DNS servers by the external web server). As we know,
the goal of a network DoS attack is to create bogus return addresses,
causing the network to squander its resources, thus
preventing access to legitimate users. Since the local web
server does not know the bogus return address associated
with the packets sent by the DoS attack, the web servers
must perform a DNS query. This resulted in the change in
the graph structure between the entities in the network. In
this particular scenario, the new relationship between the external
web and the DNS was created which was represented
as a new node “DNS” hanging off “external web”.
Therefore, a graph based approach represented the direct repercussions
of the DoS attack and discovered a potential DoS attack
in its early stages. The first known anomaly was reported
within 5 seconds of the DoS attack inception. Also, we were
able to identify all five IPs from which the DoS attack was
instigated to the external web server.

The unusual behavior of the web server was marked as an anomaly by GBAD
which helped to flag the DoS attack at its inception. Also,
it should be noted that every anomaly reported by GBAD is
related to the DoS attack. Thus, there are not any false positives
(see confusion matrix in Table below). While a nice feature
of what was performed here, we know that it should not be
taken as a standard for applying a graph-based approach, and
potentially another dataset might have produced false positives
(something we plan to investigate in the future).

For further detail, please refer the paper published in [FLAIRS-2019 Conference](https://rpaudel42.github.io/assets/MAIN-F-PaudelR.77.pdf).

<div class="topimage">
        <img src="../assets/pic/dos_result.png"
              title="dos attack result" alt="dos attack result"/>
</div>
