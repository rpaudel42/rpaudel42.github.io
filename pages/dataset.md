---
layout: page
title: Datasets Repository
description: Dataset used in Research
---

### Graphs Dataset
Several graph datasets from various application domains used in my research are listed here. Each graph are attributed in nature with nodes and edges. Original datasets are converted into .g format that can be used as input for [GBAD](http://users.csc.tntech.edu/~weberle/gbad/). Also .dot file format are available. If .dot file are not available, use graph2dot command in GBAD to convert to .dot file for visualiazation using [Graphviz](http://www.graphviz.org)). This will help people understand the relationship between entities within each dataset and facilitate the development of graph.
The .g file format 

<h4>Smart Homes Acitivity Graphs</h4>
The graphs are constructed by using [Kyoto dataset with 400 participants](http://casas.wsu.edu/datasets/) provided by Washington State University’s [CASAS program](http://casas.wsu.edu). The CASAS website provides a raw sensor log dataset for each participant containing time (HH:MM:SS), sensor identification, sensor value, and an activity number to show the activity is being executed (we have constructed graphs for first 8 activities). The dataset consist of 8 graphs for each of the 8 activities for 239 healthy patient and 3 patient with cognitive impairment (can be thought as anomaly). 
<ul>
 <li>Downloaded here (**[Activity-Graphs.zip](/datasets/smart-home-graphs.zip)**)</li>
 <li>[Read Me](/datasets/smart-home-graphs.zip)</li>
 </ul>
---

### Other Dataset

**Twitter-Newsfeed Dataset**
The dataset is collected from [News API](https://newsapi.org/) and [Twitter REST API](https://dev.twitter.com/rest/public).
The News API provides headlines from 70 worldwide sources including ABC News, BBC,
Bloomberg, Business Insider, Buzzfeed, Associated Press, CNN, CNBC, ESPN, Google News etc.
(A complete list of all the news sources we used to collect data from is shown in Appendix 1 of data documentation.) The
Twitter REST API provides tweet and publicly available twitter handler information for a specified
twitter handle.
The data collected in this set consists of news stories from 2/09/2017 to 6/23/2017, and associated
tweets that occurred 10 days before and after the corresponding news story, based upon the twitter
account (handle) mentioned in the body of the news.  

<h5>How the Data was Collected?</h5>

First, we collected news data from News API. The data from News API have author name, news
title, news headline, news url, published date, etc. Then, in order to get the body of the news story
(which is not returned from the News API), we crawled the URL for the associated news source
to get the body of the news.
Second, if the body of a news article references a twitter handle, the handle is sent to the Twitter
REST API where all tweets 10 days around the published news story are collected.
The result is two separate, comma-delimited (.csv) files, documents.csv and usertweet.csv,
corresponding to news stories and tweets respectively.

This data can be useful for text/topic mining.
The full datasets of Twitter-Newsfeed dataset can be downloaded here (**[Twitter-Newsfeed.zip](/datasets/Twitter-Newsfeed.zip)**)

---


