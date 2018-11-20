---
layout: page
title: Datasets Repository
description: Dataset used in Research
---

### Graphs Dataset
Several graph datasets from various application domains used in my research are listed here. Each graph are attributed in nature with nodes and edges. Original datasets are converted into .g format that can be used as input for [GBAD](http://users.csc.tntech.edu/~weberle/gbad/). Also .dot file format are available. If .dot file are not available, use graph2dot command in GBAD to convert to .dot file for visualiazation using [Graphviz](http://www.graphviz.org)). This will help people understand the relationship between entities within each dataset and facilitate the development of graph.
The .g file format 

**Smart Homes Acitivity Graphs**
The graphs are constructed by using [Kyoto dataset with 400 participants](http://casas.wsu.edu/datasets/) provided by Washington State University’s [CASAS program](http://casas.wsu.edu). The CASAS website provides a raw sensor log dataset for each participant containing time (HH:MM:SS), sensor identification, sensor value, and an activity number to show the activity is being executed (we have constructed graphs for first 8 activities). The dataset consist of 8 graphs for each of the 8 activities for 239 healthy patient and 3 patient with cognitive impairment (can be thought as anomaly). For more detail please refer [Anomaly Detection of Elderly Patient Activities in Smart Homes using a Graph-Based Approach](https://csce.ucmss.com/cr/books/2018/LFS/CSREA2018/ICD8019.pdf)
<ul>
 <li>Download <a href ="/datasets/smart-home-graphs.zip">Activity Graph</a></li>
</ul>
If you use this dataset, please cite <br/>
<ul>
<li><em>Paudel, R., Eberle, W., & Holder, L. B. Anomaly Detection of Elderly Patient Activities in Smart Homes using a Graph-Based Approach. Proceedings of the International Conference on Data Science, 163-169 (2018)</em>
 </li>
 </ul>

**Medicare Claim Graphs for Diabetic Patients**
The graphs are constructed by using [CMS Linkable 2008–2010 Medicare Data Entrepreneurs’ Synthetic Public Use File (DE-SynPUF)](https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/DE_Syn_PUF.html) provided by [Centers for Medicare & Medicaid Services (CMS)](https://www.cms.gov). Out of the 20 random sample files made available by the CMS, sub sample 1 is used. We have choosen 2009 beneficiaries from Tennessee and their inpatient, out- patient, carrier and prescription drug claims, when they have an initial diagnosis of diabetes. The graph input file is built from the dataset to reflects the relationship between beneficiaries, their claims, physicians involved, service provider institute, procedure performed, etc. Each beneficiary might have multiple inpatient, outpatient, carrier or prescription drug claims. The edge between a patient and a claim indicates that the patient filed, or was related to, the corresponding claim. It should also be noted that if a beneficiary has more than one claim, prescription, physician, etc., then multiple claim, prescription, physician, etc., nodes are created for each unique value, resulting in potentially multiple edges between the patient and these entities.
For more detail please refer [Detection of Anomalous Activity in Diabetic Patients Using Graph-Based Approach](https://aaai.org/ocs/index.php/FLAIRS/FLAIRS17/paper/view/15455/14978)
<ul>
 <li>Download <a href ="/datasets/diabetic-claim.zip">Medicare Claim Graph</a></li>
</ul>
If you use this dataset, please cite one of the following paper: <br/>
<ul>
<li><em>Paudel, Ramesh, William Eberle, and Doug Talbert. "Detection of Anomalous Activity in Diabetic Patients Using Graph-Based Approach." Proceedings of the Thirtieth International Florida Artificial Intelligence Research Society Conference (2017).</em></li>
<li><em>Rajbhandari, Niraj. Graph Sampling to Detect Anomalies in Large Graphs and Dynamic Graph Streams. Diss. Tennessee Technological University, 2018.</em></li>
</ul>

---

### Other Dataset

**Twitter-Trending Topic**
The dataset consists of tweets and documents(primarily news stories) mentioned in the tweets.
The data was collected using [Twitter’s standard search API](https://dev.twitter.com/rest/public). We
collected tweets related to two trending topics, “FIFA World Cup” and “NATO Summit”, during the summer of 2018. The results from Twitter’s search API contains tweet text, Twitter handle name, any hashtags and URLs mentioned in the tweet, as well as all publicly available information about the user including their name. The data for tweets is a JSON dump of individual tweets. After data collection, we manually inspected the data for the number of spam present in the dataset. We used following criteria to label the spam tweet.
<ul>
<li>If the tweet have keywords related to the trending topic but the document referred by the URL does not have any.</li>
<li>If the tweet have multiple link and if any of the link refer the document not related to the trending topic.</li>
<li>If tweet have a URL that redirects to a unrelated website before redirecting to the related website. This usually occur when the tweet have a tiny URL.</li>
</ul>
This dataset can be used for anomaly/spam detection in tweets, text mining etc. This has been use in one of our research [Spam Tweet Detection in Trending Topic](/pages/spamtweet.html). The dataset has the following types of spam/anomalies in the trending tweets that are consistent with the [spam scenarios listed by Twitter](https://help.twitter.com/en/safety-and-security/report-spam).
<ol>
 <li>Keyword/Hashtag Hijacking</li>
 <li>Bogus link</li>
 <li>Link piggybacking</li>
</ol>
The datasets of Twitter trending topic can be downloaded here (**[Twitter-Trending-Topic.zip](/datasets/trending-topic.zip)**)


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
<b>How the Data was Collected?</b>
First, we collected news data from News API. The data from News API have author name, news
title, news headline, news url, published date, etc. Then, in order to get the body of the news story
(which is not returned from the News API), we crawled the URL for the associated news source
to get the body of the news. Second, if the body of a news article references a twitter handle, the handle is sent to the Twitter REST API where all tweets 10 days around the published news story are collected.
The result is two separate, comma-delimited (.csv) files, documents.csv and usertweet.csv,
corresponding to news stories and tweets respectively.

This data can be useful for text/topic mining and is used for [Mining Heterogeneous Graph for Patterns and Anomalies](https://publish.tntech.edu/index.php/PSRCI/article/view/365)
The full datasets of Twitter-Newsfeed dataset can be downloaded here (**[Twitter-Newsfeed.zip](/datasets/Twitter-Newsfeed.zip)**)

---


