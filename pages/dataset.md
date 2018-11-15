---
layout: page
title: Datasets Repository
description: Dataset used in Research
---

### Twitter-Newsfeed Dataset
The dataset is collected from [News API](https://newsapi.org/) and [Twitter REST API](https://dev.twitter.com/rest/public).
The News API provides headlines from 70 worldwide sources including ABC News, BBC,
Bloomberg, Business Insider, Buzzfeed, Associated Press, CNN, CNBC, ESPN, Google News etc.
(A complete list of all the news sources we used to collect data from is shown in Appendix 1 of data documentation.) The
Twitter REST API provides tweet and publicly available twitter handler information for a specified
twitter handle.
The data collected in this set consists of news stories from 2/09/2017 to 6/23/2017, and associated
tweets that occurred 10 days before and after the corresponding news story, based upon the twitter
account (handle) mentioned in the body of the news.  


**How the Data was Collected?**

First, we collected news data from News API. The data from News API have author name, news
title, news headline, news url, published date, etc. Then, in order to get the body of the news story
(which is not returned from the News API), we crawled the URL for the associated news source
to get the body of the news.
Second, if the body of a news article references a twitter handle, the handle is sent to the Twitter
REST API where all tweets 10 days around the published news story are collected.
The result is two separate, comma-delimited (.csv) files, documents.csv and usertweet.csv,
corresponding to news stories and tweets respectively.

The full datasets of Twitter-Newsfeed dataset can be downloaded here (**[Twitter-Newsfeed.zip](/datasets/Twitter-Newsfeed.zip)**)
