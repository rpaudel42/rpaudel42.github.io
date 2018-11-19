---
layout: post
title: Spam Tweet Detection in Trending Topic
tags: spam detection, anomaly detection, graph-based anomaly
categories: social network, twitter
description: Ramesh Paudel; Spam Tweet Detection in Trending Topic
---

<div class="topimage">
    <a href="../assets/pics/Architecture.pdf">
        <img src="../assets/pics/Architecture.pdf"
              title="Spam on trending tweet" alt="Spam on trending tweet"/></a>
    </div>

In recent years, social media has changed the way people communicate and share information. For example, when some importantand noteworthy event occurs, many people like to "tweet" (Twitter)or post information, resulting in the event trending and becomingmore popular. Unfortunately, spammers can exploit trending topics to spread spam more quickly and to a wider audience. Recently, researchers have applied various machine learning techniques on accounts and messages to detect spam in Twitter. However, the features of typical tweets can be easily fabricated by the spammers. In this work, we propose a novel graph based approach that leverages the relationship between the named entities present in the content of the tweet and the document referenced by the URL mentioned in the tweet for detecting possible spam. It is our hypothesis that by combining multiple, heterogeneous information together into asingle graph representation, we can discover unusual patterns inthe data that reveal spammer activities - structural features that are difficult for spammers to fabricate. We will demonstrate the usefulness of this approach by collecting tweets and news feeds related to trending topics, and running graph-based anomaly detection algorithms on a graph representation of the data, in order to effectively detect anomalies on trending tweets.
