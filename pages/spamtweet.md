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

Twitter, a popular social media, or micro-blogging, site, allows users to post
information, updates, opinions, etc., using tweets. Given its wide-spread popularity
for immediately sharing thoughts and ideas, adversaries try to manipulate
the micro-blogging platform and propagate off topic content for their selfish motives.
Compounding the issue, as the popularity increases around a certain
event, more people tweet about the topic, thereby increasing its "trending" rate.
Spammers then exploit these popular, trending Twitter topics to spread their
own agendas by posting tweets containing keywords and hash-tags of the trending
topic along with their misleading content. Ideally, one would like to be able to
identify anomalous tweets on a trending topic that have the potential to mislead
the population, or even possibly cause further harm. Currently, Twitter allows
users to report spam, and after an investigation, an account can be suspended.
However, suspending a spam account is not an efficient technique to deal with
spam related to trending topics because the suspension process is slow, and the
trending topics usually last for only a few hours or a day at most. Therefore,
the focus of the anomaly detection on trending topics in this work is on the
detection of tweets containing spam, instead of detecting spam accounts.

One of the more malicious activities involves a spammer who includes a URL
in the tweet, leading the reader to a completely unrelated website. It is reported
that 90% of anomalous tweets contain unrelated or misleading URLs.
People use shortened URLs or links in their tweet because of the limited number
of characters (280) available in the tweet. Since it is common for tweets to include
shortened text, so as to fit within the character limits, spammers can conceal
their unrelated/malicious links with shortened URLs. Hence, the problem with
shortened URL is that users do not know what is the actual domain until the link
is clicked. The existing approaches for spam detection on Twitter use various
machine learning tools on user-based features (e.g., number of followers, number
of tweets, age of the user account, number of tweets posted per day or per week,
etc.) and content-based features (e.g., number of hashtags, mentions, URL, likes,
etc.). Though user and content based features can be
extracted efficiently, an issue is that these features can also be fabricated easily
by the spammer. However, being able to hide an inconsistency between
the topic of a tweet and the topic of the document referred by URLs in the tweet
is much harder.

In this research, we propose an unsupervised, two-step, graph-based approach
to detect anomalous tweets on trending topics. First, we extract named entities
(like place, person, organization, product, event, or activity) present in the tweet
and add them as key elements in the graph. As tweets on a certain topic share
the contextual similarity, we believe they also share same/similar named entities.
These named entities representing relevant/similar topics can have a relationship
(e.g., shared ontology) amongst themselves, which we believe if represented
properly, will provide broader insight on the overall context of the topic. As such,
graphs can be a logical choice for representing these kinds of information where
a node can represent a named entity and an edge can represent the relationships
between them. Using a well-known graph-based tool like GBAD, we
then discover the normal and anomalous behavior of a trending topic. Second,
we propose adding hyperlinked document information because anomalies that
could not be detected from tweets alone could be detected using both the document
and tweets. It is our assumption that a better understanding of patterns
and anomalies associated with entities like person, place, or activity, cannot be
realized through a single information source, but better insight can be realized
using multiple information sources simultaneously. For instance, one can discover
interesting patterns of behavior about an individual through a single social media
account, but better insight into their overall behavior can be realized by
examining all of their social media actions simultaneously. Analyzing multiple
information sources for anomaly detection on Twitter has been explored in the
past. For example, the inconsistencies between the tweet and the document referred
to by a URL in the tweet using cosine similarity and a language model
were studied for potential anomaly detection. But, the cost for is high
as each tweet with a link is treated as a suspect and need a predefined source
of reliable information for each topic which makes these approaches less 
exible in real-time trending topics.

Using the above mentioned 2-step approach, we aim to detect the following
types of spam/anomalies in trending tweets that are consistent with the spam
scenarios listed by Twitter.
1. Keyword/Hashtag Hijacking: Using popular keywords or hashtags to promote
the tweet that are not related to the topic. This is done to promote
anomalous tweets to a wider audience by hijacking popular hashtags and
keywords.
2. Bogus link : Posting a URL that has nothing to do with the content of the
tweet. This is done to generate more trac to the website. Another scenario
of bogus link is link piggybacking. For example, posting an auto redirecting
URL that goes to legitimate website but only after visiting an illegitimate
website. Another way is to post multiple links where one link can be a legitimate
link while another can be a malicious or unrelated link. The motivation
behind link piggybacking is to generate trac to the illegitimate website by
concealing the link inside a legitimate website. This can also be accomplished
by using a tiny URL.

To verify our approach, we collect tweets (containing URLs) related to two
separate (and very different) trending topics during the summer of 2018: FIFA
World Cup and NATO Summit. We then construct graphs using information
from the tweet text and the document referred inside the tweet, followed by using
a graph-based anomaly detection tool. We then compare the performance of our
proposed approach with several existing approaches to show the effectiveness of
a graph-based approach.

For further detail, please refer the paper published in [FTC-2019 Conference](https://aaai.org/ocs/index.php/FLAIRS/FLAIRS18/paper/view/17622/16833).



<div class="topimage">
    <a href="../assets/pics/Graph-Layout-Draw-IO.pdf">
        <img src="../assets/pics/Graph-Layout-Draw-IO.pdf"
              title="Result Spam tweet" alt="Result Spam tweet"/></a>
</div>
