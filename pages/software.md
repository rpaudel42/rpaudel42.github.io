---
layout: page
title: Software
description: Ramesh Paudel's software
---


#### <a name="GBAD"></a>![pdf]({{ BASE_PATH }}/assets/pics/gbad.ico)&nbsp;&nbsp;[Graph-Based Anomaly Detection (GBAD)](http://users.csc.tntech.edu/~weberle/gbad/)

GBAD discovers anomalous instances of structural patterns in data, where the data represents entities, relationships and actions in graph form. Input to GBAD is a labeled graph in which entities are represented by labeled vertices and relationships or actions are represented by labeled edges between entities.  Using the minimum description length (MDL) principle to identify the normative pattern that minimizes the number of bits needed to describe the input graph after being compressed by the pattern, GBAD embodies novel algorithms for identifying the three possible changes to a graph:  modifications, insertions and deletions.  Each algorithm discovers those substructures that match the closest to the normative pattern without matching exactly.  As a result, GBAD is looking for those activities that appear to match normal (or legitimate) transactions, but in fact are structurally different. <br/>
[find more...](http://users.csc.tntech.edu/~weberle/gbad/)

---

#### <a name="relcheck"></a>RelCheck
Version 0.67 (24 August 2000)
Software for verifying relationships between all pairs of
individuals in a linkage study, using the approach of Boehnke and Cox
([Am
J Hum Genet, 61:423-429, 1997](https://www.ncbi.nlm.nih.gov/pubmed/9311748)), with the modification described by
Broman and Weber ([Am
J Hum Genet 63:1563-1564, 1998](https://www.ncbi.nlm.nih.gov/pubmed/9792888)), to allow for the presence of
genotyping errors.  We look only at the relationships MZ twins,
parent/offspring, fullsibs, halfsibs and unrelated.

**Note**: I am no longer actively developing this software. You may wish to check out Mary Sara McPeek and Lei
Sun's program [PREST](http://galton.uchicago.edu/~mcpeek/software/prest/).  It has a similar aim, and calculates a
more extensive set of statistics, includes measures of statistical
significance, and also looks at avuncular and first cousin
relationships.  Other alternatives include the programs Borel,
Relpair, Relative, Reltype and Siberror.  Go to the
[Rockefeller software list](http://www.jurgott.org/linkage/ListSoftware.html)
to find these programs.

The input/output for my program is rather crude.  If you use the
software, please reference the above papers in any publications.

A perl script for converting data from linkage format to that used by
RelCheck is included with the software.

Download:
[source](https://www.biostat.wisc.edu/software/relcheck/relcheck_0.67.tar.gz) | [windows](https://www.biostat.wisc.edu/software/relcheck/relcheck_0.67.zip)

Sample data \[[tar.gz](https://www.biostat.wisc.edu/software/relcheck/sampledata.tar.gz) |
[zip](https://www.biostat.wisc.edu/software/relcheck/sampledata.zip)\]
README file: \[[README.txt](https://www.biostat.wisc.edu/software/relcheck/README.txt)\]
List of updates to the software: \[[CHANGES.txt](https://www.biostat.wisc.edu/software/relcheck/CHANGES.txt)\]

---

#### <a name="f2"></a>f2

Version 0.50 (7 Feb 2000)

Software for QTL analysis of an F2 intercross experiment,
including forward selection for multiple QTLs, all pairs of loci, and
pairwise interactions.

**Note**: This is very preliminary, the input and output
are not well documented, and I'm no longer actively developing this software.  Look at [R/qtl](http://rqtl.org), instead.

Download: [source](https://www.biostat.wisc.edu/software/f2/f2_0.50.tar.gz) | [windows](https://www.biostat.wisc.edu/software/f2/f2_0.50.zip)

Sample data \[[tar.gz](https://www.biostat.wisc.edu/software/f2/example.tar.gz) | [zip](https://www.biostat.wisc.edu/software/f2/example.zip)\]
