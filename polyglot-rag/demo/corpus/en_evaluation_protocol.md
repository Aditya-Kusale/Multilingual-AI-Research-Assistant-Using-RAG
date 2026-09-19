# Measuring Retrieval Quality in Cross-Lingual Search

## Abstract
We describe an evaluation protocol for retrieval systems that must serve
queries in one language against a corpus written in another. Aggregate scores
conceal per-language failure, so every metric in this protocol is reported per
query language and separately for the cross-lingual subset.

## Method
Each query is annotated with its language, the language of its gold documents,
and a list of gold document identifiers. We report recall at k, mean reciprocal
rank, and normalised discounted cumulative gain. A query is counted as
cross-lingual when the query language differs from the gold document language.

## Results
On our internal benchmark of 640 queries across nine languages, hybrid
retrieval reached recall at ten of 0.87 overall. The same-language subset
scored 0.91 while the cross-lingual subset scored 0.79. Adding a cross-encoder
reranker closed most of that gap, lifting the cross-lingual subset to 0.88.
Tamil and Telugu remained the weakest languages at 0.71 and 0.74 respectively,
which we attribute to sparse representation in the encoder's pretraining data
rather than to any property of the retrieval architecture.

## Discussion
The practical recommendation is to build an evaluation set of fifty to one
hundred annotated queries per target language before tuning anything else.
Teams that tune chunk size before measuring per-language recall routinely
optimise the wrong component.
