# Comment Highlight Extractor

Here I propose an automated system for grouping similar comments and then identifying the best representative from each group. These selections can be used to construct a high-level summary of the discussion in the comments, which is useful to readers who may be interested in the general response towards a piece, but unwilling to wade through many comments. This summary can function as an alternate entry point to the comments if these highlights link back to their original comments.




## Existing Implementations

Yelp uses such a system to provide users with a high-level summary of the general opinion towards a service:

![Yelp Example](yelp.png)

Yelp extracts sentences which are representative of the aggregate reviews for particular "aspects" of the service, such as "craft beer" in the example above. They also present how many reviews are aligned with the representative to provide an impression of the support for that opinion.

The YUMM project provides an implementation for a similar functionality, also on Yelp data: [https://github.com/Fossj117/opinion-mining](https://github.com/Fossj117/opinion-mining).

To my knowledge, no one has yet tried a similar technique on comments.


## Hypothesis

We can cluster the comments for a given article using an incremental hierarchical approach, which allows us to construct a cluster tree which can be persisted to disk or memory. With an incremental approach, new comments can be incorporated on a regular basis without needing to reconstruct the entire tree.

Under the assumption that we are likely to be clustering opinions towards some subject or subjects, the comments might be clustered on the following features:

- sentiment
- subjectivity metrics
- text
- article relevance
- named entities or keywords

Once the clusters are formed, there are two options:

- From these clusters we can select a representative comment, which is the "best" of the group by some metric (e.g. most recommended or highest rated).
- From these clusters we can select sentences or phrases which best represent the group.


## Current Status

A very early prototype is available at [https://github.com/ftzeng/geiger](https://github.com/ftzeng/geiger).


## Challenges

### Performance

In the current prototype, when the number of comments exceeds ~500, the time for incorporating a new comment grows dramatically. This speed should be better for production usage.

### Evaluation

As far as I know, there is no ground-truth data to evaluate the cluster quality against. What's the best way to test if the results look good?



## Next Steps

- Expand the prototype to take in the URL of any NYT article and cluster the comments.
- Try building some other features - ideally to find some more informative features and reduce dimensionality (which would help with speed).