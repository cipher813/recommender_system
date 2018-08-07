# Building a Last.fm Recommendation System
Today we are going to build a basic recommender system based on a Last.fm dataset available at [GroupLens](https://grouplens.org/datasets/hetrec-2011/) on behalf of [Lab41](https://github.com/Lab41/hermes/wiki/Datasets).  The dataset, obtained from LastFM in 2011, contains the play counts of 17,632 artists by 1,892 users.

Our agenda is as follows:
- Examine the data we are working with by performing initial Exploratory Data Analysis (EDA)
- Build a couple versions of a basic collaborative recommender system:
 - K Nearest Neighbors in sci-kit learn
 - Item Similarity Recommender in TuriCreate
- Assess results
- Respond to questions on the project including areas for improvement

Please see [project notebook](https://github.com/cipher813/recommender_system/tree/master/notebooks) for full analysis.

*As a separate analysis, for preliminary EDA work to incorporate content-based filter components to our recommender system, see [Appendix notebooks](https://github.com/cipher813/recommender_system/tree/master/notebooks) for topic modeling of artist tags with K Means Clustering and tSNE, and separately with pyLDAvis. Adding content-based features from metadata, such as artist tags by users, is a key area of further work to build upon this analysis.*
