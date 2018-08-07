# Building a Last.fm Recommendation System
Today we are going to build a basic recommender system based on a Last.fm dataset available at [GroupLens](https://grouplens.org/datasets/hetrec-2011/) on behalf of [Lab41](https://github.com/Lab41/hermes/wiki/Datasets).  The dataset, obtained from LastFM in 2011, contains the play counts of 17,632 artists by 1,892 users.

Our agenda is as follows:
- Examine the data we are working with by performing initial Exploratory Data Analysis (EDA)
- Build a couple versions of a basic collaborative recommender system:
 - K Nearest Neighbors in sci-kit learn
 - Item Similarity Recommender in TuriCreate
- Assess results
- Respond to questions on the project including areas for improvement

In addition to this README containing general information on the project, our project consists of the following [notebooks](https://github.com/cipher813/recommender_system/tree/master/notebooks):
* (01) Collaborative Filter using scikit-learn and TuriCreate (main project submission)
* Appendices of preliminary EDA/topic modeling of user/artist tags for content-based filter:
 - (A01) Using scikit-learn's K Means Clustering and t-SNE
 - (A02) LDA using pyLDAvis  

*Adding content-based features from metadata, such as artist tags by users, is a key area of further work to build upon this analysis.*

### What is a recommender system?

A _recommender system_, also known as a _recommendation engine_ or simply _recommender_, exists in several forms:  

- A _content-based recommender_ recommends based on history of similar items purchased, viewed or interacted with in the past.  

- A _collaborative recommender_ recommends based on usage trends of similar users.  For instance, if Alice and Bob like movies X, Y and Z, and you like movies X and Y, you may be recommended movie Z.  Or, if people who listen to the Beatles and the Rolling Stones also typically play Bob Dylan, then when you input the Beatles into a recommender system, Bob Dylan may be a valid recommendation.

- A _hybrid recommender_ utilizes features of both content-based and collaborative recommenders in an aim to improve quality of results.  Amazon may recommend a stereo speaker to you based on your viewing history of sound systems (content-based filtering) as well as on your website usage behavior similar to other users of the Amazon platform (collaborative filtering).  

- A _popularity-based recommender_ does not take usage history into consideration other than sheer popularity of the content.  If Britney Spears and Justin Bieber are on top of the Billboard charts (due to sheer popularity), either (or both) artist could be recommended to each new user of Spotify in lieu of basing the recommendation on their usage history.  

A recommender system works by mapping the distance (ie the "similarity") between points (ie artists) in our dataset.  Three popular metrics include:
- The [Jaccard similarity](https://en.wikipedia.org/wiki/Jaccard_index) is a good choice for implicit item feedback (ie binary feedback such as like/dislike or played/not played).  
- The [Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) is good for comparing the ratings of items, but does not consider the differences in mean and variance of the items.
- The [Pearson Correlation similarity](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) also comparies the ratings of items and effects of mean and variance have been removed.  

### The Dataset

Six .csv files are provided in this dataset (download from GroupLens [here](http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip)), consisting of the following:
- **user_artists.dat** (userID, artistID, weight). Plays of artist by user.
- **artists.dat** (id, name, url, pictureURL). ID and name of each artist.  
- **tags.dat** (tagID, tagValue). ID number and content for each tag.
- **user_taggedartists.dat** (userID, artistID, tagID, day, month, year). Tag of artist by user with date.   
- **user_taggedartists-timestamps.dat** (userID, artistID, tagID, timestamp). Tag of artist by user with timestamp.
- **user_friends.dat** (userID, friendID). User/friend relationships.  

In total, we have:
- 1,892 users
- 17,632 artists
- 92,834 artist/user plays
- 11,946 tags
- 186,479 tag assignments

To build a **collaborative recommender**, we will primarily use the user_artists.dat file, which contains artist plays by user. A sparse matrix will be built with userID in the columns and artists in the rows, mapping the listening patterns of each user to each artist. This will provide artist similarity based on usage patterns ("plays") of similar users.  If users who play the Beatles and Bob Dylan also often play the Rolling Stones, then these artists would be considered more similar and as such a user who enjoys the Beatles may be more reasonably recommended the Rolling Stones than a dissimilar artist (based one one of our distance metrics), such as Snoop Dogg.     

To build a **content-based recommender**, we would primarily use tags.dat, user_taggedartists.dat and/or user_taggedartists-timestamps.dat to determine artist similarity based on tags.  To address differing spellings of tags (such as "rock", "rocking" or "rocks"), we may consider word embeddings after certain processing, such as stemming/lemmatizing the words, which would reduce "rock", "rocks" or "rocking" to the stem rock.  

We could also incorporate both sets of information into a hybrid recommender in an effort to achieve even greater recommendation quality.  

The user_friends.dat file appears less useful for our preliminary analysis.  Denoting someone as a friend does not translate to similar listening patterns.  Friend playlists could be an relevant addition to our future recommendation system platform (Spotify has an interesting "Fried Activity" feature), but this data is unlikely to add any accuracy to the models we can construct from the data noted above.

With this data we have our choice of which recommender system to build.  In this analysis, we will build a **collaborative recommender system** and leave the content-based or hybrid recommender options as key areas for development of a future version *(although you can find preliminary topic modeling work on the artist tags by user in the [Appendix notebooks](https://github.com/cipher813/recommender_system/tree/master/notebooks) of this project)*.

### Questions to ponder:
**1. Ask yourself why would they have selected this problem for the challenge?**

Recommender systems are a key application of machine learning which have significantly permeated our everyday lives.  Notable recommendation systems include those on the platforms of Amazon and Netflix, as well as LastFM (the source of this dataset).  

Recommender systems assist the user in finding content that they may like that they would not have found otherwise.  They are also often used as search algorithms for non-traditional data.  

**2. What are some gotchas in this domain I should know about?**

Recommender systems exist in various forms, but the most notable include (1) collaborative, (2) content-based, and (3) hybrid of collaborative and content-based.  Collaborative filtering functions by comparing the behaviour of similar users.  If A and B liked movies X,Y and Z, and you like movies X and Y, then the system may recommend movie Z for you.  Content-based filter utilizes the features of the item to recommend similar items.  In music, if you like guitar acoustic music the system will recommend similarly structured music, such as same or near category (perhaps a ballad?).  Other forms of recommenders also exist, such as by popularity but in this form individual usage data is not taken into consideration.  

Recommender systems can be evaluated via a variety of metrics, discussed further in #3 below, as we would typically evaluate a machine learning model.  Possible evaluation metrics include Root Mean Squared Error (RMSE), precision/recall and Mean Average Precision (MAP).  However, recommenders are most valuable when they provide users with insight on things they may like without them previously knowing about it.  A recommender would be most accurate if it only recommended items that a user is known to like, but that would not be of any value to a user.  As such, these evaluation metrics alone are not all that should be considered when determining the relevance and usefulness of model results.  


**3. What is the highest level of accuracy that others have achieved with this dataset or similar problems / datasets ?**

Recommender systems that score high in accuracy may not necessarily be useful systems, as the value of a recommender system is to propose items that you have not interacted with previously.  For instance, you may like Top 40 music.  If the recommender system only recommends top 40 music, it will score high in accuracy but will never introduce you to music you have never heard before (assuming that you have already heard all the top 40 songs).  

In this analysis, we evaluate our recommender based on:
- _Precision/Recall_ based on whether the model is able to correctly predict the users ratings or likes. The relevant classifications in a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) would be:
 - True Positive. User liked, model predicted a like.
 - False Positive. User didn't like, model predicted a like (Type I Error).
 - True Negative. User didn't like, model predicted a didn't like.
 - False Negative.  User liked, model predicted a didn't like (Type II Error).  

 Precision and Recall are calculated as follows:

<img align="middle" src="https://wikimedia.org/api/rest_v1/media/math/render/svg/26106935459abe7c266f7b1ebfa2a824b334c807" height=150 width=150><br>

<img align="middle" src="https://wikimedia.org/api/rest_v1/media/math/render/svg/4c233366865312bc99c832d1475e152c5074891b" height=150 width=150><br>

- _Root Mean Squared Deviation or Error (RMSD/RMSE)_ where we have a target, such as a rating.  The RMSE between two vectors, x<sub>1</sub> and x<sub>2</sub>, for T different predictions, is defined as:

<img align="middle" src="https://wikimedia.org/api/rest_v1/media/math/render/svg/b2fd7c6f9e8bb14ef297e659c721e8e5922310ca" height=200 width=200><br>

Recommenders are also often evaluated using _Mean Average Precision (MAP)_, which is, as it reads, the mean of the average precision scores for each query.  With Q as the number of queries, we calculate as:

<img align="middle" src="https://wikimedia.org/api/rest_v1/media/math/render/svg/decf93706ec7c8632fdfabe41470962101f9bcd8" height=150 width=150><br>


Other key factors to consider when evaluating a recommendation system include:

- _Diversity_: How dissimilar are the recommendations?
- _Coverage_: What percentage of the user-item space can be recommended?
- _Serendipity_: How surprising are the relevant recommendations?
- _Novelty_: How surprising are the recommendations in general?
- _Relevancy_: How relevant are the recommendations?

_Based on responses of Quora discussion [here](https://www.quora.com/How-can-I-measure-the-accuracy-of-a-recommender-system)._


**4. What types of visualizations will help me grasp the nature of the problem / data?**

Although we only implement a collaborative filter in this analysis, taking a look at the greater dataset provided, including the tags, may provide insight into the data we are working with.  For instance, what are the top artists, and what tags/genres represent these artists?  Is the dataset well balanced with different types of data, such as different genres and artists? Preliminary EDA on the tags of the dataset for implementation of a content-based component to this recommender can be found in the [Appendix notebooks](https://github.com/cipher813/recommender_system/tree/master/notebooks).


**5. What feature engineering might help improve the signal?**

In this analysis we took two steps to improve the signal within our results.  Firstly we implemented a popularity threshold where an artist required a certain number of plays to remain in the results (at the 95th percentile this was ~10,000 plays).  This is to remove noise and improve accuracy.  Further, rather than using the play count by user, we took a binary approach, as in whether a particular user played the artist (1) vs never played (0).  


**6. Which modeling techniques are good at capturing the types of relationships I see in this data?**

In this analysis we implemented K Nearest Neighbors via sci-kit learn to determine the cosine similarity distance amongst the artist in the datasets based on the underlying mapping of user play data in vector space.  We also explored various models in TuriCreate, including matrix factorization and item similarity with Jaccard similarity.  The Pearson Correlation Coefficient is also a popular distance metric.


**7. Now that I have a model, how can I be sure that I didn't introduce a bug in the code? If results are too good to be true, they probably are!**

It is important to test your results using a variety of cases, and to use assert statements and tests where possible and relevant.  When you don't understand the workings of an algorithm, well-placed print statements can work wonders in visualizing its workings. Peer review is also important prior to putting code into production!

**8. What are some of the weaknesses of the model and and how can the model be improved with additional work?**

_Add data_

Due to our popularity threshold (to reduce noise), we are only including the top 5% most played artist in the dataset.  While this is still 943 artists (~5% of 17,632 in dataset), it may be useful to add more artists to the mix while maintaining our current accuracy.  Our best option for doing so would be to use a larger dataset where we could get reasonable usage patterns from more obscure artists.  

_Improve accuracy_

While our current system is providing reasonably accurate results, there are steps that we can take to perhaps increase the accuracy even further.  Firstly, we could consider incorporating content-based features into the recommendation engine, making it more of a hybrid model rather than its current purely collaborative model.  As the dataset also includes tag info, we could do a clustering analysis to determine similarities, and ideally determine something along the lines of genres within the dataset.  We could then use this grouping information to incorporate recommendations based on genre similarity in addition to the collaborative distance metric we are using. *See [Appendix notebooks](https://github.com/cipher813/recommender_system/tree/master/notebooks) for initial topic modeling of the tags to determine clustering by genre via K Means Clustering and tSNE, as well as an LDA analysis using pyLDAvis.*  

If song audio data can be obtained, we could even consider incorporating deep learning techniques which would benefit a content-based recommender.  This could learn the underlying sound profiles of song clips to recommend songs that exhibit similar characteristics.  _As demonstrated in [Deep content-based music recommendation (NIPS 2013)](http://papers.nips.cc/paper/5004-deep-content-based-music-recommendation.pdf)._

We could also try further experimentation with our thresholds in determining our active users and popular artists in the dataset.  While we decided on a binary rating approach (played vs not played), we could re-bucketize our data and only include the highest ratings, such as 4 and above.  This would ensure the recommendation results are based on artists that a user really likes.  



_Optimize compute time_

In our KNN analysis we calculate the cosine distance of the dataset at each query.  For a larger dataset, it may optimize runtime by pre-computing the similarity matrix.  While usage patterns will change over time, they generally will remain relatively static so the underlying matrix could be refreshed periodically to stay current.

In addition, we could consider implementing our recommendation engine on a distributed computing platform such as Apache Spark. While this would not necessarily speed up the queries in our current small dataset, if we were to increase the size of our dataset and recommendation engine then using a distributed computing platform can be an optimized approach.   

### Resources
[Last.FM dataset: 92,800 artist listening records from 1892 users](https://grouplens.org/datasets/hetrec-2011/). "2nd International Workshop on Information Heterogeneity and Fusion in Recommender Systems (HetRec) 2011". grouplens.

Becker, Nick. "[Music Recommendations with Collaborative Filtering and Cosine Distance](https://beckernick.github.io/music_recommender/)." August 31, 2016.  

"[How can I measure the accuracy of a recommender system?](https://www.quora.com/How-can-I-measure-the-accuracy-of-a-recommender-system)" Quora. August 28, 2017.

Jain, Aarshay.  "[Quick Guide to Build a Recommendation Engine in Python](https://www.analyticsvidhya.com/blog/2016/06/quick-guide-build-recommendation-engine-python/)." Analytics Vidhya.  June 2, 2016.

Mean Average Precision. "[Evaluation measures (information retrieval)](https://en.wikipedia.org/wiki/Evaluation_measures_%28information_retrieval%29)." Wikipedia.  

Pierson, Lillian. "[Building a Recommendation System with Python Machine Learning & AI](https://www.linkedin.com/learning/building-a-recommendation-system-with-python-machine-learning-ai)." LinkedIn Learning. July 14,2017.  
