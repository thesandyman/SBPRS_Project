import numpy as np
import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
import joblib


model_load = joblib.load("./models/sbprs_model.pkl")
tfidf_clean_model = joblib.load("./models/tfidf_vec_clean.pkl")
tfidf_title_model = joblib.load("./models/tfidf_vec_title.pkl")

def preprocess(username):
    data = pd.read_csv("data.csv")
    ratings = data[['name', 'reviews_username', 'reviews_rating']]

    ratings = ratings.sort_values(by=['reviews_username', 'name', 'reviews_rating'], ascending=False)
    ratings = ratings.drop_duplicates(subset=['reviews_username', 'name'], keep='first')

    # Test and Train split of the dataset.
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(ratings, test_size=0.20, random_state=31)

    # Pivot the train ratings' dataset into matrix format in which columns are products and the rows are user IDs.
    df_pivot = train.pivot(
        index='reviews_username',
        columns='name',
        values='reviews_rating'
    ).fillna(0)

    # ### Creating dummy train & dummy test dataset
    # These dataset will be used for prediction
    # - Dummy train will be used later for prediction of the products which has not been rated by the user. To ignore the products rated by the user, we will mark it as 0 during prediction. The products not rated by user is marked as 1 for prediction in dummy train dataset.
    #
    # - Dummy test will be used for evaluation. To evaluate, we will only make prediction on the products rated by the user. So, this is marked as 1. This is just opposite of dummy_train.

    # Copy the train dataset into dummy_train
    dummy_train = train.copy()

    # The products not rated by user is marked as 1 for prediction.
    dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x >= 1 else 1)

    # In[123]:

    # Convert the dummy train dataset into matrix format.
    dummy_train = dummy_train.pivot(
        index='reviews_username',
        columns='name',
        values='reviews_rating'
    ).fillna(1)

    # **Cosine Similarity**
    #
    # Cosine Similarity is a measurement that quantifies the similarity between two vectors [Which is Rating Vector in this case]
    #
    # **Adjusted Cosine**
    #
    # Adjusted cosine similarity is a modified version of vector-based similarity where we incorporate the fact that different users have different ratings schemes. In other words, some users might rate items highly in general, and others might give items lower ratings as a preference. To handle this nature from rating given by user , we subtract average ratings for each user from each user's rating for different productss.

    # # Item Based Similarity

    # Taking the transpose of the rating matrix to normalize the rating around the mean for different movie ID. In the user based similarity, we had taken mean for each user instead of each product.

    df_pivot = train.pivot(
        index='reviews_username',
        columns='name',
        values='reviews_rating'
    ).T

    # ##### Normalising the product rating for each product for using the Adujsted Cosine

    mean = np.nanmean(df_pivot, axis=1)
    df_subtracted = (df_pivot.T - mean).T

    # ##### Finding the cosine similarity using pairwise distances approach
    from sklearn.metrics.pairwise import pairwise_distances

    # Item Similarity Matrix
    item_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
    item_correlation[np.isnan(item_correlation)] = 0

    # ##### Filtering the correlation only for which the value is greater than 0. (Positively correlated)

    item_correlation[item_correlation < 0] = 0

    # # Prediction - Item Item

    item_predicted_ratings = np.dot((df_pivot.fillna(0).T), item_correlation)

    # ### Filtering the rating only for the movies not rated by the user for recommendation

    item_final_rating = np.multiply(item_predicted_ratings, dummy_train)

    # ### Finding the top 20 recommendation for the *user*

    # Take the username as input
    user_input = username

    # Recommending the Top 20 products to the user.
    d = item_final_rating.loc[user_input].sort_values(ascending=False)[0:20]

    # Mapping with product
    product_mapping = pd.read_csv("sample30.csv", encoding='ascii')

    d = pd.merge(d, product_mapping, left_on='name', right_on='name', how='left')
    # ### User Based Similarity gave a RSME score of 2.43 and Item Based Similarity gave a RMSE score of 3.6

    # #### Hence going with Item Based Similarity recommendation engine

    product_df = d

    # User_sentiment is influenced by two data points mainly
    # - Reviews
    # - Reviews title

    # #### Performing text preprocessing on reviews_text column for the new data frame

    # ##### Lot of noise in the reviews like '...','--', spelling mistakes, caps etc and words with contraction

    # Dictionary of English Contractions
    contractions_dict = {"ain't": "are not", "'s": " is", "aren't": "are not",
                         "can't": "cannot", "can't've": "cannot have",
                         "'cause": "because", "could've": "could have", "couldn't": "could not",
                         "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not",
                         "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
                         "hasn't": "has not", "haven't": "have not", "he'd": "he would",
                         "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have",
                         "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                         "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                         "I'll've": "I will have", "I'm": "I am", "I've": "I have", "isn't": "is not",
                         "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
                         "it'll've": "it will have", "let's": "let us", "ma'am": "madam",
                         "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                         "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                         "mustn't've": "must not have", "needn't": "need not",
                         "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
                         "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                         "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                         "she'll": "she will", "she'll've": "she will have", "should've": "should have",
                         "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                         "that'd": "that would", "that'd've": "that would have", "there'd": "there would",
                         "there'd've": "there would have", "they'd": "they would",
                         "they'd've": "they would have", "they'll": "they will",
                         "they'll've": "they will have", "they're": "they are", "they've": "they have",
                         "to've": "to have", "wasn't": "was not", "we'd": "we would",
                         "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                         "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
                         "what'll've": "what will have", "what're": "what are", "what've": "what have",
                         "when've": "when have", "where'd": "where did", "where've": "where have",
                         "who'll": "who will", "who'll've": "who will have", "who've": "who have",
                         "why've": "why have", "will've": "will have", "won't": "will not",
                         "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                         "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                         "y'all'd've": "you all would have", "y'all're": "you all are",
                         "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have",
                         "you'll": "you will", "you'll've": "you will have", "you're": "you are",
                         "you've": "you have"}

    # Regular expression for finding contractions
    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

    # Function for expanding contractions
    def expand_contractions(text, contractions_dict=contractions_dict):
        def replace(match):
            return contractions_dict[match.group(0)]

        return contractions_re.sub(replace, text)

    # Expanding Contractions in the reviews
    product_df['reviews_text'] = product_df['reviews_text'].apply(lambda x: expand_contractions(x))

    # ##### Lowercasing the reviews
    product_df['cleaned'] = product_df['reviews_text'].apply(lambda x: x.lower())

    # ##### Removing words containing digits
    product_df['cleaned'] = product_df['cleaned'].apply(lambda x: re.sub('\w*\d\w*', ' ', x))

    # ##### Remove Punctuations
    product_df['cleaned'] = product_df['cleaned'].apply(
        lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x))

    # ##### Removing extra spaces
    product_df['cleaned'] = product_df['cleaned'].apply(lambda x: re.sub(' +', ' ', x))

    # ##### replace the phrase "this review was collected as part of a promotion"

    product_df['cleaned'] = product_df['cleaned'].str.replace("this review was collected as part of a promotion", ' ')

    # ##### Creating a function for stop words and stemming

    from nltk.stem.porter import PorterStemmer
    from nltk.corpus import stopwords

    stemmer = PorterStemmer()

    def preprocess(document):
        # change sentence to lower case
        # document = document.lower()

        # tokenize into words
        words = word_tokenize(document)

        # remove stop words
        words = [word for word in words if word not in stopwords.words("english")]

        # stem
        # words = [stemmer.stem(word) for word in words]

        # join words to make sentence
        document = " ".join(words)

        return document

    product_df['cleaned'] = product_df['cleaned'].apply(lambda x: preprocess(x))

    # ### Performing text preprocessing on reviews_title column
    product_df['reviews_title'] = product_df['reviews_title'].apply(str)

    # ##### Lot of noise in the titles like '...','--', spelling mistakes, caps etc and words with contraction
    # calling contraction function which is defined earlier
    # Expanding Contractions in the titles
    product_df['reviews_title'] = product_df['reviews_title'].apply(lambda x: expand_contractions(x))

    # ##### Lower casing all the titles
    product_df['title_cleaned'] = product_df['reviews_title'].apply(lambda x: x.lower())

    # ##### Removing words containing digits
    product_df['title_cleaned'] = product_df['title_cleaned'].apply(lambda x: re.sub('\w*\d\w*', ' ', x))

    # ##### Removing punctuations
    product_df['title_cleaned'] = product_df['title_cleaned'].apply(
        lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x))

    # ##### Removing extra spaces

    product_df['title_cleaned'] = product_df['title_cleaned'].apply(lambda x: re.sub(' +', ' ', x))

    product_df['title_cleaned'] = product_df['title_cleaned'].apply(lambda x: preprocess(x))

    cleaned_tfidf_vec = tfidf_clean_model.transform(product_df['cleaned'])
    title_tfidf_vec = tfidf_title_model.transform(product_df['title_cleaned'])

    prod_clean_df = pd.DataFrame(cleaned_tfidf_vec.todense(), columns=tfidf_clean_model.get_feature_names())

    prod_title_df = pd.DataFrame(title_tfidf_vec.todense(), columns=tfidf_title_model.get_feature_names())

    final_prod_df = pd.concat([prod_clean_df, prod_title_df], axis=1)

    product_df['predict_sentiments'] = model_load.predict(final_prod_df)

    prod_rec = product_df[['name', 'id', 'reviews_text', 'reviews_title', 'predict_sentiments']]

    prod_rec['predict_sentiments'] = prod_rec['predict_sentiments'].replace(['Positive'], '1')
    prod_rec['predict_sentiments'] = prod_rec['predict_sentiments'].replace(['Negative'], '0')

    prod_rec['predict_sentiments'] = prod_rec['predict_sentiments'].apply(pd.to_numeric)

    df = prod_rec.groupby(['name', 'id']).agg({'predict_sentiments': ['mean']})
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df = df.sort_values(by='predict_sentiments_mean', ascending=False).reset_index()

    #top 5 products
#    x = df['name'].iloc[:5].to_string(index=False)
    x = df['name'].iloc[:5]
    return x