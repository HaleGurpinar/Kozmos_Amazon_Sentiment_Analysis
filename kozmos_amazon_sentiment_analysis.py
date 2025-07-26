##################################################
# SENTIMENT ANALYSIS and SENTIMENT MODELING for AMAZON REVIEWS
##################################################

##################################################
# Business Problem
##################################################
# Kozmos, which sells home textiles and everyday clothing products on Amazon,
#  aims to increase it's sales by analyzing customer reviews and improving product features based on the complaints it receives.
# To achieve this goal, sentiment analysis will be performed on reviews, which will then be tagged, and a classification model will be created using the tagged data.

##################################################
# Data Set Story
##################################################
# The data set includes variables about product comments, comment titles, star ratings, and usefulness votes.
# Review     # Title     # HelpFul     # Star

##############################################################
# TASKS
##############################################################

# TASK 1: Text preprocessing steps.
#         # 1. Read the amazon.xlsx data.
#         # 2. On the “Review” variable
#             # a. Convert all letters to lowercase
#             # b. Remove punctuation marks
#             # c. Remove numerical expressions found in reviews
#             # d. Remove stopwords from the data
#             # e. Remove words that appear less than 1,000 times from the data
#             # f. Apply lemmatization.
#

# TASK 2: Text Visualization
#         # Step 1: Bar plot visualization process
#            # a. Calculate the frequencies of the words contained in the “Review” variable and save them as tf
#            # b. Rename the columns of the tf dataframe as “words” and ‘tf’
#            # c. Filter the “tf” variable for values greater than 500 and complete the bar plot visualization process.
#
#        # Step 2: WordCloud visualization process
#            # a. Save all words contained in the “Review” variable as a string named “text”
#            # b. Use WordCloud to define and save your template shape
#            # c. Generate the saved wordcloud using the string created in the first step.
#            # d. Complete the visualization steps. (figure, imshow, axis, show)
#

# TASK 3: Sentiment Analysis
#       # Step 1: Create a SentimentIntensityAnalyzer object defined in the NLTK package in Python
#
#       # Step 2: Examine polarity scores using the SentimentIntensityAnalyzer object
#           # a. Calculate polarity_scores() for the first 10 observations of the “Review” variable
#           # b. Filter the first 10 observations based on compound scores and review them again
#           # c. If the compound scores for the 10 observations are greater than 0, update them as “pos”; otherwise, update them as ‘neg’
#           # d. Assign pos-neg values to all observations in the “Review” variable and add them to the dataframe as a new variable
#
# # NOTE: By labeling the comments with SentimentIntensityAnalyzer,
#          a dependent variable has been created for the comment classification machine learning model.


# TASK 4: Preparing for machine learning!
#   Step 1: Separate the data into train and test sets by identifying the dependent and independent variables.

#   Step 2: We need to convert the data into numerical form so that we can feed it into the machine learning model.
#     a. Create an object using TfidfVectorizer.
#     b. Fit the object we created using the train data we previously separated.
#     c. Apply the transform operation to the train and test data using the vector we created and save it.

# TASK 5: Modeling (Logistic Regression)
# Step 1: Build a logistic regression model and fit it with the train data.

#  # Step 2: Perform prediction operations with the model you have built.
#      a. Predict the test data with the predict function and save it.
#      b. Report and observe your prediction results with classification_report.
#      c. Calculate the average accuracy value using the cross validation function.

# # Step 3: Randomly select comments from the data and ask the model about them.
#     a. Select a sample from the “Review” variable using the sample function and assign it a new value.
#     b. Vectorize the sample you obtained using CountVectorizer so that the model can predict it.
#     c. Save the vectorized sample by performing the fit and transform operations.
#     d. Provide the sample to the model you have built and save the prediction result.
#     e. Print the sample and prediction result to the screen.

# TASK 6: Modelling (Random Forest)
# # Step 1: Observe the prediction results of the Random Forest model.
#    a. Set up and fit the RandomForestClassifier model.
#    b. Calculate the average accuracy value using the cross-validation function.
#    c. Compare the results with the logistic regression model.

# ###########################################################################################################################


import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from textblob import Word
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
from warnings import filterwarnings

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 200)

##############################################################
# # TASK 1: TEXT PREPROCESSING
##############################################################
#  # 1. Read the amazon.xlsx data.
#  # 2. On the “Review” variable
#         # a. Convert all letters to lowercase
#         # b. Remove punctuation marks
#         # c. Remove numerical expressions found in reviews
#         # d. Remove stopwords from the data
#         # e. Remove words that appear less than 1,000 times from the data
#         # f. Apply lemmatization.

df = pd.read_excel('amazon.xlsx', engine='openpyxl')
print(df.head())

###############################
# Normalizing Case Folding
###############################
df['Review'] = df['Review'].str.lower()

###############################
# Punctuations
###############################
df['Review'] = df['Review'].str.replace('[^\w\s]', '')

###############################
# Numbers
###############################
df['Review'] = df['Review'].str.replace('\d', '')

###############################
# Stopwords
###############################
# nltk.download('stopwords')
sw = stopwords.words('english')
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

###############################
# Rarewords / Custom Words
###############################

sil = pd.Series(' '.join(df['Review']).split()).value_counts()[-1000:]
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))

###############################
# Lemmatization
###############################

# nltk.download('wordnet')
df['Review'] = df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

df['Review'].head(10)

##############################################################
# # TASK 2: TEXT VISUALIZATION
##############################################################

###############################
# Bar plot
###############################
# # Step 1: Bar plot visualization process
#         # a. Calculate the frequencies of the words contained in the “Review” variable and save them as tf
#         # b. Rename the columns of the tf dataframe as “words” and ‘tf’
#         # c. Filter the “tf” variable for values greater than 500 and complete the bar plot visualization process.
tf = df["Review"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

###############################
# Wordcloud
###############################
# Step 2: WordCloud visualization process
#            # a. Save all words contained in the “Review” variable as a string named “text”
#            # b. Use WordCloud to define and save your template shape
#            # c. Generate the saved wordcloud using the string created in the first step.
#            # d. Complete the visualization steps. (figure, imshow, axis, show)

text = " ".join(i for i in df.Review)

wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

##############################################################
# TASK 3: SENTIMENT ANALYSIS
##############################################################

# # Step 1: Create a SentimentIntensityAnalyzer object defined in the NLTK package in Python
sia = SentimentIntensityAnalyzer()

# Step 2: Examine polarity scores using the SentimentIntensityAnalyzer object
#       # a. Calculate polarity_scores() for the first 10 observations of the “Review” variable
#       # b. Filter the first 10 observations based on compound scores and review them again
#       # c. If the compound scores for the 10 observations are greater than 0, update them as “pos”; otherwise, update them as ‘neg’
#       # d. Assign pos-neg values to all observations in the “Review” variable and add them to the dataframe as a new variable
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x))

df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["Sentiment_Label"] = df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df.groupby("Sentiment_Label")["Star"].mean()

# NOTE: By labeling the comments with SentimentIntensityAnalyzer,
#        a dependent variable has been created for the comment classification machine learning model.


###############################
# TASK 4: PREPARING for MACHINE LEARNING
###############################
# Step 1: Separate the data into train and test sets by identifying the dependent and independent variables.

# Test-Train
train_x, test_x, train_y, test_y = train_test_split(df["Review"],
                                                    df["Sentiment_Label"],
                                                    random_state=42)

# Step 2: We need to convert the data into numerical form so that we can feed it into the machine learning model.
# a. Create an object using TfidfVectorizer.
# b. Fit the object we created using the train data we previously separated.
# c. Apply the transform operation to the train and test data using the vector we created and save it.

# TF-IDF Word Level
tf_idf_word_vectorizer = TfidfVectorizer().fit(train_x)
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

###############################
# TASK 5: MODELLING (Logistic Regression)
###############################

# Step 1: Build a logistic regression model and fit it with the train data.

log_model = LogisticRegression().fit(x_train_tf_idf_word, train_y)

# # Step 2: Perform prediction operations with the model you have built.
#      a. Predict the test data with the predict function and save it.
#      b. Report and observe your prediction results with classification_report.
#      c. Calculate the average accuracy value using the cross validation function.

y_pred = log_model.predict(x_test_tf_idf_word)

print(classification_report(y_pred, test_y))

cross_val_score(log_model, x_test_tf_idf_word, test_y, cv=5).mean()

# # Step 3: Randomly select comments from the data and ask the model about them.
#      a. Select a sample from the “Review” variable using the sample function and assign it a new value.
#      b. Vectorize the sample you obtained using CountVectorizer so that the model can predict it.
#      c. Save the vectorized sample by performing the fit and transform operations.
#      d. Provide the sample to the model you have built and save the prediction result.
#      e. Print the sample and prediction result to the screen.


random_review = pd.Series(df["Review"].sample(1).values)
yeni_yorum = CountVectorizer().fit(train_x).transform(random_review)
pred = log_model.predict(yeni_yorum)
print(f'Review:  {random_review[0]} \n Prediction: {pred}')

###############################
# TASK 6: MODELLING (Random Forest)
###############################
# # Step 1: Observe the prediction results of the Random Forest model.
#    a. Set up and fit the RandomForestClassifier model.
#    b. Calculate the average accuracy value using the cross-validation function.
#    c. Compare the results with the logistic regression model.

rf_model = RandomForestClassifier().fit(x_train_tf_idf_word, train_y)
cross_val_score(rf_model, x_test_tf_idf_word, test_y, cv=5, n_jobs=-1).mean()
