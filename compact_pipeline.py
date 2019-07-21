
# coding: utf-8

# In[86]:

# Import required packages
import pandas as pd
from pprint import pprint
from time import time
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[37]:

# Function to load Data
def get_data(file):
    """Function to import Data and combine
    Relevent and Non_Relevant Sentences with labels 1 and 0 respectively"""
    Relevant = pd.read_excel(file,sheet_name=0,usecols=0)
    Non_Relevant = pd.read_excel(file,sheet_name=1,usecols=0)
    Relevant['label'] = 1
    Non_Relevant['label'] = 0
    combined = pd.concat([Relevant,Non_Relevant], ignore_index=True)
    return combined


# In[38]:

# Call the get_data Function to load the data from excel
Data = get_data('Data.xlsx')


# In[62]:

# Splitting Data into training and test sets
X,y = Data.iloc[:,0], Data.iloc[:,1]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,y,test_size = 0.2,random_state=42 )


# In[79]:

# Create Pipeline object to combine TfidfVectorizer and a Classifier
pipeline = Pipeline([
    ('vect',TfidfVectorizer()),
    ('clf',RandomForestClassifier())
])


# In[89]:

# Parameter grid to find the best peforming parameters for Vectorizer
# and classifer
parameters = {
    'vect__stop_words': (None, 'english'),
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'vect__use_idf': (True, False),
    'vect__norm': ('l1', 'l2'),
#     'clf__n_estimators': (100,200,300)
}


# In[90]:

# GridSearchCV object to find best performing parameters
if __name__ == '__main__':
    grid_search = GridSearchCV(pipeline,parameters,cv=5,
                               n_jobs =-1,verbose=1
    )

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(Xtrain, ytrain)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


# In[93]:

# Model evaluation
grid_search.score(Xtest,ytest)
