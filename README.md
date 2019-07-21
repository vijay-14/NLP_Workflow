# NLP_Workflow
NLP workflow using pipeline and gridsearchcv objects to build reusable and extensible projects.
Features of this program:
1. Loading the data from excel notebooks(default), you can change few lines of code to load data from other file types.
2. Splitting the data into train test splits so that you can validate your models.
3. Preprocessing and feature extraction using TFidfvectorizer.
4. Using pipeline objects we can combine the preprocessing and classifier(RandomForestclassfier - default).
5. To find the best parameters for preprocessing and classification we can use Gridsearchcv.
