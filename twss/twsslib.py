import os
import sys
import datetime

from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer

class TWSS: 
    training_data = [] 
    training_value = []
    numerical_features = []
    classifier = MultinomialNB()
    is_trained = False
    stop = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around', 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de', 'describe', 'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'it', 'its', 'itself', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thickv', 'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves', 'the']
    vectorizer = CountVectorizer(min_df=1, stop_words=stop)

    def __init__(self, sentence=None, positive_corpus_file=None, negative_corpus_file=None):
        if positive_corpus_file and negative_corpus_file: 
            self.import_training_data(positive_corpus_file, negative_corpus_file)
        if sentence:
            self.__call__(sentence)

    def __call__(self, phrase):
        if not self.is_trained: 
            self.train()
            self.is_trained = True
        print self.is_twss(phrase)

    def import_training_data(self,
            positive_corpus_file=os.path.join(os.path.dirname(__file__),
                "positive.txt"),
            negative_corpus_file=os.path.join(os.path.dirname(__file__),
                "negative.txt")
            ):
        """
        This method imports the positive and negative training data from the
        two corpus files and creates the training data list. 
        """
        with open(positive_corpus_file) as file_object:
            for line in file_object:
                self.training_data.append(line.strip())
                self.training_value.append(True)

        with open(negative_corpus_file) as file_object:
            for line in file_object:
                self.training_data.append(line.strip())
                self.training_value.append(False)
        
        assert len(self.training_data) == len(self.training_value)

    def train(self): 
        """
        This method generates the classifier. This method assumes that the
        training data has been loaded
        """
        if not self.training_data: 
            self.import_training_data()
        if not self.numerical_features:
            self.numerical_features = self.vectorizer.fit_transform(self.training_data)
        self.classifier.fit(self.numerical_features.toarray(), self.training_value)

    def extract_features(self, phrase):
        """
        This function will extract features from the phrase being used. 
        Currently, the feature we are extracting are unigrams of the text corpus.
        """
        return self.vectorizer.transform([phrase]).toarray()[0]
        

    def is_twss(self, phrase):
        """
        The magic function- this accepts a phrase and tells you if it
        classifies as an entendre
        """
        featureset = self.extract_features(phrase)
        return self.classifier.predict(featureset)

    def cross_validate(self):
        """
            Used to cross_validate the algorithm.
        """
        if not self.training_data:
            self.import_training_data()
        if not self.numerical_features:
            self.numerical_features = self.vectorizer.fit_transform(self.training_data)
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            self.numerical_features.toarray(), self.training_value, test_size=0.33, random_state=42)
        self.classifier.fit(X_train, y_train)
        print self.classifier.score(X_test, y_test)