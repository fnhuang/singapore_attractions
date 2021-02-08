import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import os, sys
import random
import glob
from .preprocess import Document, Preprocessor
from . import preprocess
import csv

class SentimentAnalyzer():

    # @param safolder folder where the training set resides, and where to put the test result
    # @param model algorithm used for sentiment analysis
    def __init__(self, dir, model):
        self.dir = dir
        self.model = model
        self.preprocessor = Preprocessor(dir)

        # open existing files if a model has been built before. no need to reprocess
        self.classifier = pickle.load(open(f"{self.dir}/training/model/{model}_clf.pickle", "rb")) \
            if os.path.isfile(f"{self.dir}/training/model/{model}_clf.pickle") else None
        self.features = pickle.load(open(f"{self.dir}/training/model/features.pickle", "rb")) \
            if os.path.isfile(f"{self.dir}/training/model/features.pickle") else []
        self.classes = pickle.load(open(f"{self.dir}/training/model/classes.pickle", "rb")) \
            if os.path.isfile(f"{self.dir}/training/model/classes.pickle") else []
        self.most_common_words = [w.strip() for w in open(f"{self.dir}/training/model/most_common_words.txt","r",encoding="utf8").readlines()] \
            if os.path.isfile(f"{self.dir}/training/model/most_common_words.txt") else []

    # remove most common words (top 1%) that appear in both positive and negative documents
    def _remove_most_common_words(self, documents):
        print("Define most common words...")

        most_common_words = set([])
        for cls in self.classes:
            docs = [d for d in documents if d.sentiment == cls]

            doc_words = [w for d in docs for w in word_tokenize(d.content.replace(".", ""))]
            fdist = nltk.FreqDist(doc_words)

            if len(most_common_words) == 0:
                most_common_words = set([w[0] for w in fdist.most_common(int(0.01 * len(doc_words)))])
            else:
                most_common_words = set([w[0] for w in fdist.most_common(int(0.01 * len(doc_words))) if w[0] in most_common_words])

        self.most_common_words = most_common_words
        with open(f"{self.dir}/training/model/most_common_words.txt","w",encoding="utf8") as writer:
            writer.writelines([f"{w}\n" for w in self.most_common_words])


        ndocs = []
        doc_count = 0
        for d in documents:
            doc_count += 1
            ncontent = " ".join([w for w in word_tokenize(d.content) if w not in most_common_words])

            ndocs.append(Document(d.name, ncontent, d.sentiment, d.location))
            print("\r", end="")
            print("Removing most common words progress", int(doc_count/len(documents) * 100), "%", end="", flush=True)
        print("")
        return ndocs

    # only keep adjectives, adverbs, and nouns
    def _reduce_dimension_by_postag(self, documents):
        reduced_documents = []

        doc_count = 0
        for doc in documents:
            reduced_sentence = " ".join([p[0] for p in nltk.pos_tag(word_tokenize(doc.content.replace(".","")))
                                         if p[1] in preprocess.ADJ or p[1] in preprocess.ADV or p[1] in preprocess.NOUN])

            if not reduced_sentence.isspace():
                reduced_documents.append(Document(doc.name, reduced_sentence, doc.sentiment, doc.location))

            doc_count += 1
            print("\r",end="")
            print("Reducing dimension in progress", int(doc_count*100/len(documents)), "%", end="", flush=True)
        print("")

        return reduced_documents

    def create_frequency_plot(self, words, top_k):
        p = nltk.FreqDist(words)
        p.plot(top_k)


    def _undersample(self, documents):

        # find the minimum number of documents in a class

        docs_by_class = []
        minclass_length = len(documents)
        for cls in self.classes:
            docs = [d for d in documents if d.sentiment == cls]
            docs_by_class.append(docs)

            if len(docs) < minclass_length:
                minclass_length = len(docs)


        # sample all classes based on the minimum number of documents
        undersampled_docs = []
        for docs in docs_by_class:
            random.shuffle(docs)
            undersampled_docs.extend(docs[:minclass_length])


        return undersampled_docs


    # preprocessing
    def prepare_documents(self):
        documents = []

        for file in os.listdir(f"{self.dir}/training/data"):
            documents.extend(pickle.load(open(f"{self.dir}/training/data/{file}", "rb")))

        if len(self.classes) == 0:
            self.classes = set([doc.sentiment for doc in documents])
            pickle.dump(self.classes, open(f"{self.dir}/training/model/classes.pickle", "wb"))

        print("Perform undersampling...")
        documents = self._undersample(documents)

        documents = self._reduce_dimension_by_postag(documents)

        documents = self._remove_most_common_words(documents)


        return documents

    def transform_into_featuresets(self, documents):

        self.features = set([w for d in documents for w in set(word_tokenize(d.content))])
        pickle.dump(self.features, open(f"{self.dir}/training/model/features.pickle", "wb"))
        print("Features length:", len(self.features))
        featuresets = []

        print("Transforming into featuresets....")
        doc_count = 0
        for doc in documents:
            # checking whether a word exists in an array takes a significantly longer time
            # thus we check whether a word exists in a string
            featuresets.append(({w:True for w in word_tokenize(doc.content) if w in self.features}, doc.sentiment))
            doc_count += 1

            print("\r", end='')
            print("Preparing featureset in progress", int(doc_count*100/len(documents)),"%",end='', flush=True)
        print("")

        return featuresets

    def get_training_validation_set(self, featuresets, valid_ratio):
        if len(self.classes) == 0:
            classes = set([f[1] for f in featuresets])
            pickle.dump(self.classes, open(f"{self.dir}/training/model/classes.pickle", "wb"))

        trainingset = []
        validset = []

        for c in self.classes:
            subfeat = [f for f in featuresets if f[1] == c]
            random.shuffle(subfeat)

            trainct = int((1-valid_ratio) * len(subfeat))
            trainingset.extend(subfeat[:trainct])
            validset.extend(subfeat[trainct:])

        return trainingset, validset

    def train(self, validation_ratio):
        os.makedirs(os.path.dirname(f"{self.dir}/training/model/"), exist_ok=True)

        documents = self.prepare_documents()
        featuresets = self.transform_into_featuresets(documents)
        trainset, validset = self.get_training_validation_set(featuresets, validation_ratio)

        print("Building classifier...")
        if self.model == "NB":
            self.classifier = nltk.NaiveBayesClassifier.train(trainset)
            self.classifier.show_most_informative_features(15)
        elif self.model == "MNB":
            self.classifier = SklearnClassifier(MultinomialNB()).train(trainset)
        elif self.model == "SVM":
            self.classifier = SklearnClassifier(SVC()).train(trainset)
        elif self.model == "LR":
            self.classifier = SklearnClassifier(LogisticRegression()).train(trainset)

        print("Accuracy per class")
        for cls in self.classes:
            print(f"{cls} accuracy:",
                  (nltk.classify.accuracy(self.classifier, [v for v in validset if v[1] == cls])) * 100)
        print("Classifier accuracy percent:", (nltk.classify.accuracy(self.classifier, validset)) * 100)
        pickle.dump(self.classifier, open(f"{self.dir}/training/model/{self.model}_clf.pickle", "wb"))

    def show_most_informative_features(self,n):
        self.classifier.show_most_informative_features(n)

    def sentiment(self, text):
        # to ensure that the word is lemmatized properly so it is detected in self.features
        cleaned_text = self.preprocessor.basic_preprocess(text).replace(".","")

        # no need advanced self processing because the features have been determined
        feature = {w:True for w in word_tokenize(cleaned_text) if w in self.features}
        prob_dict = self.classifier.prob_classify(feature)

        cls = prob_dict.max()
        prob = prob_dict.prob(cls)

        return cls, prob


    def classify(self, test_dir):
        print("Start classifying...")

        if self.classifier == None:
            self.train(0.2)
        else:
            self.classifier.show_most_informative_features(15)

        files = [os.path.basename(x) for x in glob.glob(f"{self.dir}/{test_dir}/data/*.csv")]
        done_files = [f.strip() for f in open(f"{self.dir}/testing/classify_done.txt", 'r').readlines()] \
            if os.path.isfile(f"{self.dir}/testing/classify_done.txt") else []
        tbp_files = [f for f in files if f not in done_files]

        headers = ["review_page","review_title","review_content","review_star",
                                        "reviewer_location","review_date","crawled_date"]
        os.makedirs(os.path.dirname(f"{self.dir}/{test_dir}/results/"), exist_ok=True)

        for file in tbp_files:
            with open(f"{self.dir}/{test_dir}/data/{file}", "r", encoding="utf8") as f:
                csvreader = csv.DictReader(f)

                with open(f"{self.dir}/{test_dir}/results/{file}","w", encoding="utf8", newline="") \
                        as w:
                    csvwriter = csv.writer(w)
                    csvwriter.writerow(headers)

                    rowid = 0
                    rownum = self.preprocessor.count_lines(f"{self.dir}/{test_dir}/data/{file}")
                    for row in csvreader:
                        review_page = row["review_page"]
                        review_title = row["review_title"]
                        review_content = row["review_content"]

                        cat = self.sentiment(f"{row['review_title']}. {row['review_content']}")
                        review_star = "45" if cat[0] == "pos" else "20"

                        reviewer_location = row["user_location"]
                        review_date = row["review_date"]
                        crawled_date = "00000000"

                        csvwriter.writerow([review_page, review_title, review_content, review_star,
                                            reviewer_location, review_date, crawled_date])

                        w.flush()

                        rowid += 1
                        print("\r", end='')
                        print("Classifying in progress",int(rowid*100/rownum),"% for",file, end='', flush=True)

            with open(f"{self.dir}/testing/classify_done.txt","a",encoding="utf8") as writer:
                writer.write(f"{file}\n")



if __name__ == "__main__":
    #dir = "sentiment_analysis"
    #model = "NB"
    #test_dir = "testing"

    dir = sys.argv[1]
    model = sys.argv[2]
    test_dir = sys.argv[3]

    saz = SentimentAnalyzer(dir, model)
    saz.train(0.2)
    #saz.classify(test_dir)

