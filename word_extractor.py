from preprocess import Document
import os, pickle, sys
from nltk import FreqDist
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import numpy as np
from sentiment_analysis import SentimentAnalyzer
import pandas as pd
import preprocess
import argparse
import logging

logging.basicConfig(level=logging.INFO)

class WordExtractor():
    #get non as ovs

    def __init__(self, dir, sentiment, sentiment_dir, sentiment_model):
        self.dir = dir
        self.sentiment = sentiment

        self.sentiment_classifier = SentimentAnalyzer(sentiment_dir, sentiment_model)

        self.id2doc = pickle.load(open(f"{self.dir}/{self.sentiment}_model/id2doc.pickle","rb")) \
            if os.path.isfile(f"{self.dir}/{self.sentiment}_model/id2doc.pickle") else {}
        self.tfidf_vectorizer = pickle.load(open(f"{self.dir}/{self.sentiment}_model/tfidf_vectorizer.pickle", "rb")) \
            if os.path.isfile(f"{self.dir}/{self.sentiment}_model/tfidf_vectorizer.pickle") else []
        self.tfidf_matrix = pickle.load(open(f"{self.dir}/{self.sentiment}_model/tfidf_matrix.pickle","rb")) \
            if os.path.isfile(f"{self.dir}/{self.sentiment}_model/tfidf_matrix.pickle") else []
        self.kmeans_model = pickle.load(open(f"{self.dir}/{self.sentiment}_model/kmeans_model.pickle", "rb")) \
            if os.path.isfile(f"{self.dir}/{self.sentiment}_model/kmeans_model.pickle") else None


    def prepare_documents_by_group(self, filter_xtrim):
        subdocs = []

        # preprocessed data from preprocess.py is processed further
        file_count = 0
        files = os.listdir(f"{self.dir}/data/")
        for file in files:
            docs = pickle.load(open(f"{self.dir}/data/{file}", "rb"))
            docs = [d for d in docs if d.sentiment == self.sentiment]

            for doc in docs:

                cust_sentence = self.get_custom_ngram(sent_tokenize(doc.content))
                if not cust_sentence.isspace():
                    subdocs.append(Document(doc.name, cust_sentence, doc.sentiment, doc.location))

            file_count += 1
            logging.info("\r", end="")
            logging.info("Getting custom words", int(file_count * 100 / len(files)), "%", end="", flush=True)
        logging.info("")

        logging.info("Combining documents of the same location...")
        ndocs = self._combine_documents_of_the_same_location(subdocs)

        logging.info("Combining documents of the same tags...")
        ndocs = self._combine_documents_of_the_same_group(ndocs)

        logging.info("Removing most common words")
        ndocs = self.remove_most_common_words(ndocs, filter_xtrim)

        return ndocs

    # preprocessing
    def prepare_documents(self, filter_xtrim):
        subdocs = []

        # preprocessed data from preprocess.py is processed further
        file_count = 0
        files = os.listdir(f"{self.dir}/data/")
        for file in files:
            docs = pickle.load(open(f"{self.dir}/data/{file}", "rb"))
            docs = [d for d in docs if d.sentiment == self.sentiment]

            for doc in docs:

                cust_sentence = self.get_custom_ngram(sent_tokenize(doc.content))
                if not cust_sentence.isspace():
                    subdocs.append(Document(doc.name, cust_sentence, doc.sentiment, doc.location))

            file_count += 1
            logging.info("\r", end="")
            logging.info("Getting custom words", int(file_count * 100 / len(files)), "%", end="", flush=True)
        logging.info("")

        logging.info("Combining documents of the same location...")
        ndocs = self._combine_documents_of_the_same_location(subdocs)

        logging.info("Removing most common words")
        ndocs = self.remove_most_common_words(ndocs, filter_xtrim)

        return ndocs


    def _retrieve_most_common_words(self, docs):
        all_words = []
        for doc in docs:
            # non-repeating words in a document
            all_words.extend(set([w for w in word_tokenize(doc.content.replace(".",""))]))
        docfdist = FreqDist(all_words)

        all_words = set(all_words)
        with open(f"{self.dir}/{self.sentiment}_model/most_common_words.txt","w",encoding="utf8") as writer:
            writer.writelines([f"{p}\n" for p in docfdist .most_common(len(all_words))])

        return docfdist

    def _combine_documents_of_the_same_group(self, documents):
        basic_info = self.get_basic_file_info("basic_info.csv")

        tags = set(basic_info["tag"])
        locations = set([d.location for d in documents])

        combined_docs = []
        for loc in locations:
            for tag in tags:
                # do something
                files = [row["csv"] for index, row in basic_info.iterrows() if row["tag"] == tag]
                docs = [d for d in documents if d.name in files and d.location == loc]

                combined_content = ""
                for doc in docs:
                    combined_content += doc.content + " "
                combined_content.replace(".", "")
                combined_content.strip()

                if not combined_content.isspace() and combined_content != "":
                    combined_docs.append(Document(f"{tag}.csv", combined_content, self.sentiment, loc))

        return combined_docs


    def _combine_documents_of_the_same_location(self, documents):

        ldocs = []
        for d in documents:
            if d.location == "non":
                ldocs.append(Document(d.name, d.content, d.sentiment, "ovs"))
            else:
                ldocs.append(d)

        combined_docs = []

        locations = set([d.location for d in ldocs])
        names = set([d.name for d in ldocs])

        for loc in locations:
            for name in names:
                docs = [d for d in ldocs if d.location == loc and d.name == name]
                combined_content = ""
                for doc in docs:
                    combined_content += doc.content + " "
                combined_content.replace(".","")
                combined_content.strip()

                if not combined_content.isspace():
                    combined_docs.append(Document(name, combined_content, self.sentiment, loc))
        return combined_docs

    #remove words that exist in more than filter_xtrim% of # documents
    def remove_most_common_words(self, documents, filter_xtrim):
        docfdist = self._retrieve_most_common_words(documents)

        most_common_words = [w[0] for w in docfdist.most_common(len(docfdist))
                             if w[1] > int(filter_xtrim * len(documents))]

        ndocs = []
        for doc in documents:
            ncontent = " ".join([w for w in word_tokenize(doc.content) if w not in most_common_words])

            ndocs.append(Document(doc.name, ncontent, doc.sentiment, doc.location))
        return ndocs


    def get_custom_ngram(self, sentences):
        custom_words = []

        for s in sentences:
            s = s.replace(".","")
            words = [w for w in word_tokenize(s.replace(".","")) if w != ""]
            pos_tags = nltk.pos_tag(words)
            taken_indices = []

            #allowed_type = [(preprocess.ADJ,preprocess.NOUN),(preprocess.NOUN, preprocess.NOUN)]
            allowed_type = [(preprocess.ADJ, preprocess.NOUN)]

            for at in allowed_type:
                for i in range(0, len(pos_tags)-1):
                    if i not in taken_indices and i+1 not in taken_indices:
                        if (pos_tags[i][1] in at[0] and pos_tags[i + 1][1] in at[1]) :
                            if pos_tags[i][0] != "" and pos_tags[i+1][0] != "":
                                custom_words.append(f"{pos_tags[i][0]}_{pos_tags[i+1][0]}")
                                taken_indices.extend([i, i+1])

            for i in range(0, len(pos_tags)-1):
                if i not in taken_indices and i+1 not in taken_indices:
                    if (pos_tags[i][1] in preprocess.ADJ and pos_tags[i + 1][1] not in preprocess.ADJ + preprocess.ADV) :
                        if pos_tags[i][0] != "" and pos_tags[i+1][0] != "":
                            custom_words.append(f"{pos_tags[i][0]}_{pos_tags[i+1][0]}")
                            taken_indices.extend([i, i+1])


            for id in [i for i in range(0, len(pos_tags)) if i not in taken_indices]:
                #if pos_tags[id][1] in preprocess.NOUN or pos_tags[id][1] in preprocess.ADJ:
                if pos_tags[id][1] in preprocess.ADJ:
                    custom_words.append(pos_tags[id][0])

        return " ".join(custom_words)

    def transform_into_featuresets(self, documents):
        self.id2doc = {}

        doc_id = 0
        corpus = []

        for doc in documents:
            corpus.append(doc.content.replace(".",""))
            self.id2doc[doc_id] = doc
            doc_id += 1
        pickle.dump(self.id2doc, open(f"{self.dir}/{self.sentiment}_model/id2doc.pickle", "wb"))

        self.tfidf_vectorizer = TfidfVectorizer(token_pattern=r'[a-z]+_[a-z]+|[a-z]+')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        pickle.dump(self.tfidf_vectorizer, open(f"{self.dir}/{self.sentiment}_model/tfidf_vectorizer.pickle", "wb"))
        pickle.dump(self.tfidf_matrix,open(f"{self.dir}/{self.sentiment}_model/tfidf_matrix.pickle", "wb"))


    def get_basic_file_info(self, file_name):
        info = pd.read_csv(f"{self.dir}/{file_name}")

        if 'name' in info.columns:
            info["name"] = info["name"].str.lower()
        info["tag"] = info["tag"].str.lower()

        if 'csv' not in info.columns:
            info["csv"] = info["url"].map(lambda url: url[url.rindex("/"):].split("-"))
            info["csv"] = info["csv"].map(lambda url: f"{url[len(url)-2].lower()}.csv")

        to_be_dropped = ["url","starting page","name"]
        for col in to_be_dropped:
            if col in info.columns:
                info = info.drop([col], axis=1)

        return info


    def rank_words(self, dict1):
        wdiff = []

        words = [w for w in dict1.keys()]

        for w in words:
            p = dict1[w]

            # w prob is l2 normalized, the sum of square of each word in a doc add up to 1
            # thus, we take the difference in the square value
            diff = p**2
            wdiff.append((w, diff))

        wdiff = sorted(wdiff, key=lambda x: x[1], reverse=True)

        return wdiff


    def run(self, filter_xtrim):

        if len(self.id2doc) == 0:
            os.makedirs(os.path.dirname(f"{self.dir}/{self.sentiment}_model/"), exist_ok=True)
            documents = self.prepare_documents_by_group(filter_xtrim)
            self.transform_into_featuresets(documents)


        os.makedirs(os.path.dirname(f"{self.dir}/{self.sentiment}_results/local_v_foreign/"), exist_ok=True)

        doc_pairs = {}
        doc_count = 0
        for doc_num in self.id2doc:
            doc_count += 1
            doc = self.id2doc[doc_num]

            # get word probability distribution - it is l2 normalized
            prob_dist = csr_matrix.toarray(self.tfidf_matrix[doc_num, :])[0]

            # get term (features) and its probability in descending format
            sorted_indices = np.argsort(prob_dist)[::-1]
            sorted_features = np.array(self.tfidf_vectorizer.get_feature_names())[sorted_indices]

            temp = [i for i in prob_dist]
            temp.sort()
            sorted_prob = temp[::-1]

            word_prob = list(zip(sorted_features, sorted_prob))

            # keep words with probability more than 0 and sentiment prob is larger than 0.6
            rep_words = []
            for w in [w for w in word_prob if w[1] > 0]:
                sent = " ".join(w[0].split("_"))
                sent_prob = self.sentiment_classifier.sentiment(sent)

                if sent_prob[0] == self.sentiment and sent_prob[1] > 0.6:
                    rep_words.append(w)

            doc_pairs.setdefault(doc.name,[]).append((doc.location, rep_words))

            logging.info("\r",end="")
            logging.info("Getting relevant sentimental words", int(doc_count/len(self.id2doc) * 100), "percent", end="", flush=True)

        # local-foreign review difference
        doc_count = 0
        for doc_name in doc_pairs:
            doc_count += 1

            local_pdist = []
            foreign_pdist = []

            # find unique words
            for loc_prob_tuple in doc_pairs[doc_name]:
                if loc_prob_tuple[0] == "sgp":
                    local_pdist = loc_prob_tuple[1]
                else:
                    foreign_pdist = loc_prob_tuple[1]

            local_dict = {k: v for (k, v) in local_pdist}
            foreign_dict = {k: v for (k, v) in foreign_pdist}

            wdiff = self.rank_words(local_dict)
            filename = doc_name.replace(".csv", "") + "_sgp.csv"
            with open(f"{self.dir}/{self.sentiment}_results/local_v_foreign/{filename}","w",
                      encoding="utf8") as writer:
                writer.writelines([f"{w[0]},{w[1]}\n" for w in wdiff])


            wdiff = self.rank_words(foreign_dict)
            filename = doc_name.replace(".csv", "") + "_ovs.csv"
            with open(f"{self.dir}/{self.sentiment}_results/local_v_foreign/{filename}", "w",
                      encoding="utf8") as writer:
                writer.writelines([f"{w[0]},{w[1]}\n" for w in wdiff])



if __name__ == "__main__":
    #tc = TfidfCluster("tfidf_clustering","pos","sentiment_analysis","NB")

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="directory where you store data/results of word extraction")
    parser.add_argument("senti", help="sentiment specification of reviews to extract (positive/negative reviews)")
    parser.add_argument("moddir", help="directory to store training data/model of sentiment analysis")
    parser.add_argument("model", help="sentiment analysis model you want to use")
    args = parser.parse_args()

    if args.senti not in ["pos","neg"]:
        logging.info("Sentiment is not valid. Please specify either pos or neg")

    tc = WordExtractor(args.dir, args.senti, args.moddir, args.model)
    tc.run(1)

