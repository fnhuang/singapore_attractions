import preprocess
from preprocess import Document
import os, pickle, sys
from nltk import FreqDist
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
import numpy as np
from sentiment_analysis import SentimentAnalyzer



class TfidfCluster():
    #print topic clusters into document
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


    def set_datawords(self, documents):

        #only allow adjectives
        for doc in documents:
            self.data_words.append([w for w in word_tokenize(doc.content) if w != "."])

    # advanced processing
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
            print("\r", end="")
            print("Getting custom words", int(file_count * 100 / len(files)), "%", end="", flush=True)
        print("")

        print("Combining documents of the same location...")
        ndocs = self._combine_documents_of_the_same_location(subdocs)

        print("Removing most common words")
        ndocs = self.remove_most_common_words(ndocs, filter_xtrim)


        return ndocs

    def calculate_silhouette(self):
        s_is = []
        for i in range(0, len(self.kmeans_model.labels_)):
            cluster = self.kmeans_model.labels_[i]
            a_indices = [j for j,k in enumerate(self.kmeans_model.labels_) if k == cluster and j != i]
            i_point = csr_matrix.toarray(self.tfidf_matrix[i,:])

            clust_dist = 0
            for doc_id in a_indices:
                doc_point = csr_matrix.toarray(self.tfidf_matrix[doc_id,:])
                clust_dist += np.sum(np.square(i_point - doc_point))
            a_i = 0
            if len(a_indices) > 0:
                a_i = 1/len(a_indices) * clust_dist

            clusters = set([c for c in self.kmeans_model.labels_ if c != cluster])
            dists = []
            for cluster in clusters:
                b_indices = [j for j,k in enumerate(self.kmeans_model.labels_) if k == cluster]

                clust_dist = 0
                for doc_id in b_indices:
                    doc_point = csr_matrix.toarray(self.tfidf_matrix[doc_id, :])
                    clust_dist += np.sum(np.square(i_point - doc_point))
                b = 0
                if len(b_indices) > 0:
                    b = 1/len(b_indices) * clust_dist
                dists.append(b)
            b_i = min(dists)

            s_i = 0
            if a_i < b_i:
                s_i = 1 - (a_i/b_i)
            elif a_i > b_i:
                s_i = (b_i/a_i) - 1

            s_is.append(s_i)

        return np.average(s_is)


    def calculate_ch_index(self):
        bk = self.calculate_between_cluster_variation()
        wk = self.kmeans_model.inertia_

        k = max(self.kmeans_model.labels_)+1
        n = len(self.kmeans_model.labels_)

        ch = (bk/(k-1))/(wk/(n-k))

        return ch

    def calculate_between_cluster_variation(self):

        average_point = np.array([i for i in self.kmeans_model.cluster_centers_[0]])
        for i in range(1, len(self.kmeans_model.cluster_centers_)):
            average_point += self.kmeans_model.cluster_centers_[i]
        average_point = average_point/len(self.kmeans_model.cluster_centers_)

        distance = []
        for i in range(0, len(self.kmeans_model.cluster_centers_)):
            count = sum(1 for j in self.kmeans_model.labels_ if j == i)
            d = np.sum(np.square(self.kmeans_model.cluster_centers_[i] - average_point))
            d = d * count
            distance.append(d)
        total_distance = sum(distance)

        return total_distance


    def assess_k(self, filter_xtrim):
        if len(self.id2doc) == 0:
            os.makedirs(os.path.dirname(f"{self.dir}/{self.sentiment}_model/"), exist_ok=True)
            documents = self.prepare_documents(filter_xtrim)
            self.transform_into_featuresets(documents)
        #print([(doc.name,doc.location) for doc in self.id2doc.values()])

        with open(f"{self.dir}/{self.sentiment}_model/assess_k.txt","w",encoding="utf8") as writer:
            for i in range(2, len(self.id2doc) + 1):
                self.kmeans_model = KMeans(n_clusters=i).fit(self.tfidf_matrix)
                writer.write(f"{i},{self.calculate_silhouette()}\n")


    def _retrieve_most_common_words(self, docs):
        all_words = []
        for doc in docs:
            all_words.extend(set([w for w in word_tokenize(doc.content.replace(".",""))]))
        docfdist = FreqDist(all_words)

        all_words = set(all_words)
        with open(f"{self.dir}/{self.sentiment}_model/most_common_words.txt","w",encoding="utf8") as writer:
            writer.writelines([f"{p}\n" for p in docfdist .most_common(len(all_words))])

        return docfdist

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

            allowed_type = [(preprocess.ADJ,preprocess.NOUN),(preprocess.NOUN, preprocess.NOUN)]

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
                if pos_tags[id][1] in preprocess.ADV or pos_tags[id][1] in preprocess.NOUN or pos_tags[id][1] in preprocess.ADJ:
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

    def run(self, n_cluster, filter_xtrim):
        if len(self.id2doc) == 0:
            os.makedirs(os.path.dirname(f"{self.dir}/{self.sentiment}_model/"), exist_ok=True)
            documents = self.prepare_documents(filter_xtrim)
            self.transform_into_featuresets(documents)

        kmeans = KMeans(n_clusters=n_cluster).fit(self.tfidf_matrix)

        print("total words:", len(self.tfidf_vectorizer.get_feature_names()))

        os.makedirs(os.path.dirname(f"{self.dir}/{self.sentiment}_results/"), exist_ok=True)
        print("Topics clusters found")
        for n in range(0, n_cluster):
            print("Topic",n)
            indices = [i for i, x in enumerate(kmeans.labels_) if x == n]
            docs_in_the_cluster = [(self.id2doc[i].name, self.id2doc[i].location, self.id2doc[i].sentiment) for i in indices]
            print(docs_in_the_cluster)

            sorted_indices = np.argsort(kmeans.cluster_centers_[n])[::-1]
            sorted_features = np.array(self.tfidf_vectorizer.get_feature_names())[sorted_indices]

            center_copy = [i for i in kmeans.cluster_centers_[n]]
            center_copy.sort()
            sorted_prob = center_copy[::-1]

            word_prob = list(zip(sorted_features, sorted_prob))
            rep_words = []
            for w in [w for w in word_prob if w[1] > 0]:
                sent = " ".join(w[0].split("_"))
                prob_dist = self.sentiment_classifier.sentiment(sent)

                if prob_dist[0] == self.sentiment \
                    and prob_dist[1] > 0.8:
                    rep_words.append(w)

            with open(f"{self.dir}/{self.sentiment}_results/topic_{n}.txt","w",encoding="utf8") as writer:
                writer.write(f"{docs_in_the_cluster}\n")
                for w in rep_words:
                    writer.write(f"{w[0]},{w[1]}\n")

        self.kmeans_model = kmeans
        print("silhouette:", self.calculate_silhouette())
        pickle.dump(self.kmeans_model, open(f"{self.dir}/{self.sentiment}_model/kmeans_model.pickle","wb"))

if __name__ == "__main__":
    tc = TfidfCluster("tfidf_clustering","pos","sentiment_analysis","NB")
    #tc.assess_k(0.8)
    tc.run(10, 0.8)

