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
from scipy import spatial, sparse
import pandas as pd



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
            print("\r", end="")
            print("Getting custom words", int(file_count * 100 / len(files)), "%", end="", flush=True)
        print("")

        print("Combining documents of the same location...")
        ndocs = self._combine_documents_of_the_same_location(subdocs)

        print("Combining documents of the same tags...")
        ndocs = self._combine_documents_of_the_same_group(ndocs)

        print("Removing most common words")
        ndocs = self.remove_most_common_words(ndocs, filter_xtrim)

        return ndocs



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

    def calculate_silhouette(self, kmeans_model, data_matrix):
        s_is = []
        for i in range(0, len(kmeans_model.labels_)):
            cluster = kmeans_model.labels_[i]
            a_indices = [j for j,k in enumerate(kmeans_model.labels_) if k == cluster and j != i]
            if isinstance(data_matrix, sparse.csc_matrix):
                i_point = csr_matrix.toarray(data_matrix[i,:])
            else:
                i_point = data_matrix[i,:]

            clust_dist = 0
            for doc_id in a_indices:
                if isinstance(data_matrix, sparse.csc_matrix):
                    doc_point = csr_matrix.toarray(data_matrix[doc_id,:])
                else:
                    doc_point = data_matrix[doc_id, :]
                clust_dist += np.sum(np.square(i_point - doc_point))
            a_i = 0
            if len(a_indices) > 0:
                a_i = 1/len(a_indices) * clust_dist

            clusters = set([c for c in kmeans_model.labels_ if c != cluster])
            dists = []
            for cluster in clusters:
                b_indices = [j for j,k in enumerate(kmeans_model.labels_) if k == cluster]

                clust_dist = 0
                for doc_id in b_indices:
                    if isinstance(data_matrix, sparse.csc_matrix):
                        doc_point = csr_matrix.toarray(data_matrix[doc_id, :])
                    else:
                        doc_point = data_matrix[doc_id, :]
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
                writer.write(f"{i},{self.calculate_silhouette(self.kmeans_model,self.tfidf_matrix)}\n")


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

    # find uqwords of tuple1
    def find_comwords(self, dict1, dict2):
        common_words = [w for w in dict1.keys() if w in dict2.keys()]
        wdiff = []

        for w in common_words:
            p = dict1[w]

            # w prob is l2 normalized, the sum of square of each word in a doc add up to 1
            # thus, we take the difference in the square value
            denom = p**2 + dict2[w]**2
            diff = (p ** 2 - dict2[w] ** 2)/denom
            wdiff.append((w, diff, p**2, dict2[w]**2))

        wdiff = sorted(wdiff, key=lambda x: x[1], reverse=True)

        return wdiff

    # find uqwords of tuple1
    def find_uqwords(self, dict1, dict2):
        wdiff = []

        words = [w for w in dict1.keys() if w not in dict2.keys()]

        for w in words:
            p = dict1[w]

            # w prob is l2 normalized, the sum of square of each word in a doc add up to 1
            # thus, we take the difference in the square value
            diff = p**2
            wdiff.append((w, diff))

        wdiff = sorted(wdiff, key=lambda x: x[1], reverse=True)

        return wdiff

    def get_basic_file_info(self, file_name):
        info = pd.read_csv(f"{self.dir}/{file_name}")

        info["name"] = info["name"].str.lower()
        info["tag"] = info["tag"].str.lower()

        info["csv"] = info["url"].map(lambda url: url[url.rindex("/"):].split("-"))
        info["csv"] = info["csv"].map(lambda url: f"{url[len(url)-2].lower()}.csv")

        info = info.drop(["url","starting page","name"], axis=1)

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


    def get_local_foreigner_difference(self, filter_xtrim, by_group):

        if len(self.id2doc) == 0:
            if by_group:
                os.makedirs(os.path.dirname(f"{self.dir}/{self.sentiment}_model/"), exist_ok=True)
                documents = self.prepare_documents_by_group(filter_xtrim)
                self.transform_into_featuresets(documents)
            else:
                os.makedirs(os.path.dirname(f"{self.dir}/{self.sentiment}_model/"), exist_ok=True)
                documents = self.prepare_documents(filter_xtrim)
                self.transform_into_featuresets(documents)

        os.makedirs(os.path.dirname(f"{self.dir}/{self.sentiment}_results/local_v_foreign/"), exist_ok=True)

        # create doc_pair item = {battlebox.csv:[(sgp,[(battle,0.2),(box,0.1)]),
        # (ovs,[(hot,0.2),(tour,0.1)])]}
        doc_pairs = {}
        doc_count = 0
        for doc_num in self.id2doc:
            doc_count += 1
            doc = self.id2doc[doc_num]

            # get word probability distribution - it is l2 normalized
            prob_dist = csr_matrix.toarray(self.tfidf_matrix[doc_num, :])[0]
            # print(np.sum(np.square(prob_dist)))

            # get term (features) and its probability in descending format
            sorted_indices = np.argsort(prob_dist)[::-1]
            sorted_features = np.array(self.tfidf_vectorizer.get_feature_names())[sorted_indices]

            temp = [i for i in prob_dist]
            temp.sort()
            sorted_prob = temp[::-1]

            word_prob = list(zip(sorted_features, sorted_prob))

            # keep words with probability more than 0 and sentiment prob is larger than 0.9
            rep_words = []
            for w in [w for w in word_prob if w[1] > 0]:
                sent = " ".join(w[0].split("_"))
                prob_dist = self.sentiment_classifier.sentiment(sent)

                if prob_dist[0] == self.sentiment and prob_dist[1] > 0.8:
                    rep_words.append(w)

            doc_pairs.setdefault(doc.name,[]).append((doc.location, rep_words))

            print("\r",end="")
            print("Getting relevant sentimental words", int(doc_count/len(self.id2doc) * 100), "percent", end="", flush=True)

        # local-foreign review difference
        doc_count = 0
        for doc_name in doc_pairs:
            doc_count += 1

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


            all_words = set([w[0] for w in local_pdist])
            all_words.update(set([w[0] for w in foreign_pdist]))

            # get cosine similarity
            '''prob_dist1 = []
            prob_dist2 = []
            for w in all_words:
                prob_dist1.append(local_dict[w] if w in local_dict.keys() else 0)
                prob_dist2.append(foreign_dict[w] if w in foreign_dict.keys() else 0)
            cos_sim = spatial.distance.cosine(prob_dist1, prob_dist2)'''


            print("\r",end="")
            print("Calculating cosine sim", int(doc_count/len(doc_pairs) * 100), "percent", end="", flush=True)


    def run(self, n_cluster, filter_xtrim):
        if len(self.id2doc) == 0:
            os.makedirs(os.path.dirname(f"{self.dir}/{self.sentiment}_model/"), exist_ok=True)
            documents = self.prepare_documents(filter_xtrim)
            self.transform_into_featuresets(documents)

        kmeans = KMeans(n_clusters=n_cluster).fit(self.tfidf_matrix)

        print("total words:", len(self.tfidf_vectorizer.get_feature_names()))

        os.makedirs(os.path.dirname(f"{self.dir}/{self.sentiment}_results/clustering/"), exist_ok=True)
        print("Topics clusters found")
        for n in range(0, n_cluster):
            print("Topic",n)
            indices = [i for i, x in enumerate(kmeans.labels_) if x == n]
            docs_in_the_cluster = [(self.id2doc[i].name, self.id2doc[i].location, self.id2doc[i].sentiment) for i in indices]

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

            with open(f"{self.dir}/{self.sentiment}_results/clustering/topic_{n}.txt","w",encoding="utf8") as writer:
                writer.write(f"{docs_in_the_cluster}\n")
                for w in rep_words:
                    writer.write(f"{w[0]},{w[1]}\n")

        self.kmeans_model = kmeans
        print("silhouette:", self.calculate_silhouette(self.kmeans_model, self.tfidf_matrix))
        pickle.dump(self.kmeans_model, open(f"{self.dir}/{self.sentiment}_model/kmeans_model.pickle","wb"))

if __name__ == "__main__":
    #dir = sys.argv[1]
    #sentiment = sys.argv[2]
    #model_dir = sys.argv[3]
    #model = sys.argv[4]
    #tc = TfidfCluster(dir, sentiment, model_dir, model)
    tc = TfidfCluster("yelp_tfidf","neg","yelp_senti_analysis","NB")
    #tc.assess_k(0.8)
    #tc.run(14, 0.8)
    tc.get_local_foreigner_difference(1,True)

