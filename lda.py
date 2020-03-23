import os, sys
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from collections import namedtuple
from nltk import FreqDist, word_tokenize
from gensim.models import CoherenceModel
import gensim.corpora as corpora
from gensim.models import LdaModel as GensimLda
import nltk


Document = namedtuple("Document", "name content location sentiment")

class MyLDA():
    #do stopwords removal #gensim
    #add bigram trigram #gensim (later)
    #use gensim (later)
    def __init__(self, dir):
        self.dir = dir
        self.data_words = []
        self.id2word = {}
        self.corpus = []
        self.allowed_word_types = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS"]
        self.count_vectorizer = CountVectorizer(stop_words='english')


    def print_topics(self, model, n_top_words):
        words = self.count_vectorizer.get_feature_names()
        for topic_idx, topic in enumerate(model.components_):
            # topic is an array with probability for each word in words
            # topic.argsort gives index of word from low to high probability
            print("\nTopic #%d:" % topic_idx)
            print(" ".join([words[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))

    def assess_topic(self, feval):
        documents = self.prepare_documents()


        print("number topics,topic coherence")
        for number_topics in range(2, 50, 1):
            self.data_words = []
            self.id2word = {}
            self.corpus = []

            self.prepare_featuresets(documents, feval)

            lda_model = GensimLda(corpus=self.corpus, num_topics=number_topics)

            coherence_model_lda = CoherenceModel(model=lda_model, corpus=self.corpus, coherence='c_v',
                                                 texts=self.data_words, dictionary=self.id2word, processes=1)

            print(f"{number_topics},{coherence_model_lda.get_coherence()}")

    def assess_filter_extreme_val(self, number_topics):
        documents = self.prepare_documents()

        print("filter extreme value,topic coherence")
        for feval in range(3, 10, 1):
            self.prepare_featuresets(documents, feval * 1.0 / 10)

            lda_model = GensimLda(corpus=self.corpus, num_topics=number_topics)

            coherence_model_lda = CoherenceModel(model=lda_model, corpus=self.corpus, coherence='c_v',
                                                 texts=self.data_words, dictionary=self.id2word, processes=1)

            print(f"{feval * 1.0 / 10},{coherence_model_lda.get_coherence()}")


    def run(self, number_topics, feval):
        documents = self.prepare_documents()


        self.prepare_featuresets(documents, feval)

        #lda = LDA(n_components=number_topics)
        print("fitting model...")
        lda_model = GensimLda(corpus=self.corpus, num_topics=number_topics, eta=0.9)

        # Print the topics found by the LDA model
        print("Topics found via LDA:")


        for topic in lda_model.show_topics(num_topics=number_topics, formatted=False, num_words=10):
            print(topic[0],[(self.id2word[int(d[0])],d[1]) for d in topic[1]])

        # print get coherence model
        coherence_model_lda = CoherenceModel(model=lda_model, corpus=self.corpus, coherence='c_v', texts=self.data_words,
                                             dictionary=self.id2word, processes=1)
        print("coherence score",coherence_model_lda.get_coherence())

        pickle.dump(lda_model, open(f"{self.dir}/model/lda.pickle", "wb"))




    def set_datawords(self, documents):

        #only allow adjectives
        for doc in documents:
            self.data_words.append(word_tokenize(doc.content))
            #self.data_words.append([p[0] for p in nltk.pos_tag(word_tokenize(doc.content)) if p[1] in self.allowed_word_types])


    def retrieve_most_common_words(self):
        all_words = []
        for dword in self.data_words:
            all_words.extend(set(dword))
        fdist = FreqDist(all_words)

        with open(f"{self.dir}/most_common_words.txt","w",encoding="utf8") as writer:
            writer.writelines([f"{p}\n" for p in fdist.most_common(len(set(all_words)))])


    def prepare_featuresets(self, documents, filter_extreme):
        #print("preparing featuresets...")

        #data format for gensim libraries.
        self.set_datawords(documents)

        self.id2word = corpora.Dictionary(self.data_words)

        #filter out words that appear in more than 0.5 of documents
        #self.id2word.filter_extremes(no_above=filter_extreme)
        self.retrieve_most_common_words()

        # term document frequency
        self.corpus = [self.id2word.doc2bow(dw) for dw in self.data_words]



        #featuresets = self.count_vectorizer.fit_transform([d.content for d in documents])
        #print(len(documents), len(self.count_vectorizer.get_feature_names()), featuresets.shape)

        #return featuresets



    def prepare_documents(self):
        #print("preparing documents....")
        documents = []

        '''for file in os.listdir(f"{self.dir}/pos/"):
            d = pickle.load(open(f"{self.dir}/pos/{file}", "rb"))
            doc_content = ""
            for k in d.keys():
                doc_content = doc_content + " ".join([s for s in d[k] for k in d.keys()])
            documents.append(Document(file, doc_content, "all", "pos"))'''

        for file in os.listdir(f"{self.dir}/neg/"):
            d = pickle.load(open(f"{self.dir}/neg/{file}", "rb"))
            doc_content = ""
            for k in d.keys():
                doc_content = doc_content + " ".join([s for s in d[k] for k in d.keys()])
            documents.append(Document(file, doc_content, "all", "neg"))
            '''for k in d.keys():
                doc_content = " ".join([s for s in d[k]])
                documents.append(Document(file, doc_content, k, "neg"))'''

        return documents

lda = MyLDA("lda_noun")
#lda.run(5, 0.8) #best for positive data
lda.run(10, 0.7) #for negative data
#lda.assess_filter_extreme_val(4)
#lda.assess_topic(0.4)
#print(word_tokenize("hello world"))
