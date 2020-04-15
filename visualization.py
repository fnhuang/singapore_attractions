import pandas as pd
from scipy.stats import pearsonr
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from tfidf_clustering import TfidfCluster
import sys, os
import matplotlib.pyplot as plt
import matplotlib
import csv
from mpl_toolkits.axes_grid1 import make_axes_locatable


pd.set_option('display.max_columns', 8)
pd.set_option('display.max_rows', 300)

class Visualize():

    def __init__(self, dir, file_name):
        self.dir = dir
        self.file_name = file_name

    def _rgb2hex(self, color_list):
        return "#{:02x}{:02x}{:02x}".format(color_list[0], color_list[1], color_list[2])

    def _get_catego(self, li, v):
        if v >= np.min(li) and v < np.percentile(li,25):
            return "very low"
        elif v >= np.percentile(li,25) and v < np.percentile(li,40):
            return "low"
        elif v >= np.percentile(li,40) and v < np.percentile(li,60):
            return "med"
        elif v >= np.percentile(li,60) and v < np.percentile(li,75):
            return "high"
        else:
            return "very high"

    def get_local_visitor_stats(self):
        info = pd.read_csv(f"{self.dir}/local_visitor_stat.csv")

        l_reviews = list(info["l_reviews"])
        v_reviews = list(info["v_reviews"])
        a_reviews = list(info["a_reviews"])

        l_ratings = list(info["l_avg_ratings"])
        v_ratings = list(info["v_avg_ratings"])
        a_ratings = list(info["a_avg_ratings"])

        reviews_data = [l_reviews, v_reviews, a_reviews]
        ratings_data = [l_ratings, v_ratings, a_ratings]

        # Create a figure instance
        fig = plt.figure(1, figsize=(9, 6))

        # Create an axes instance
        ax = fig.add_subplot(111)

        ax.set_title("Locals vs Visitors Reviews")

        # Create the boxplot
        bp = ax.boxplot(reviews_data, showfliers=False)
        plt.xticks([1, 2, 3], ['locals', 'visitors', "all"])
        plt.show()


        info["lrev_catego"] = [self._get_catego(l_reviews, v) for v in l_reviews]

        print(info.head(5))

    def get_basic_file_info(self):
        info = pd.read_csv(f"{self.dir}/{self.file_name}")

        info["name"] = info["name"].str.lower()
        info["tag"] = info["tag"].str.lower()

        info["csv"] = info["url"].map(lambda url: url[url.rindex("/"):].split("-"))
        info["csv"] = info["csv"].map(lambda url: f"{url[len(url)-2].lower()}.csv")

        info = info.drop(["url","starting page","name","number"], axis=1)

        geoinfo = pd.read_csv(f"{self.dir}/geocoordinates.csv")
        geoinfo["csv"] = geoinfo["url"].map(lambda url: url[url.rindex("/"):].split("-"))
        geoinfo["csv"] = geoinfo["csv"].map(lambda url: f"{url[len(url)-2].lower()}.csv")

        info = pd.merge(info, geoinfo, on="csv", how="inner")
        info = info.drop(["rating_1", "rating_2", "rating_3", "rating_4", "rating_5", "url"], axis=1)
        #print(info.head(5))
        #print(info.shape)

        return info

    def get_tag_info(self):
        basic_info = self.get_basic_file_info()
        basic_info = basic_info.drop(["done","csv","latitude","longitude"],axis=1)

        grouped = basic_info.groupby(["tag"])

        tag_info = grouped.mean()
        tag_info["count"] = grouped[["tag"]].count()

        #get correlation info
        review_vs_rating = pearsonr(np.array(tag_info["reviews"]).astype(np.float),
                          np.array(tag_info["rating"]).astype(np.float))
        review_vs_count = pearsonr(np.array(tag_info["reviews"]).astype(np.float),
                                    np.array(tag_info["count"]).astype(np.float))
        rating_vs_count = pearsonr(np.array(tag_info["rating"]).astype(np.float),
                                   np.array(tag_info["count"]).astype(np.float))

        #print(review_vs_rating, review_vs_count, rating_vs_count)

        #print(tag_info.sort_values(by=['rating'], ascending=False))

        return tag_info

    def _get_locational_cluster(self, latitude, longitude):

        if longitude >= 103.7575 and longitude <= 103.925 and latitude >= 1.2 and latitude <= 1.315:
            return "south"
        elif longitude >= 103.925 and longitude <= 104.0 and latitude >= 1.315 and latitude <= 1.4:
            return "east"
        elif longitude >= 103.7575 and longitude <= 103.925 and latitude >= 1.315 and latitude <= 1.4:
            return "centre"
        elif longitude >= 103.6725 and longitude <= 103.7575 and latitude >= 1.315 and latitude <= 1.4:
            return "west"
        elif longitude >= 103.6725 and longitude <= 104 and latitude >= 1.4 and latitude <= 1.5:
            return "north"
        else:
            return "unclustered"

    def _get_specific_info(self, sub_info):
        tag_info = self.get_tag_info()

        # includes get tag, % and location
        grouped = sub_info.groupby(["tag"])
        group_info = grouped.mean()
        group_info = group_info.drop(["reviews", "rating", "done", "latitude", "longitude"], axis=1)
        group_info["count"] = grouped[["tag"]].count()
        group_info = pd.merge(group_info, tag_info, on="tag", how="inner")
        group_info = group_info.drop(["reviews", "rating"], axis=1)
        group_info = group_info.rename(columns={"count_x": "count", "count_y": "total"})
        group_info["concentration"] = group_info["count"]/group_info["total"] * 100

        #location info
        locations = set(sub_info["location"])
        for loc in locations:
            group_info[loc] = [len(sub_info[(sub_info["tag"] == i) &
                                            (sub_info["location"] == loc)])
                                    for i, row in group_info.iterrows()]

        for loc in locations:
            group_info[f"%{loc}"] = [len(sub_info[(sub_info["tag"] == i) &
                                            (sub_info["location"] == loc)]) * 1.0 /
                                     float(group_info[(group_info.index == i)]["count"]) * 100
                                    for i, row in group_info.iterrows()]

        #print(group_info)
        return group_info



    def cluster_location(self):
        basic_info = self.get_basic_file_info()

        #coords = basic_info[["latitude","longitude"]].values
        #db = DBSCAN(eps=4/6371, min_samples=1, algorithm='auto', metric='haversine').fit(np.radians(coords))

        basic_info["location"] = basic_info.apply(lambda x: self._get_locational_cluster(x.latitude, x.longitude), axis=1)

        clusters = list(basic_info["location"])

        #print(basic_info)

        #get info for each cluster
        for cluster in set(clusters):
            sub_info = basic_info[basic_info["location"] == cluster]

            spec = self._get_specific_info(sub_info)
            print(cluster, ", total attractions:", sum(spec["count"]))
            print(spec.sort_values(by=["concentration"], ascending=False))


        #draw map
        BBox = ((103.5599, 104.1353, 1.1858, 1.5036))

        sgmap = plt.imread(f"{self.dir}/singapore.jpg")

        #colors = [self._rgb2hex(list(np.random.choice(range(256), size=3))) for i in range(0,len(clusters))]
        colors = ["red","green","blue","yellow","cyan","magenta","white"]

        fig, ax = plt.subplots(figsize=(8, 7))
        i = 0
        for location, df in basic_info.groupby("location"):

            #plt.scatter(dff['x'], dff['y'], s=size, c=cmap(norm(dff['colors'])),
            #            edgecolors='none', label="Feature {:g}".format(i))
            ax.scatter(df.longitude, df.latitude, zorder=1, alpha=1,
                       c=colors[i], s=20, edgecolor='black',
                       label=location)
            i += 1

        ax.set_title('Plotting Spatial Data on Singapore Map')
        ax.set_xlim(BBox[0], BBox[1])
        ax.set_ylim(BBox[2], BBox[3])
        im = ax.imshow(sgmap, zorder=0, alpha=0.5,  extent=BBox, aspect='equal')

        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes('right', size='5%', pad=0.05)

        #cb = fig.colorbar(im, cax=cax)
        #loc = np.arange(0, max(clusters), max(clusters) / float(len(colors)))
        #cb.set_ticks(loc)
        #cb.set_ticklabels(colors)
        plt.legend()
        plt.show()

    def cluster_tag_info(self, no_clust):
        tag_info = self.get_tag_info(self.file_name)

        print("quantity boxplot")
        print(np.min(tag_info["count"]), np.percentile(tag_info["count"], 25), np.percentile(tag_info["count"], 40),
              np.percentile(tag_info["count"], 50), np.percentile(tag_info["count"], 60), np.percentile(tag_info["count"], 75),
              np.max(tag_info["count"]))
        print("reviews boxplot")
        print(np.min(tag_info["reviews"]), np.percentile(tag_info["reviews"], 25), np.percentile(tag_info["reviews"], 40),
              np.percentile(tag_info["reviews"], 50), np.percentile(tag_info["reviews"], 60), np.percentile(tag_info["reviews"], 75),
              np.max(tag_info["reviews"]))
        print("rating boxplot")
        print(np.min(tag_info["rating"]), np.percentile(tag_info["rating"], 25), np.percentile(tag_info["rating"], 40),
              np.percentile(tag_info["rating"], 50), np.percentile(tag_info["rating"], 60), np.percentile(tag_info["rating"], 75),
              np.max(tag_info["rating"]))

        #data preparation
        data = []
        id2tag = {}
        id = 0
        for tag, row in tag_info.iterrows():
            id2tag[id] = tag
            data.append([row["count"], row["reviews"], row["rating"]])
            id += 1

        data = np.array(data)
        # no_clust -1 = find k
        if no_clust == -1:
            for i in range(2, len(data) + 1):
                kmeans = KMeans(n_clusters=i).fit(data)
                print(f"{i},{self.tfc.calculate_silhouette(kmeans, data)}")
        else:
            # run cluster
            kmeans = KMeans(n_clusters=no_clust).fit(data)
            print(f"{no_clust},{self.tfc.calculate_silhouette(kmeans, data)}")
            print("Clusters found")
            for n in range(0, no_clust):
                print("Cluster", n)
                indices = [i for i, x in enumerate(kmeans.labels_) if x == n]
                attr_in_the_cluster = [id2tag[i] for i in indices]
                print(attr_in_the_cluster)
                print("number,reviews,rating_mean,rating_std")
                to_be_printed = ""
                for i in range(0, len(kmeans.cluster_centers_[n])):
                    to_be_printed += f"{kmeans.cluster_centers_[n][i]},"
                print(to_be_printed[0:len(to_be_printed) - 1])



viz = Visualize("visualize","top300.csv")
viz.get_local_visitor_stats()