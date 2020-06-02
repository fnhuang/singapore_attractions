import pandas as pd
from scipy.stats import pearsonr
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from tfidf_clustering import TfidfCluster
import sys, os, csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 300)

class Visualize():

    def __init__(self, dir, file_name, data_source, tag_type):
        self.dir = dir
        self.file_name = file_name
        self.tfc = TfidfCluster("","","","")
        self.data_source = data_source
        self.tag_type = tag_type
        self.sg_poly = self._get_singapore_region_poly()

    def _get_singapore_region_poly(self):
        sg_poly = {}
        with open(f"{self.dir}/singapore_region_poly.csv","r") as reader:
            lines = reader.readlines()

            for line in lines[1:]:
                data = line.split(",")
                region = data[0]
                polygon = data[2].split(";")

                coords = []
                geocs = [[float(geoc.split("-")[0]),float(geoc.split("-")[1])] for poly in polygon for geoc in poly.split(",")]
                coords.extend(geocs)

                if region not in sg_poly.keys():
                    sg_poly[region] = []
                sg_poly[region].append(coords)


        return sg_poly

    def _rgb2hex(self, color_list):
        return "#{:02x}{:02x}{:02x}".format(color_list[0], color_list[1], color_list[2])

    def _get_catego(self, li, v):
        if v < np.percentile(li,25):
            return "low"
        elif v >= np.percentile(li,25) and v < np.percentile(li,40):
            return "low"
        elif v >= np.percentile(li,40) and v < np.percentile(li,60):
            return "med"
        elif v >= np.percentile(li,60) and v < np.percentile(li,75):
            return "high"
        else:
            return "high"

    def get_local_visitor_category(self, tag, l_column, v_column):
        # get_info
        info = pd.read_csv(f"{self.dir}/local_visitor_stat.csv")
        basic_info = self.get_basic_file_info()
        info = pd.merge(info, basic_info, on="csv", how="inner")
        info["location"] = info.apply(lambda x: self._get_locational_cluster(x.latitude, x.longitude),
                                      axis=1)

        l_reviews = list(info["l_reviews"])
        v_reviews = list(info["v_reviews"])
        a_reviews = list(info["a_reviews"])

        l_ratings = list(info["l_avg_ratings"])
        v_ratings = list(info["v_avg_ratings"])
        a_ratings = list(info["a_avg_ratings"])

        info["lrev_catego"] = [self._get_catego(a_reviews, v) for v in l_reviews]
        info["vrev_catego"] = [self._get_catego(a_reviews, v) for v in v_reviews]
        info["lrat_catego"] = [self._get_catego(a_ratings, v) for v in l_ratings]
        info["vrat_catego"] = [self._get_catego(a_ratings, v) for v in v_ratings]

        #print(info["tag"])
        subinfo = info[info[self.tag_type] == tag]
        print(subinfo[(subinfo["lrat_catego"] == "very high") & (subinfo["vrat_catego"] == "high")])

        group_by_review = subinfo.groupby([l_column, v_column]).size().reset_index().rename(columns={0: 'count'})
        total = np.sum(group_by_review["count"])
        group_by_review["percentage"] = [v/total for v in group_by_review["count"]]

        print(group_by_review.sort_values(by=['percentage'], ascending=False))

    def _make_grouped_bar_plot(self, df, l_column, v_column, obj_name):
        #print(df[(df["lrev_catego"] == "very high") & (df["vrev_catego"] == "med")])

        group_by_review = df.groupby([l_column, v_column]).size().reset_index().rename(columns={0: 'count'})
        #print(group_by_review)

        categories = ["very low", "low", "med", "high", "very high"]
        datas = []
        for v_catego in categories:
            sub_data = []
            for l_catego in categories:
                subg = group_by_review[
                    (group_by_review[l_column] == l_catego) & (group_by_review[v_column] == v_catego)]

                count = int(subg["count"]) if len(subg) > 0 else 0

                sub_data.append(count)
            datas.append(sub_data)

        fig, ax = plt.subplots()

        color_list = ['b', 'g', 'r', "yellow", "magenta"]
        gap = 0.8 / len(datas)

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        for i, row in enumerate(datas):
            #print(i, row)
            X = np.arange(len(row))
            rects = ax.bar(X + i * gap, row,
                           width=gap,
                           color=color_list[i % len(color_list)], edgecolor="black", label=f"tourist:{categories[i]}")
            autolabel(rects)

        x = np.arange(len(categories), step=1.15)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()

        ax.set_title(f"Number of {obj_name.capitalize()}: Locals vs Tourists")

        fig.tight_layout()

        plt.xlabel(f"local {obj_name}")
        plt.ylabel(f"# poi")

        plt.show()

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
        ax = fig.add_subplot(111)
        ax.set_title("Locals vs Tourists Reviews")


        # Create boxplot of reviews and ratings
        bp = ax.boxplot(reviews_data, showfliers=False)
        plt.xticks([1, 2, 3], ['locals', 'tourists', "all"])
        plt.show()

        plt.close()

        fig = plt.figure(1, figsize=(9, 6))
        ax = fig.add_subplot(111)
        ax.set_title("Locals vs Tourists Ratings")

        bp = ax.boxplot(ratings_data, showfliers=False)
        plt.xticks([1, 2, 3], ['locals', 'tourists', "all"])
        plt.show()

        plt.close()

        info["lrev_catego"] = [self._get_catego(a_reviews, v) for v in l_reviews]
        info["vrev_catego"] = [self._get_catego(a_reviews, v) for v in v_reviews]
        info["lrat_catego"] = [self._get_catego(a_ratings, v) for v in l_ratings]
        info["vrat_catego"] = [self._get_catego(a_ratings, v) for v in v_ratings]

        # pearson correl info
        review_stats_correl = pearsonr(np.array(l_reviews).astype(np.float),
                                       np.array(v_reviews).astype(np.float))
        print(review_stats_correl)

        rating_stats_correl = pearsonr(np.array(l_ratings).astype(np.float),
                                       np.array(v_ratings).astype(np.float))
        print(rating_stats_correl)

        # create grouped bar plot of locals vs visitors statistics
        self._make_grouped_bar_plot(info, "lrev_catego", "vrev_catego", "review")
        plt.close()

        self._make_grouped_bar_plot(info, "lrat_catego", "vrat_catego", "rating")
        plt.close()
        #print(info)

        return info



    def get_local_visitor_specs(self, l_catego, v_catego, l_column, v_column):
        # get_info
        info = pd.read_csv(f"{self.dir}/local_visitor_stat.csv")
        basic_info = self.get_basic_file_info()
        info = pd.merge(info, basic_info, on="csv", how="inner")
        info["location"] = info.apply(lambda x: self._get_locational_cluster(x.latitude, x.longitude),
                                            axis=1)

        l_reviews = list(info["l_reviews"])
        v_reviews = list(info["v_reviews"])
        a_reviews = list(info["a_reviews"])

        l_ratings = list(info["l_avg_ratings"])
        v_ratings = list(info["v_avg_ratings"])
        a_ratings = list(info["a_avg_ratings"])


        info["lrev_catego"] = [self._get_catego(a_reviews, v) for v in l_reviews]
        info["vrev_catego"] = [self._get_catego(a_reviews, v) for v in v_reviews]
        info["lrat_catego"] = [self._get_catego(a_ratings, v) for v in l_ratings]
        info["vrat_catego"] = [self._get_catego(a_ratings, v) for v in v_ratings]


        subinfo = info[(info[l_column] == l_catego) & (info[v_column] == v_catego)]
        #print(subinfo)
        specs = self._get_specific_info(subinfo)
        specs = specs.drop(["l_reviews", "v_reviews", "a_reviews",
                            "l_avg_ratings", "v_avg_ratings", "a_avg_ratings",
                            "count", "total"], axis=1)
        #specs = specs[["concentration", "%south", "%centre", "%north", "%west", "%east"]]
        print(specs.sort_values(by=['concentration'], ascending=False))

    def get_basic_file_info(self):
        info = pd.read_csv(f"{self.dir}/{self.file_name}")

        info["name"] = info["name"].str.lower()
        info[self.tag_type] = info[self.tag_type].str.lower()

        if self.data_source == "tripadvisor":
            info["csv"] = info["url"].map(lambda url: url[url.rindex("/"):].split("-"))
            info["csv"] = info["csv"].map(lambda url: f"{url[len(url)-2].lower()}.csv")
            geoinfo = pd.read_csv(f"{self.dir}/geocoordinates.csv")
            geoinfo["csv"] = geoinfo["url"].map(lambda url: url[url.rindex("/"):].split("-"))
            geoinfo["csv"] = geoinfo["csv"].map(lambda url: f"{url[len(url)-2].lower()}.csv")
            info = pd.merge(info, geoinfo, on="csv", how="inner")
            info = info.drop(["rating_1", "rating_2", "rating_3", "rating_4", "rating_5"], axis=1)
        elif self.data_source == "yelp":
            info["csv"] = info["url"].map(lambda url: url[url.rindex("/")+1:])
            info["csv"] = info["csv"].map(lambda url: f"{url.lower()}.csv")

        to_be_dropped = ["url", "starting page", "name", "number",
                         "crawl_seq", "yelp_start_index"]
        for col in to_be_dropped:
            if col in info.columns:
                info = info.drop([col], axis=1)

        # duplicate the row if there are double tags
        for index, row in info.iterrows():
            tags = row[self.tag_type].split(",")
            info.ix[index, self.tag_type] = tags[0]
            tags.pop(0)
            for t in tags:
                row[self.tag_type] = t
                info = info.append(row, ignore_index=True)

        #print(info[info["latitude"].isnull()][["csv"]])
        #print(info.shape)
        #print(info.tail(5))
        return info

    def get_rating_of_rarely_visited_places(self):
        basic_info = self.get_basic_file_info()
        basic_info = basic_info.drop(["done", "csv", "latitude", "longitude"], axis=1)

        basic_info = basic_info[basic_info["rating"] > 0]
        grouped = basic_info.groupby([self.tag_type])

        tag_info = grouped.mean()
        tag_info["count"] = grouped[[self.tag_type]].count()

        # get correlation info
        review_vs_rating = pearsonr(np.array(tag_info["reviews"]).astype(np.float),
                                    np.array(tag_info["rating"]).astype(np.float))
        review_vs_count = pearsonr(np.array(tag_info["reviews"]).astype(np.float),
                                   np.array(tag_info["count"]).astype(np.float))
        rating_vs_count = pearsonr(np.array(tag_info["rating"]).astype(np.float),
                                   np.array(tag_info["count"]).astype(np.float))

        reviews = list(tag_info["reviews"])
        tag_info["rev_catego"] = [self._get_catego(reviews, v) for v in reviews]

        ratings = list(tag_info["rating"])
        tag_info["rat_catego"] = [self._get_catego(ratings, v) for v in ratings]

        counts = list(tag_info["count"])
        tag_info["count_catego"] = [self._get_catego(counts, v) for v in counts]


        # print(review_vs_rating, review_vs_count, rating_vs_count)
        # print(tag_info.sort_values(by=['rev_catego', 'rat_catego', 'count_catego'], ascending=False)[['rev_catego', 'rat_catego', 'count_catego']])
        print(tag_info.sort_values(by=['rating'], ascending=False)["rating"])
        print(sum(tag_info["rating"]))

        return tag_info

    def get_tag_info(self, reviewer):
        if reviewer == "visitor":
            basic_info = self.get_basic_file_info()
            lv_info = self.get_local_visitor_stats()[["csv", "v_reviews", "v_avg_ratings"]]
            basic_info = pd.merge(basic_info, lv_info, on="csv", how="inner")
            basic_info = basic_info.drop(["reviews", "rating"], axis=1)
            basic_info = basic_info.rename(columns={"v_reviews": "reviews", "v_avg_ratings": "rating"})
        elif reviewer == "local":
            basic_info = self.get_basic_file_info()
            lv_info = self.get_local_visitor_stats()[["csv","l_reviews","l_avg_ratings"]]
            basic_info = pd.merge(basic_info, lv_info, on="csv", how="inner")
            basic_info = basic_info.drop(["reviews", "rating"], axis=1)
            basic_info = basic_info.rename(columns={"l_reviews": "reviews", "l_avg_ratings": "rating"})
        else:
            basic_info = self.get_basic_file_info()
            basic_info = basic_info.drop(["done", "csv", "latitude", "longitude"], axis=1)


        grouped = basic_info.groupby([self.tag_type])

        tag_info = grouped.mean()
        tag_info["count"] = grouped[[self.tag_type]].count()

        #sd_li = list(grouped[["reviews"]].std()["reviews"])
        #mu_li = list(tag_info["reviews"])
        #tag_info["reviews_sd"] = [i / j for i, j in zip(sd_li, mu_li)]
        #tag_info["rating_sd"] = grouped[["rating"]].std()

        #get correlation info
        review_vs_rating = pearsonr(np.array(tag_info["reviews"]).astype(np.float),
                          np.array(tag_info["rating"]).astype(np.float))
        review_vs_count = pearsonr(np.array(tag_info["reviews"]).astype(np.float),
                                    np.array(tag_info["count"]).astype(np.float))
        rating_vs_count = pearsonr(np.array(tag_info["rating"]).astype(np.float),
                                   np.array(tag_info["count"]).astype(np.float))


        reviews = list(tag_info["reviews"])
        tag_info["rev_catego"] = [self._get_catego(reviews, v) for v in reviews]

        ratings = list(tag_info["rating"])
        tag_info["rat_catego"] = [self._get_catego(ratings, v) for v in ratings]

        counts = list(tag_info["count"])
        tag_info["count_catego"] = [self._get_catego(counts, v) for v in counts]

        '''print(np.min(tag_info["count"]), np.percentile(tag_info["count"], 25), np.percentile(tag_info["count"], 40),
              np.percentile(tag_info["count"], 60),
              np.percentile(tag_info["count"], 75),
              np.max(tag_info["count"]))

        print(np.min(tag_info["reviews"]), np.percentile(tag_info["reviews"], 25), np.percentile(tag_info["reviews"], 40),
              np.percentile(tag_info["reviews"], 60),
              np.percentile(tag_info["reviews"], 75),
              np.max(tag_info["reviews"]))

        print(np.min(tag_info["rating"]), np.percentile(tag_info["rating"], 25),
              np.percentile(tag_info["rating"], 40),
              np.percentile(tag_info["rating"], 60),
              np.percentile(tag_info["rating"], 75),
              np.max(tag_info["rating"]))'''

        #print(review_vs_rating, review_vs_count, rating_vs_count)
        #print(tag_info.sort_values(by=['rev_catego', 'rat_catego', 'count_catego'], ascending=False)[['rev_catego', 'rat_catego', 'count_catego']])
        print(tag_info.sort_values(by=['rating'], ascending=False)["rating"])
        #print(sum(tag_info["rating"]))

        return tag_info

    def _get_locational_cluster(self, latitude, longitude):

        p = Point(longitude, latitude)

        area = "Unknown"

        for poly in self.sg_poly["central"]:
            if Polygon(poly).contains(p):
                area = "Central"
        if latitude == 1.2237 and longitude == 103.85889 or \
            latitude == 1.2260739999999999 and longitude == 103.752578:
            area = "Central"

        for poly in self.sg_poly["north"]:
            if Polygon(poly).contains(p):
                area = "North"

        for poly in self.sg_poly["east"]:
            if Polygon(poly).contains(p):
                area = "East"
        if latitude == 1.395197 and longitude == 103.99177399999999 or \
            latitude == 1.392677 and longitude == 103.979648 or \
                latitude == 1.3920350000000001 and longitude == 103.97587 or \
            latitude == 1.392677 and longitude == 103.979648:
            area = "East"

        for poly in self.sg_poly["north-east"]:
            if Polygon(poly).contains(p):
                area = "North-East"

        for poly in self.sg_poly["west"]:
            if Polygon(poly).contains(p):
                area = "West"

        if area == "Unknown":
            print(latitude,longitude)
        return area

    def _get_specific_info(self, sub_info):
        tag_info = self.get_tag_info("all")

        #print(sub_info)
        # includes get tag, % and location
        grouped = sub_info.groupby([self.tag_type])
        group_info = grouped.mean()
        group_info = group_info.drop(["reviews", "rating", "done", "latitude", "longitude"], axis=1)
        group_info["count"] = grouped[[self.tag_type]].count()
        group_info = pd.merge(group_info, tag_info, on=self.tag_type, how="inner")
        group_info = group_info.drop(["reviews", "rating"], axis=1)
        group_info = group_info.rename(columns={"count_x": "count", "count_y": "total"})
        group_info["concentration"] = group_info["count"]/group_info["total"] * 100

        #location info
        locations = set(sub_info["location"])
        for loc in locations:
            group_info[loc] = [len(sub_info[(sub_info[self.tag_type] == i) &
                                            (sub_info["location"] == loc)])
                                    for i, row in group_info.iterrows()]

        for loc in locations:
            group_info[f"%{loc}"] = [len(sub_info[(sub_info[self.tag_type] == i) &
                                            (sub_info["location"] == loc)]) * 1.0 /
                                     float(group_info[(group_info.index == i)]["count"]) * 100
                                    for i, row in group_info.iterrows()]

        #print(group_info)
        return group_info


    def draw_gradient_scatter_on_map(self, cluster_by):
        basic_info = self.get_basic_file_info()

        p90 = np.percentile(basic_info["reviews"], 90)
        basic_info["reviews"] = [min(r, p90) for r in basic_info["reviews"]]
        print(p90)

        fig, ax = plt.subplots(figsize=(8, 7))

        BBox = ((103.5599, 104.1353, 1.1858, 1.5036))
        sgmap = plt.imread(f"{self.dir}/singapore.jpg")
        ax.set_title('Plotting Spatial Data on Singapore Map')
        ax.set_xlim(BBox[0], BBox[1])
        ax.set_ylim(BBox[2], BBox[3])
        im = ax.imshow(sgmap, zorder=0, alpha=0.5, extent=BBox, aspect='equal')

        plt.scatter(basic_info.longitude, basic_info.latitude, alpha=1,
                   c=basic_info[cluster_by], s=30, edgecolor='black', cmap="Blues")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = plt.colorbar(cax=cax)
        cbar.set_label(cluster_by)

        plt.show()

    def cluster_location(self):
        basic_info = self.get_basic_file_info()

        #coords = basic_info[["latitude","longitude"]].values
        #db = DBSCAN(eps=4/6371, min_samples=1, algorithm='auto', metric='haversine').fit(np.radians(coords))

        basic_info["location"] = basic_info.apply(lambda x: self._get_locational_cluster(x.latitude, x.longitude), axis=1)

        clusters = list(basic_info["location"])
        print(basic_info[(basic_info["location"]=="Central") & (basic_info[self.tag_type]=="japanese")])

        #get info for each cluster
        for cluster in set(clusters):
            sub_info = basic_info[basic_info["location"] == cluster]

            spec = self._get_specific_info(sub_info)
            print(cluster, ", total attractions:", sum(spec["count"]))
            print(spec.sort_values(by=["concentration"], ascending=False)[["count","total","concentration"]])

        #print(basic_info.sort_values(by=["location"], ascending=False))

        #draw map
        BBox = ((103.5599, 104.1353, 1.1858, 1.5036))

        sgmap = plt.imread(f"{self.dir}/singapore.jpg")

        #colors = [self._rgb2hex(list(np.random.choice(range(256), size=3))) for i in range(0,len(clusters))]
        colors = ["red","green","blue","yellow","cyan","magenta"]

        fig, ax = plt.subplots(figsize=(8, 7))
        i = 0
        for location, df in basic_info.groupby("location"):

            #plt.scatter(dff['x'], dff['y'], s=size, c=cmap(norm(dff['colors'])),
            #            edgecolors='none', label="Feature {:g}".format(i))
            ax.scatter(df.longitude, df.latitude, zorder=1, alpha=1,
                       c=colors[i], s=30, edgecolor='black',
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

    def cluster_tag_info(self, no_clust, incl_quant):
        tag_info = self.get_tag_info("all")
        #print(tag_info)

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

        ct_vals = list(tag_info["count"])
        tag_info["count_scaled"] = [(c - min(ct_vals)) / (max(ct_vals) - min(ct_vals)) for c in ct_vals]

        rev_vals = list(tag_info["reviews"])
        tag_info["reviews_scaled"] = [(c - min(rev_vals)) / (max(rev_vals) - min(rev_vals)) for c in rev_vals]

        rate_vals = list(tag_info["rating"])
        tag_info["rating_scaled"] = [(c - min(rate_vals)) / (max(rate_vals) - min(rate_vals)) for c in rate_vals]

        #data preparation
        data = []
        id2tag = {}
        id = 0
        for tag, row in tag_info.iterrows():
            id2tag[id] = tag
            if incl_quant:
                data.append([row["count_scaled"], row["reviews_scaled"], row["rating_scaled"]])
                vals = [ct_vals, rev_vals, rate_vals]
            else:
                data.append([row["reviews_scaled"], row["rating_scaled"]])
                vals = [rev_vals, rate_vals]

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
                #print(tag_info[tag_info.index.isin(attr_in_the_cluster)])
                #print("rating sd",np.average(tag_info[tag_info.index.isin(attr_in_the_cluster)]["rating_sd"]))
                print("number,reviews,rating")
                to_be_printed = ""
                for i in range(0, len(kmeans.cluster_centers_[n])):
                    z_score = kmeans.cluster_centers_[n][i]
                    to_be_printed += f"{z_score * (max(vals[i]) - min(vals[i])) + min(vals[i])},"
                print(to_be_printed[0:len(to_be_printed) - 1])



#viz = Visualize("visualize","bottom500.csv","tripadvisor","tag")
viz = Visualize("visualize","yelp_bottom782.csv","yelp","region_tag")
#viz.get_basic_file_info()
#viz.cluster_tag_info(10, False)
#viz.get_tag_info("all")
#viz.get_rating_of_rarely_visited_places()
#viz.get_local_visitor_stats()
viz.cluster_location()
#viz.get_local_visitor_specs("low","med", "lrat_catego", "vrat_catego")
#viz.get_local_visitor_category("theatres", "lrat_catego", "vrat_catego")
#viz.draw_gradient_scatter_on_map("rating")
