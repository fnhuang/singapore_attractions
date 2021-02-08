import numpy as np
import random
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import KMeans, DBSCAN
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


    def get_local_visitor_stats(self):
        if self.data_source == "yelp":
            info = pd.read_csv(f"{self.dir}/yelp_local_visitor_stat.csv")
        else:
            info = pd.read_csv(f"{self.dir}/local_visitor_stat.csv")

        basic_info = self.get_basic_file_info()[["csv", "reviews", "rating"]]
        #data for Yelp is duplicated if there are more than one region
        basic_info = basic_info.drop_duplicates(subset=['csv'])

        l_reviews = list(info["l_reviews"])
        v_reviews = list(info["v_reviews"])
        a_reviews = list(info["a_reviews"])
        reviews = list(basic_info["reviews"])
        print("TOTAL")
        print(sum(l_reviews))
        print(sum(v_reviews))
        print(sum(a_reviews))
        print((sum(l_reviews) + sum(v_reviews)) * 1.0 / sum(reviews))

        l_ratings = list(info["l_avg_ratings"])
        v_ratings = list(info["v_avg_ratings"])
        a_ratings = list(info["a_avg_ratings"])
        ratings = list(basic_info["rating"])

        reviews_data = [l_reviews, v_reviews, reviews]
        ratings_data = [l_ratings, v_ratings, ratings]

        # Create a figure instance
        fig = plt.figure(1, figsize=(9, 6))
        ax = fig.add_subplot(111)
        ax.set_title("Locals vs Tourists Reviews", fontsize=18)

        print(np.min(l_reviews), np.percentile(l_reviews,25), np.percentile(l_reviews,50),
            np.percentile(l_reviews,75), np.max(l_reviews), np.average(l_reviews), np.std(l_reviews))
        print(np.min(v_reviews), np.percentile(v_reviews, 25), np.percentile(v_reviews, 50),
            np.percentile(v_reviews, 75), np.max(v_reviews), np.average(v_reviews), np.std(v_reviews))
        print(np.min(reviews), np.percentile(reviews, 25), np.percentile(reviews, 50),
              np.percentile(reviews, 75), np.max(reviews), np.average(reviews), np.std(reviews))


        # Create boxplot of reviews and ratings
        bp = ax.boxplot(reviews_data, showfliers=False)
        plt.xticks([1, 2, 3], ['locals', 'tourists', "overall"],fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

        plt.close()

        fig = plt.figure(1, figsize=(9, 6))
        ax = fig.add_subplot(111)
        ax.set_title("Locals vs Tourists Ratings", fontsize=18)

        bp = ax.boxplot(ratings_data, showfliers=False)
        plt.xticks([1, 2, 3], ['locals', 'tourists', "all"], fontsize=14)
        plt.yticks(fontsize=14)
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
        plt.close()

        plt.close()
        #print(info)

        return info


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
            info.loc[index, self.tag_type] = tags[0]
            tags.pop(0)
            for t in tags:
                row[self.tag_type] = t
                info = info.append(row, ignore_index=True)

        #print(info[info["latitude"].isnull()][["csv"]])
        #print(info.shape)
        #print(info.tail(5))
        info.to_csv('basic_info.csv', encoding='utf-8')
        return info


    def _get_k(self, X):
        wcss = []
        for i in range(1, 21):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, 21), wcss)
        #plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.show()



    def cluster_rating_review_quantity(self, reviewer):
        tag_info = self.get_info_group_by_tag(reviewer)[["count_pctl", "rev_pctl", "rat_pctl"]]

        count = list(tag_info["count_pctl"])
        review = list(tag_info["rev_pctl"])
        rating = list(tag_info["rat_pctl"])
        tag = list(tag_info.index)

        #cluster the tags
        feature = []
        for i in range(len(count)):
            feature.append([rating[i],review[i],count[i]])
        X = np.array(feature)

        #self._get_k(X)

        kmeans = KMeans(n_clusters=8, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)

        colors = ['#000099', '#990000', '#009933', '#cc0099',
                  '#336699', '#333300', '#cc9900', '#ff5c33', '#ff0066']
        '''r = lambda: random.randint(0, 255)
        for o in range(9):
            colors.append(f'#%02X%02X%02X' % (r(), r(), r()))'''

        print(kmeans.labels_)
        for i in range(8):
            print("Cluster", i)
            indices = [j for j, x in enumerate(kmeans.labels_) if x == i]
            places = [tag[j] for j in indices]
            print(",".join(places))

        ax = plt.subplot(111, projection='3d')

        #ax.scatter(rating, review, count, color='r')

        for i in range(len(rating)):  # plot each point + it's index as text above
            ax.bar3d(rating[i], review[i], 0, 0.001, 0.001, count[i], color=colors[kmeans.labels_[i]])
            ax.scatter(rating[i], review[i], count[i], color=colors[kmeans.labels_[i]])
            ax.text(rating[i], review[i], count[i]+0.1,
                    '%s' % (tag[i][0:4]), size=6, zorder=1,
                    color='k')
            #print(rating[i], review[i], count[i], tag[i])

        ax.set_xlabel('Rating')
        ax.set_ylabel('Review')
        ax.set_zlabel('Count')

        plt.show()


    def get_info_group_by_tag(self, reviewer):
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

        tag_info["%neg_rating"] = 0.0
        for i, row in tag_info.iterrows():
            rating_list = basic_info[basic_info[self.tag_type] == i]["rating"].tolist()
            neg_rating = [int(r < 4) for r in rating_list]
            percent_neg = sum(neg_rating) * 1.0/len(neg_rating)
            tag_info.at[i, '%neg_rating'] = percent_neg



        #get correlation info
        review_vs_rating = pearsonr(np.array(tag_info["reviews"]).astype(np.float),
                          np.array(tag_info["rating"]).astype(np.float))
        review_vs_count = pearsonr(np.array(tag_info["reviews"]).astype(np.float),
                                    np.array(tag_info["count"]).astype(np.float))
        rating_vs_count = pearsonr(np.array(tag_info["rating"]).astype(np.float),
                                   np.array(tag_info["count"]).astype(np.float))


        reviews = list(tag_info["reviews"])
        tag_info["rev_pctl"] = [sum(np.array(reviews) <= v) / len(reviews) for v in reviews]
        tag_info["rev_catego"] = [self._get_catego(reviews, v) for v in reviews]

        ratings = list(tag_info["rating"])
        tag_info["rat_pctl"] = [sum(np.array(ratings) <= v) / len(ratings) for v in ratings]
        tag_info["rat_catego"] = [self._get_catego(ratings, v) for v in ratings]

        counts = list(tag_info["count"])
        tag_info["count_pctl"] = [sum(np.array(counts) <= v) / len(counts) for v in counts]
        tag_info["count_catego"] = [self._get_catego(counts, v) for v in counts]

        tag_info["overall_demand"] = [(list(tag_info["rev_pctl"])[i] + list(tag_info["rat_pctl"])[i])/2 for i in range(0,len(tag_info))]
        tag_info["mismatch"] = [(list(tag_info["count_pctl"])[i] - list(tag_info["overall_demand"])[i]) for i in range(0,len(tag_info))]
        tag_info["abs_mismatch"] = [abs(list(tag_info["mismatch"])[i]) for i in range(0,len(tag_info))]
        tag_info["MAA"] = [(list(tag_info["overall_demand"])[i] / (list(tag_info["abs_mismatch"])[i] + 1)) for i in range(0,len(tag_info))]



        #print(review_vs_rating, review_vs_count, rating_vs_count)
        #print(tag_info.sort_values(by=['mismatch'], ascending=False)[['count','count_pctl','reviews',
        # 'rev_pctl','rating','rat_pctl','overall_demand', 'mismatch', 'MAA']])
        #print(tag_info.sort_values(by=['rating'], ascending=False)["rating"])
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
        if latitude == 1.395197 and longitude == 103.991774 or \
            latitude == 1.392677 and longitude == 103.979648 or \
                latitude == 1.392035 and longitude == 103.97587 or \
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
        tag_info = self.get_info_group_by_tag("all")

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



    def cluster_location(self):
        basic_info = self.get_basic_file_info()

        basic_info["location"] = basic_info.apply(lambda x: self._get_locational_cluster(x.latitude, x.longitude), axis=1)

        clusters = list(basic_info["location"])

        #print(basic_info)
        #filter
        print(basic_info)
        basic_info = basic_info[(basic_info[self.tag_type]=="shopping malls") |
                                (basic_info[self.tag_type]=="") |
                                (basic_info[self.tag_type] == "")]


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

        #ax.set_title('POIs with Fewer Than 25 Reviews')
        ax.set_xlim(BBox[0], BBox[1])
        ax.set_ylim(BBox[2], BBox[3])
        im = ax.imshow(sgmap, zorder=0, alpha=0.5,  extent=BBox, aspect='equal')


        plt.legend()
        plt.show()



    def _get_localvforeign_word_similarity_for_sentiment(self, top_k, sentiment):
        home_folder = f"{self.dir}/{self.data_source}/{sentiment}_results"

        writer = open(f"{home_folder}/similarity.csv", "w", encoding="utf8")
        writer.write("food type,similarity\n")

        for file_name in os.listdir(f"{home_folder}/local_v_foreign"):
            similarity = 0

            if "_ovs" in file_name:
                ovs_file = f"{home_folder}/local_v_foreign/{file_name}"
                local_file = f"{home_folder}/local_v_foreign/{file_name.replace('_ovs','_sgp')}"

                word_list = []

                with open(ovs_file, "r", encoding="utf8") as reader:
                    lines = reader.readlines()

                    for line in lines:
                        datas = line.split(",")
                        word_list.append(datas[0])

                with open(local_file, "r", encoding="utf8") as reader:
                    lines = reader.readlines()

                    for line in lines:
                        datas = line.split(",")
                        if datas[0] in word_list:
                            index = word_list.index(datas[0])

                            similarity += 1.0
                        else:
                            word_list.append(datas[0])

                # Document similarity
                similarity = similarity / len(word_list)

                writer.write(f"{file_name.replace('_ovs.csv','')},{similarity}\n")
                writer.flush()

        writer.close()

    def _draw_word_graph_for_sentiment(self, top_k, sentiment):
        home_folder = f"{self.dir}/{self.data_source}/{sentiment}_results"

        for file_name in os.listdir(f"{home_folder}/local_v_foreign"):

            if "_ovs" in file_name:
                ovs_file = f"{home_folder}/local_v_foreign/{file_name}"
                local_file = f"{home_folder}/local_v_foreign/{file_name.replace('_ovs','_sgp')}"

                word_list = []
                foreign_values = []
                local_values = []

                with open(ovs_file,"r",encoding="utf8") as reader:
                    lines = reader.readlines()

                    for line in lines[0:top_k]:
                        datas = line.split(",")
                        word_list.append(datas[0])
                        foreign_values.append(float(datas[1]))
                        local_values.append(0)

                with open(local_file,"r",encoding="utf8") as reader:
                    lines = reader.readlines()

                    for line in lines[0:top_k]:
                        datas = line.split(",")
                        if datas[0] in word_list:
                            index = word_list.index(datas[0])
                            foreign_values[index] = float(datas[1])
                            local_values[index] = float(datas[1])

                        else:
                            word_list.append(datas[0])
                            local_values.append(float(datas[1]))
                            foreign_values.append(0)

                '''if "asian fusion" in file_name:
                    print(len(local_values), len(foreign_values))
                    for i in range(0, len(word_list)):
                        print(word_list[i], local_values[i], foreign_values[i])
                    sys.exit()'''

                # Start drawing graph
                ind = np.arange(len(foreign_values))  # the x locations for the groups
                width = 0.35

                fig, ax = plt.subplots(figsize=(15,8))
                rects1 = ax.bar(ind - width / 2, foreign_values, width, label='tourists')
                rects2 = ax.bar(ind + width / 2, local_values, width, label='locals')

                ax.set_ylabel('Scores')
                senti = "negative" if sentiment == "neg" else "positive"
                ax.set_title(f"Top {top_k} {senti} words "
                             f"for {file_name.replace('_ovs.csv','')} food places")
                ax.set_xticks(ind)
                ax.set_xticklabels(word_list, rotation=90, fontsize=8)
                ax.legend()

                fig.tight_layout()

                os.makedirs(os.path.dirname(f"{home_folder}/graphs/"), exist_ok=True)
                plt.savefig(f"{home_folder}/graphs/{file_name.replace('_ovs.csv','.png')}")
                plt.close()


    def draw_word_graph(self, top_k):
        self._draw_word_graph_for_sentiment(top_k, "neg")
        self._draw_word_graph_for_sentiment(top_k, "pos")

viz = Visualize("visualize","top299.csv","tripadvisor","tag")
#viz = Visualize("visualize","yelp_top97.csv","yelp","region_tag")
viz.cluster_location()
viz.cluster_rating_review_quantity("all")
