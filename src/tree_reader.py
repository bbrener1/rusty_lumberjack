import matplotlib.pyplot as plt
import numpy as np
import re
import json
import sys
import os
import random
import glob

import scipy.special
from scipy.stats import linregress
from scipy.spatial.distance import jaccard
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS

from hdbscan import HDBSCAN

from scipy.cluster import hierarchy as hrc
from sklearn.decomposition import PCA

from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import jaccard_similarity_score
jaccard_index = jaccard_similarity_score

import copy

class Node:

    def __init__(self, node_json,tree,forest,prerequisites=None,level=0):
        if prerequisites is None:
            prerequisites = []
        self.tree = tree
        self.forest = forest
        self.level = level
        self.feature = node_json['feature']
        self.split = node_json['split']
        self.features = node_json['features']
        self.samples = node_json['samples']
        self.medians = node_json['medians']
        self.dispersions = node_json['dispersions']
        self.local_gains = node_json['local_gains']
        self.absolute_gains = node_json['absolute_gains']
        self.children = []
        self.prerequisites = prerequisites
        if len(node_json['children']) > 0:
            self.children.append(Node(node_json['children'][0],self.tree,self.forest,prerequisites = prerequisites + [(self.feature,self.split,'<')],level=level+1))
            self.children.append(Node(node_json['children'][1],self.tree,self.forest,prerequisites = prerequisites + [(self.feature,self.split,'>')],level=level+1))

    def nodes(self):
        nodes = []
        for child in self.children:
            nodes.extend(child.nodes())
        for child in self.children:
            nodes.append(child)
        return nodes

    def leaves(self):
        leaves = []
        for child in self.children:
            leaves.extend(child.leaves())
        if len(leaves) < 1:
            for child in self.children:
                leaves.append(child)
        return leaves

    def nodes_by_level(self):
        levels = [[self]]
        for child in self.children:
            child_levels = child.nodes_by_level()
            for i,child_level in enumerate(child_levels):
                if len(levels) < i+1:
                    levels.push([])
                levels[i+1].extend(child_level)
        levels


    def feature_levels(self):
        feature_levels = []
        for child in self.children:
            feature_levels.extend(child.feature_levels())
        feature_levels.append((self.feature,self.level))
        return feature_levels

    def l2_fit(self,truth_dictionary):
        l2_sum = 0.0
        samples = self.sample_index()
        sliced_counts = self.forest.counts[samples]
        for f,feature in enumerate(self.features):
            global_feature_index = truth_dictionary.feature_dictionary[feature]
            feature_l2_sum = 0.0
            for s,sample in enumerate(self.samples):
                truth = truth_dictionary.look(sample,feature)
                feature_l2_sum += (truth - self.medians[f])**2

#             l2_sum += np.sqrt(feature_l2_sum)
            l2_sum += feature_l2_sum
        return l2_sum
#             print((np.var(sliced_counts[:,global_feature_index]),self.dispersions[f]))

    def level(self,target):
        level_nodes = []
        if len(self.children) > 0:
            if self.level <= target:
                level_nodes.extend(self.children[0].level(target))
                level_nodes.extend(self.children[1].level(target))
        else:
            level_nodes.append(self)
        return level_nodes


    def feature_index(self, truth_dictionary=None):
        if truth_dictionary is None:
            truth_dictionary = self.forest.truth_dictionary
        feature_index = np.zeros(len(truth_dictionary.feature_dictionary),dtype=bool)
        for feature in self.features:
            feature_index[truth_dictionary.feature_dictionary[feature]] = True
        return feature_index

    def sample_index(self, truth_dictionary=None):
        if truth_dictionary is None:
            truth_dictionary = self.forest.truth_dictionary
        sample_index = np.zeros(len(truth_dictionary.obs_dictionary),dtype=bool)
        for sample in self.samples:
            sample_index[truth_dictionary.obs_dictionary[sample]] = True
        # if np.sum(sample_index) == 0:
        #     print(self.samples)
        #     print(self.prerequisites)
        #     raise IndexError
        return sample_index

    def feature_sample_boolean_index(self,truth_dictionary=None):
        return self.sample_index(truth_dictionary),self.feature_index(truth_dictionary)

    def node_counts(self,counts,truth_dictionary=None):
        return counts[self.sample_index()].T[self.feature_index()].T

    def split_feature_index(self):
        split_feature_index = self.features.index(self.feature)

    def singly_sorted_counts(self):
        counts = self.node_counts()
        split_feature_index = self.split_feature_index()
        node_sample_split_sort = np.argsort(counts[:,split_feature_index])
        return counts[node_sample_split_sort]

    def split_sample_index(self):
        counts = self.singly_sorted_counts()
        return np.sum(counts[:,self.split_feature_index] > float(self.split))


class Tree:

    def __init__(self, tree_json, forest):
        self.root = Node(tree_json, self, forest)
        self.forest = forest

    def nodes(self):
        nodes = []
        nodes.extend(self.root.nodes())
        nodes.append(self.root)
        return nodes

    def leaves(self):
        leaves = self.root.leaves()
        if len(leaves) < 1:
            leaves.append(self.root)
        return leaves

    def level(self,target):
        level_nodes = []
        for node in self.nodes():
            if node.level == target:
                level_nodes.append(node)
        return level_nodes

    def seek(self,directions):
        if len(directions) > 0:
            self.children[directions[0]].seek(directions[1:])
        else:
            return self

    def l2_fit_leaves(self):
        l2_sum = 0.0
        leaves = 0
        for node in self.leaves():
            l2_sum += node.l2_fit()
        return l2_sum/leaves

    def feature_levels(self):
        return self.root.feature_levels()

    def plotting_representation(self,width=10,height=10):
        coordinates = []
        connectivities = []
        levels = self.root.nodes_by_level()
        jump = height / len(levels)
        for i,level in enumerate(levels):
            level_samples = sum([len(node.samples) for node in level])
            next_level_samples = 0
            if i < len(levels):
                next_level_samples = sum([len(node.samples) for node in levels[i+1]])
            consumed_width = 0
            next_consumed_width = 0
            for j,node in enumerate(level):
                sample_weight = float(len(node.samples)) / float(level_samples)
                half_width = (width * sample_weight)/2
                center = consumed_width + half_width
                consumed_width = consumed_width + (half_width * 2)
                coordinates.append((i*jump,center)))
                if i < len(levels):
                    for child in node.children:
                        child_sample_weight = float(len(child.samples)) / float(next_level_samples)
                        child_half_width = (width * sample_weight)/2
                        child_center = next_consumed_width + child_half_width
                        next_consumed_width = next_consumed_width + (child_half_width * 2)
                        connectivities.append((i*jump,j*(center),((i+1)*jump,child_center)))
        np.array(coordinates)
        plt.figure()
        plt.scatter(coordinates[:,0],[:,1])
        for connection in connectivities:
            plot(connection[0],connection[1])
        plt.show()

        # return coordinates,connectivities


    def summary(self, verbose=True):
        nodes = len(self.nodes)
        leaves = len(self.leaves)
        if verbose:
            print("Nodes: {}".format(nodes))
            print("Leaves: {}".format(leaves))

class Forest:

    def __init__(self,trees,counts,features=None,samples=None):
        if features is None:
            features = list(map(lambda x: str(x),range(counts.shape[1])))
        if samples is None:
            samples = list(map(lambda x: str(x),range(counts.shape[0])))
        self.truth_dictionary = TruthDictionary(counts,features,samples)

        self.counts = counts
        self.features = features
        self.samples = samples

        self.dim = counts.shape

        self.trees = list(map(lambda x: Tree(x,self),trees))

    def nodes(self):
        nodes = []
        for tree in self.trees:
            nodes.extend(tree.nodes())
        return nodes

    def leaves(self):
        leaves = []
        for tree in self.trees:
            leaves.extend(tree.leaves())
        return leaves

    def level(self,target):
        level = []
        for tree in self.trees:
            level.extend(tree.level(target))
        return level

    def load(location, prefix="/run.*.compact", header="/run.prediction_header",truth="run.prediction_truth"):
        combined_tree_files = glob.glob(location + prefix)

        print(combined_tree_files)

        raw_forest = []

        for tree_file in combined_tree_files:
            raw_forest.append(json.load(open(tree_file.strip())))

        counts = np.loadtxt(location+truth)
        header = np.loadtxt(location+header,dtype=str)

        first_forest = Forest(raw_forest[1:],counts,features=header)

        return first_forest

class TruthDictionary:

    def __init__(self,counts,header,observations=None):

        self.counts = counts
        self.header = header
        self.feature_dictionary = {}
        self.obs_dictionary = {}
        for i,feature in enumerate(header):
            self.feature_dictionary[feature.strip('""').strip("''")] = i
        if observations is None:
            observations = map(lambda x: str(x),range(counts.shape[0]))
        for i,observation in enumerate(observations):
            self.obs_dictionary[observation.strip("''").strip('""')] = i

    def look(self,observation,feature):
#         print(feature)
        return(self.counts[self.obs_dictionary[observation],self.feature_dictionary[feature]])



################################
################################
################################
################################
#
#       Rest of module
#       Split out later
#
################################
################################
################################
################################

from sklearn.cluster import AgglomerativeClustering
import community
import networkx as nx


def node_sample_encoding(nodes,samples):

    encoding = np.zeros((len(nodes),samples),dtype=bool)
    for i,node in enumerate(nodes):
        encoding[i] = node.sample_index()
    return encoding

def sample_node_encoding(nodes,samples):
    encoding = np.zeros((len(nodes),samples),dtype=bool)
    for i,node in enumerate(nodes):
        encoding[i] = node.sample_index()
    sample_encoding = encoding.T
    unrepresented = np.sum(sample_encoding,axis=1) == 0
    if np.sum(unrepresented) > 0:
        sample_encoding[unrepresented] = 1;
    return sample_encoding

def coocurrence_matrix(sample_encoding):

#     co_mtx = np.zeros((sample_encoding.shape[1],sample_encoding.shape[1]))
#     for node in sample_encoding:
#         node_tile = np.tile(node,(node.shape[0],1))
#         intersect = np.logical_and(node_tile,node_tile.T)
#         co_mtx += intersect.astype(dtype=int)
#     return co_mtx

    co_mtx = np.matmul(sample_encoding.astype(dtype=int),sample_encoding.T.astype(dtype=int)).astype(dtype=bool)

    print(co_mtx.shape)

    return np.array(co_mtx)

def coocurrence_distance(sample_encoding):

    co_mtx = coocurrence_matrix(sample_encoding)
    distance_mtx = 1.0/(co_mtx + 1)
    return distance_mtx

def node_tsne_sample(nodes,samples):

    encoding = node_sample_encoding(nodes,samples)
    embedding_model = TSNE(n_components=2,metric='correlation')
    coordinates = embedding_model.fit_transform(encoding)

    #     distances = coocurrence_matrix(encoding)

    return coordinates

def sample_tsne(nodes,samples):

    encoding = node_sample_encoding(nodes,samples)
    print("TSNE Encoding: {}".format(encoding.shape))
    pre_computed_distance = coocurrence_distance(encoding.T)
#     pre_computed_distance = scipy.spatial.distance.squareform(pdist(encoding.T,metric='jaccard'))
    print("TSNE Distance Matrix: {}".format(pre_computed_distance.shape))
#     pre_computed_distance[pre_computed_distance == 0] += .000001
    pre_computed_distance[np.isnan(pre_computed_distance)] = 10000000
    embedding_model = TSNE(n_components=2,metric='precomputed')
    coordinates = embedding_model.fit_transform(pre_computed_distance)

    #     distances = coocurrence_matrix(encoding)

    return coordinates

def node_hdbscan_samples(nodes,samples):

    node_encoding = node_sample_encoding(nodes,samples)

    pre_computed_distance = pdist(node_encoding,metric='cityblock')

    clustering_model = HDBSCAN(min_cluster_size=50, metric='precomputed')

#     plt.figure()
#     plt.title("Dbscan observed distances")
#     plt.hist(pre_computed_distance,bins=50)
#     plt.show()

    clusters = clustering_model.fit_predict(scipy.spatial.distance.squareform(pre_computed_distance))

#     clusters = clustering_model.fit_predict(node_encoding)

    return clusters

def node_gain_table(nodes,forest):
    node_gain_table = np.zeros((len(nodes),len(forest.features)))
    for i,node in enumerate(nodes):
        for j,feature in enumerate(node.features):
            feature_index = forest.truth_dictionary.feature_dictionary[feature]
            try:
                feature_gain = node.absolute_gains[j]
                node_gain_table[i,feature_index] = feature_gain
            except:
                print(node.absolute_gains)
    return node_gain_table

def hacked_louvain(nodes,samples):

    node_encoding = node_sample_encoding(nodes,len(samples))
    print("Louvain Encoding: {}".format(node_encoding.shape))
    pre_computed_distance = coocurrence_distance(node_encoding.T)
    print("Louvain Distances: {}".format(pre_computed_distance.shape))
    sample_graph = nx.from_numpy_matrix(pre_computed_distance)
    print("Louvain Cast To Graph")
    least_spanning_tree = nx.minimum_spanning_tree(sample_graph)
    print("Louvain Least Spanning Tree constructed")
    part_dict = community.best_partition(least_spanning_tree)
    print("Louvain Partition Done")
    clustering = np.zeros(len(part_dict))
    for i,sample in enumerate(samples):
        clustering[i] = part_dict[int(sample)]
    print("Louvain: {}".format(clustering.shape))
    return clustering

def embedded_hdbscan(coordinates):

    clustering_model = HDBSCAN(min_cluster_size=50)
    clusters = clustering_model.fit_predict(coordinates)
    return clusters

def sample_hdbscan(nodes,samples):

    node_encoding = node_sample_encoding(nodes,samples)
    embedding_model = PCA(n_components=100)
    pre_computed_embedded = embedding_model.fit_transform(node_encoding.T)
    print("Sample HDBscan Encoding: {}".format(pre_computed_embedded.shape))
#     pre_computed_distance = coocurrence_distance(node_encoding)
    pre_computed_distance = scipy.spatial.distance.squareform(pdist(pre_computed_embedded,metric='correlation'))
    print("Sample HDBscan Distance Matrix: {}".format(pre_computed_distance.shape))
#     pre_computed_distance[pre_computed_distance == 0] += .000001
    pre_computed_distance[np.isnan(pre_computed_distance)] = 10000000
    clustering_model = HDBSCAN(min_samples=3,metric='precomputed')
    clusters = clustering_model.fit_predict(pre_computed_distance)

    return clusters

def sample_agglomerative(nodes,samples,n_clusters):

    node_encoding = node_sample_encoding(nodes,samples)

    pre_computed_distance = pdist(node_encoding.T,metric='cosine')

    clustering_model = AgglomerativeClustering(n_clusters=n_clusters,affinity='precomputed')

    clusters = clustering_model.fit_predict(scipy.spatial.distance.squareform(pre_computed_distance))

#     clusters = clustering_model.fit_predict(node_encoding)

return clusters
