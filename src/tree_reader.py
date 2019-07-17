import matplotlib.pyplot as plt
import numpy as np
import re
import json
import sys
import os
import random
import glob
import pickle

from functools import reduce

import scipy.special
from scipy.stats import linregress
from scipy.spatial.distance import jaccard
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.optimize import nnls

from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS
from sklearn.linear_model import Ridge,Lasso
from sklearn.decomposition import NMF

# from hdbscan import HDBSCAN

from scipy.cluster import hierarchy as hrc
from scipy.cluster.hierarchy import dendrogram,linkage

from sklearn.decomposition import PCA

from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import jaccard_similarity_score
jaccard_index = jaccard_similarity_score

from multiprocessing import Pool

import copy

try:
    sys.path.append("/Users/boris/haxx/python/smooth_density_graph")
except:
    pass
try:
    sys.path.append("../smooth_density_graph/")
except:
    pass
import smooth_density_graph as sdg

class Node:

    def __init__(self, node_json,tree,forest,parent=None,lr=None,prerequisites=None,level=0):
        if prerequisites is None:
            prerequisites = []
        self.tree = tree
        self.forest = forest
        self.parent = parent
        self.lr = lr
        self.level = level
        self.feature = node_json['feature']
        self.split = node_json['split']
        self.prerequisites = node_json['prerequisites']
        self.braids = node_json['braids']
        self.features = [f['name'] for f in node_json['features']]
        self.samples = [s['name'] for s in node_json['samples']]
        self.medians = node_json['medians']
        self.dispersions = node_json['dispersions']
        self.weights = np.ones(len(self.features),dtype=float)
        self.local_gains = node_json['local_gains']
        self.absolute_gains = node_json['absolute_gains']
        self.children = []
        self.child_clusters = ([],[])
        self.prerequisites = prerequisites
        if len(node_json['children']) > 0:
            self.children.append(Node(node_json['children'][0],self.tree,self.forest,parent=self,lr=0,prerequisites = prerequisites + [(self.feature,self.split,'<')],level=level+1))
            self.children.append(Node(node_json['children'][1],self.tree,self.forest,parent=self,lr=1,prerequisites = prerequisites + [(self.feature,self.split,'>')],level=level+1))

    def null():
        null_dictionary = {}
        null_dictionary['feature'] = None
        null_dictionary['split'] = None
        null_dictionary['features'] = []
        null_dictionary['samples'] = []
        null_dictionary['medians'] = []
        null_dictionary['dispersions'] = []
        null_dictionary['local_gains'] = None
        null_dictionary['absolute_gains'] = None
        null_dictionary['children'] = []
        return Node(null_dictionary,None,None)

    def test_node(feature,split,features,samples,medians,dispersions):
        test_node = Node.null()
        test_node.feature = feature
        test_node.split = split
        test_node.features.extend(features)
        test_node.samples.extend(samples)
        test_node.medians.extend(medians)
        test_node.dispersions.extend(dispersions)
        return test_node

    def nodes(self):
        nodes = []
        for child in self.children:
            nodes.extend(child.nodes())
        for child in self.children:
            nodes.append(child)
        return nodes

    def cluster_nodes(self,cluster):
        nodes = []
        if cluster in self.child_clusters[0]:
            nodes.extend(self.children[0].cluster_nodes(cluster))
        if cluster in self.child_clusters[1]:
            nodes.extend(self.children[1].cluster_nodes(cluster))
        if self.cluster == cluster:
            nodes.append(self)
        return nodes

    def leaves(self):
        leaves = []
        for child in self.children:
            leaves.extend(child.leaves())
        if len(leaves) < 1:
            for child in self.children:
                leaves.append(child)
        return leaves

    def stems(self):
        stems = []
        for child in self.children:
            stems.extend(child.stems())
        for child in self.children:
            if len(child.children) > 0:
                stems.append(child)
        return stems

    def sister(self):
        if self.parent is None:
            return None
        else:
            for child in self.parent.children:
                if child is not self:
                    return child
            return None

    def descend(self,n):
        nodes = []
        if n > 0:
            for child in self.children:
                nodes.extend(child.descend(n-1))
            if len(nodes) < 1:
                nodes.append(self)
        else:
            nodes.append(self)
        return nodes

    def root(self):
        if self.parent is not None:
            return self.parent.root()
        else:
            return self

    def ancestors(self):
        ancestors = []
        if self.parent is not None:
            ancestors.append(parent)
            ancestors.extend(self.parent.ancestors())
        return ancestors

    def nodes_by_level(self):
        levels = [[self]]
        for child in self.children:
            child_levels = child.nodes_by_level()
            for i,child_level in enumerate(child_levels):
                if len(levels) < i+2:
                    levels.append([])
                levels[i+1].extend(child_level)
        return levels

    def plotting_representation(self):
        total_width = sum([len(x.samples) for x in self.children])
        child_proportions = []
        for child in self.children:
            child_proportions.append([float(len(child.samples)) / float(total_width),])
            child_proportions[-1].append(child.plotting_representation())
        # print(child_proportions)
        return child_proportions

    def feature_levels(self):
        feature_levels = []
        for child in self.children:
            feature_levels.extend(child.feature_levels())
        feature_levels.append((self.feature,self.level))
        return feature_levels

    def l2_fit(self,truth_dictionary):
        l2_sum = 0.0
        samples = self.sample_mask()
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

    def depth(self,d=0):
        for child in self.children:
            d = max(child.depth(d+1),d)
        return d

    def feature_mask(self, truth_dictionary=None):
        if truth_dictionary is None:
            truth_dictionary = self.forest.truth_dictionary
        feature_mask = np.zeros(len(truth_dictionary.feature_dictionary),dtype=bool)
        for feature in self.features:
            feature_mask[truth_dictionary.feature_dictionary[feature]] = True
        return feature_mask

    def sample_mask(self, truth_dictionary=None):
        if truth_dictionary is None:
            truth_dictionary = self.forest.truth_dictionary
        sample_mask = np.zeros(len(truth_dictionary.sample_dictionary),dtype=bool)
        for sample in self.samples:
            sample_mask[truth_dictionary.sample_dictionary[sample]] = True
        # if np.sum(sample_mask) == 0:
        #     print(self.samples)
        #     print(self.prerequisites)
        #     raise IndexError
        return sample_mask

    def feature_sample_mask(self,truth_dictionary=None):
        return self.sample_mask(truth_dictionary),self.feature_mask(truth_dictionary)

    def node_counts(self,counts=None,truth_dictionary=None):
        if counts is None:
            counts = self.forest.counts
        return counts[self.sample_mask()].T[self.feature_mask()].T

    def sorted_node_counts(self):
        sample_mask = self.sample_mask()
        feature_mask = self.feature_mask()
        node_counts = self.forest.counts[sample_mask].T[feature_mask].T
        try:
            sort_feature_index = self.forest.truth_dictionary.feature_dictionary[self.feature]
            sort_order = np.argsort(self.forest.counts[:,sort_feature_index][sample_mask])
        except:
            sort_order = np.arange(node_counts.shape[0])

        return node_counts[sort_order]

    def total_feature_counts(self):
        sample_mask = self.sample_mask()
        counts = self.forest.counts[sample_mask]
        return counts

    def total_feature_medians(self):
        counts = self.total_feature_counts()
        medians = np.median(counts,axis=0)
        return medians

    def ordered_node_counts(self,counts=None,truth_dictionary=None):
        if counts is None:
            counts = self.forest.counts
        if truth_dictionary is None:
            truth_dictionary = self.forest.truth_dictionary
        node_counts = np.zeros((len(self.samples),len(self.features)))
        for i,sample in enumerate(self.samples):
            for j,feature in enumerate(self.features):
                node_counts[i,j] = truth_dictionary.look(sample,feature)
        return node_counts

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

    def ranked_feature_gain(self):

        gain_rankings = np.argsort(self.local_gains)
        sorted_gains = np.array(self.local_gains)[gain_rankings]
        sorted_features = np.array(self.features)[gain_rankings]
        sorted_dispersions = np.array(self.dispersions)[gain_rankings]
        sorted_original_dispersions = np.array(self.tree.root.dispersions)[gain_rankings]

        return sorted_features,sorted_gains,sorted_dispersions,sorted_original_dispersions

    def ranked_error_vs_root(self):
        error_change = self.total_error_vs_root()
        gain_rankings = np.argsort(error_change)
        sorted_gains = np.array(error_change[gain_rankings])
        sorted_features = self.forest.features[gain_rankings]
        sorted_error_change = error_change[gain_rankings]
        return sorted_features,sorted_gains

    def aborting_sample_descent(self,sample):
        if self.feature is not None:
            if self.feature in sample:
                if sample[self.feature] <= self.split:
                    return self.children[0].aborting_sample_descent(sample)
                else:
                    return self.children[1].aborting_sample_descent(sample)
            else:
                return []
        else:
            return [self,]

    def predict_dictionary(self):
        return {x:y for x,y in zip(self.features,self.medians)}

    def predict_weighted_dictionary(self):
        return {x:(y,z) for x,y,z in zip(self.features,self.medians,self.weights)}

    def local_gain_dictionary(self):
        return {x:y for x,y in zip(self.features,self.local_gains)}

    def absolute_gain_dictionary(self):
        return {x:y for x,y in zip(self.features,self.absolute_gains)}

    def compare_to_sample(self,sample):
        differences = []
        for i,feature in enumerate(self.features):
            if feature in sample:
                differences.append(sample[feature] - self.medians[i])
        return differences

    def tree_path_vector(self):
        return np.array([0 if prerequisite[2] == '<' else 1 for prerequisite in self.prerequisites])

    def leaf_distances(self):
        leaves = self.leaves()
        distances = np.zeros((len(leaves),len(leaves)))
        for i in range(len(leaves)):
            l1v = leaves[i].tree_path_vector()
            for j in range(i,len(leaves)):
                l2v = leaves[j].tree_path_vector()
                distance = leaves[i].level + leaves[j].level
                for l1,l2 in zip(l1v,l2v):
                    if l1 == l2:
                        distance -= 2
                    else:
                        break
                distances[i,j] = distance
                distances[j,i] = distance
        return distances

    def set_cluster(self,cluster):
        self.cluster = cluster
        if self.parent is not None:
            self.parent.add_child_cluster(cluster,self.lr)

    def add_child_cluster(self,cluster,lr):
        self.child_clusters[lr].append(cluster)
        if self.parent is not None:
            self.parent.add_child_cluster(cluster,self.lr)

    def find_cluster_divergence(self,cluster):
        if self.parent is not None:
            if cluster in self.parent.child_clusters[0] or cluster in self.parent.child_clusters[1]:
                return self.parent
            else:
                return self.parent.find_cluster_divergence(cluster)
        else:
            return self


    def total_error_vs_root(self):
        own_medians = self.total_feature_medians()
        root_medians = self.root().total_feature_medians()
        counts = self.total_feature_counts()
        own_error = np.zeros(len(own_medians))
        root_error = np.zeros(len(root_medians))
        for i in range(counts.shape[0]):
            own_error += np.power((counts[i] - own_medians),2)
            root_error += np.power((counts[i] - root_medians),2)
        return own_error,root_error

    # def cluster_distances(self):

    def total_error_vs_parent(self):
        counts = self.total_feature_counts()
        own_medians = self.total_feature_medians()
        parent_medians = None
        if self.parent is not None:
            parent_medians = self.parent.total_feature_medians()
        else:
            parent_medians = self.total_feature_medians()
        own_error = np.zeros(len(own_medians))
        parent_error = np.zeros(len(parent_medians))
        for i in range(counts.shape[0]):
            own_error += np.power((counts[i] - own_medians),2)
            parent_error += np.power((counts[i] - parent_medians),2)
        return own_error,parent_error

    def cluster_divergence_encoding(self):
        clusters = [c.id for c in self.forest.leaf_clusters]
        divergence = np.zeros((len(clusters),len(clusters)))
        for lc in self.child_clusters[0]:
            lci = clusters.index(lc)
            for rc in self.child_clusters[1]:
                rci = clusters.index(rc)
                divergence[lci,rci] = 1
        return divergence

    def sample_divergence_encoding(self):
        samples = len(self.forest.samples)
        divergence = np.zeros((samples,samples,2))
        child_samples = reduce(lambda x,y: x+y.samples, self.children,[])
        for l,s1 in enumerate(child_samples):
            s1i = self.forest.truth_dictionary.sample_dictionary[s1]
            for s2 in child_samples[l:]:
                s2i = self.forest.truth_dictionary.sample_dictionary[s2]
                divergence[s1i,s2i,0] = 1
                divergence[s2i,s1i,0] = 1
        for child in self.children[:1]:
            child_samples = child.samples
            sister_samples = child.sister().samples
            for s1 in child_samples:
                s1i = self.forest.truth_dictionary.sample_dictionary[s1]
                for s2 in sister_samples:
                    s2i = self.forest.truth_dictionary.sample_dictionary[s2]
                    divergence[s1i,s2i,1] = 1
                    divergence[s2i,s1i,1] = 1
        return divergence

    def lr_encoding_vectors(self):
        left = np.zeros(len(self.forest.samples),dtype=bool)
        right = np.zeros(len(self.forest.samples),dtype=bool)
        child_masks = [left,right]
        for i,child in enumerate(self.children):
            child_masks[i] = child.sample_mask()
        return child_masks

    # def make_fat_node(self,samples,fat_tree):
    #
    #     fat_node = Node.null()
    #     fat_samples = []
    #     for sample in samples:
    #         si = self.forest.truth_dictionary.sample_dictionary[sample]
    #         prerequisite_flag = True
    #         for prerequisite in self.prerequisites:
    #             feature = prerequisite[0]
    #             fi = self.forest.truth_dictionary.feature_dictionary[feature]
    #             split = float(prerequisite[1])
    #             if prerequisite[2] == '>':
    #                 if truth_dictionary[feature] < split:
    #                     prerequisite_flag = False
    #             else if prerequisite[2] == '<':
    #                 if sample[feature] > split:
    #                     prerequisite_flag = False
    #         if prerequisite_flag:
    #             fat_samples.append(sample)
    #
    #     fat_node.tree = fat_tree
    #     fat_node.forest = self.forest
    #     fat_node.lr = self.lr
    #     fat_node.level = self.level
    #     fat_node.feature = self.feature['feature']
    #     fat_node.split = self.split['split']
    #
    #     fat_node.features = self.forest.output_features
    #     fat_node.samples =
    #     fat_node.medians = self.node_json['medians']
    #     fat_node.dispersions = self.node_json['dispersions']
    #     fat_node.weights = self.np.ones(len(self.features),dtype=float)
    #     fat_node.local_gains = self.node_json['local_gains']
    #     fat_node.absolute_gains = self.node_json['absolute_gains']
    #     fat_node.children = self.[]
    #     fat_node.child_clusters = self.([],[])
    #     fat_node.prerequisites = self.prerequisites
    #     if len(node_json['children']) > 0:
    #         self.children.append(Node(node_json['children'][0],self.tree,self.forest,parent=self,lr=0,prerequisites = prerequisites + [(self.feature,self.split,'<')],level=level+1))
    #         self.children.append(Node(node_json['children'][1],self.tree,self.forest,parent=self,lr=1,prerequisites = prerequisites + [(self.feature,self.split,'>')],level=level+1))

class Tree:

    def __init__(self, tree_json, forest):
        self.root = Node(tree_json, self, forest)
        self.forest = forest

    def nodes(self,root=True):
        nodes = []
        nodes.extend(self.root.nodes())
        if root:
            nodes.append(self.root)
        return nodes

    def leaves(self):
        leaves = self.root.leaves()
        if len(leaves) < 1:
            leaves.append(self.root)
        return leaves

    def stems(self):
        stems = self.root.stems()
        return stems

    def level(self,target):
        level_nodes = []
        for node in self.nodes():
            if node.level == target:
                level_nodes.append(node)
        return level_nodes

    def descend(self,level):
        return self.root.descend(level)

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
        bars = []
        levels = self.root.nodes_by_level()
        jump = height / len(levels)
        for i,level in enumerate(levels):
            level_samples = sum([len(node.samples) for node in level])
            next_level_samples = 0
            if i < (len(levels)-1):
                next_level_samples = sum([len(node.samples) for node in levels[i+1]])
            consumed_width = 0
            next_consumed_width = 0
            for j,node in enumerate(level):
                sample_weight = float(len(node.samples)) / float(level_samples)
                half_width = (width * sample_weight)/2
                center = consumed_width + half_width
                consumed_width = consumed_width + (half_width * 2)
                coordinates.append((i*jump,center))
                if i < (len(levels)-1):
                    for child in node.children:
                        child_sample_weight = float(len(child.samples)) / float(next_level_samples)
                        child_half_width = (width * child_sample_weight)/2
                        child_center = next_consumed_width + child_half_width
                        next_consumed_width = next_consumed_width + (child_half_width * 2)
                        connectivities.append(([i*jump,(i+1)*jump],[center,child_center]))
        coordinates = np.array(coordinates)
        plt.figure()
        plt.scatter(coordinates[:,0],coordinates[:,1],s=1)
        for connection in connectivities:
            plt.plot(connection[0],connection[1])
        plt.show()

        # return coordinates,connectivities

    def recursive_plotting_repesentation(self,axes,height=None,height_step=None,representation=None,limits=None):
        if limits is None:
            limits = axes.get_xlim()
        current_position = limits[0]
        width = float(limits[1] - limits[0])
        center = (limits[1] + limits[0]) / 2
        if representation is None:
            representation = self.root.plotting_representation()
            print(representation)
        if height_step is None or height is None:
            depth = self.root.depth()
            height_limits = axes.get_ylim()
            height = height_limits[1]
            height_step = -1 * (height_limits[1] - height_limits[0]) / depth
        # print(representation)
        for i,current_representation in enumerate(representation):
            width_proportion = current_representation[0]
            children = current_representation[1]
            node_start = current_position
            node_width = width_proportion * width
            padding = node_width * .05
            node_width = node_width - padding
            node_center = (node_width/2) + current_position
            node_height = height + height_step
            node_end = (node_width) + current_position
            current_position = node_end + padding

            color = ['r','b'][(i%2)]

            axes.plot([center,node_center],[height,node_height],c=color)
            # axes.plot([node_center],[node_height])
            axes.plot([node_start,node_end],[node_height,node_height],c=color)

            self.recursive_plotting_repesentation(axes,height=node_height,height_step=height_step,representation=children,limits=(node_start,node_end))

    def plot(self):
        fig = plt.figure(figsize=(10,20))
        ax = fig.add_subplot(111)
        self.recursive_plotting_repesentation(ax)
        fig.show()

    def tree_movie_frame(self,location,level=0,sorted=True,previous_frame=None,split_lines=True):
        descent_nodes = self.descend(level)
        total_samples = sum([len(node.samples) for node in descent_nodes])
        heatmap = np.zeros((total_samples,len(self.root.features)))
        node_splits = []
        running_samples = 0
        for node in descent_nodes:
            if sorted:
                node_counts = node.sorted_node_counts()
            else:
                node_counts = node.node_counts()
            node_samples = node_counts.shape[0]
            heatmap[running_samples:running_samples+node_samples] = node_counts
            running_samples += node_samples
            node_splits.append(running_samples)
        plt.figure(figsize=(10,10))
        if previous_frame is None:
            plt.imshow(heatmap,aspect='auto')
        else:
            plt.imshow(previous_frame,aspect='auto')
        if split_lines:
            for split in node_splits[:-1]:
                plt.plot([0,len(self.root.features)-1],[split,split],color='w')
        plt.savefig(location)
        return heatmap

    def tree_movie(self,location):
        max_depth = max([leaf.level for leaf in self.leaves()])
        previous_frame = None
        for i in range(max_depth):
            self.tree_movie_frame(location+"."+str(i)+".a.png",level=i,sorted=False,previous_frame=previous_frame)
            previous_frame = self.tree_movie_frame(location+"."+str(i)+".b.png",level=i,sorted=True)
        self.tree_movie_frame(location+"."+str(i+1)+".b.png",level=i,sorted=True,split_lines=False)

    def summary(self, verbose=True):
        nodes = len(self.nodes)
        leaves = len(self.leaves)
        if verbose:
            print("Nodes: {}".format(nodes))
            print("Leaves: {}".format(leaves))

    def aborting_sample_descent(self,sample):
        return self.root.aborting_sample_descent(sample)

    def plot_leaf_counts(self):
        leaves = self.leaves()
        total_samples = sum([len(x.samples) for x in leaves])
        heatmap = np.zeros((total_samples,len(self.root.features)))
        running_samples = 0
        for leaf in leaves:
            leaf_counts = leaf.node_counts()
            leaf_samples = leaf_counts.shape[0]
            heatmap[running_samples:running_samples+leaf_samples] = leaf_counts
            running_samples += leaf_samples

        ordering = dendrogram(linkage(heatmap.T),no_plot=True)['leaves']
        heatmap = heatmap.T[ordering].T
        plt.figure()
        im = plt.imshow(heatmap,aspect='auto')
        plt.colorbar()
        plt.show()

        return heatmap
    # def cluster_distances(self):
    #     for leaf in self.leaves():


class Forest:

    def __init__(self,trees,input,output,input_features=None,output_features=None,samples=None,split_labels=None):
        if input_features is None:
            input_features = [str(i) for i in range(input.shape[1])]
        if output_features is None:
            output_features = [str(i) for i in range(output.shape[1])]
        if samples is None:
            samples = [str(i) for i in range(input.shape[0])]
        self.truth_dictionary = TruthDictionary(output,output_features,samples)

        self.input = input
        self.output = output
        self.samples = samples

        self.input_features = input_features
        self.output_features = output_features

        self.input_dim = input.shape
        self.output_dim = output.shape

        self.split_labels = split_clusters

        self.trees = list(map(lambda x: Tree(x,self),trees))

        for i,node in enumerate(self.nodes()):
            node.index = i

    def test_forest(roots,inputs,outputs,samples=None):
        test_forest = Forest([],inputs,outputs,samples)
        test_trees = [Tree.test_tree(root,test_forest) for root in roots]
        test_forest.trees = test_trees

    def backup(self,location):
        with open(location,mode='bw') as f:
            pickle.dump(self,f)

    def reconstitute(location):
        with open(location,mode='br') as f:
            return pickle.load(f)

    def nodes(self,root=True):
        nodes = []
        for tree in self.trees:
            nodes.extend(tree.nodes(root=root))
        return nodes

    def leaves(self):
        leaves = []
        for tree in self.trees:
            leaves.extend(tree.leaves())
        return leaves

    def stems(self):
        stems = []
        for tree in self.trees:
            stems.extend(tree.stems())
        return stems

    def roots(self):
        return [tree.root for tree in self.trees]

    def feature_leaves(self,feature):
        leaves = []
        for tree in self.trees:
            if feature in tree.root.features:
                leaves.extend(tree.leaves)
        return leaves

    def sample_leaves(self,sample):
        leaves = self.leaves()
        encoding = self.node_sample_encoding(leaves)
        leaf_indecies = np.arange(len(leaves))[encoding[sample]]
        sample_leaves = [leaves[i] for i in leaf_indecies]
        return sample_leaves

    def leaves_of_samples(self,samples):
        sample_leaves_total = []
        leaves = self.leaves()
        encoding = self.node_sample_encoding(leaves)
        for sample in samples:
            leaf_indecies = np.arange(len(leaves))[encoding[sample]]
            sample_leaves = [leaves[i] for i in leaf_indecies]
            sample_leaves_total.extend(sample_leaves)
        return sample_leaves_total

    def node_sample_encoding(self,nodes):
        encoding = np.zeros((len(self.samples),len(nodes)),dtype=bool)
        sd = self.truth_dictionary.sample_dictionary
        for i,node in enumerate(nodes):
            for sample in node.samples:
                encoding[sd[sample],i] = True
        return encoding

    def node_feature_encoding(self,nodes):

        fd = self.truth_dictionary.feature_dictionary
        encoding = np.zeros((len(nodes),len(fd)),dtype=bool)
        for i,node in enumerate(nodes):
            for feature in node.features:
                encoding[i,fd[feature]] = True
        return encoding

    def absolute_gain_matrix(self,nodes):
        gains = np.zeros((len(self.features),len(nodes)))
        fd = self.truth_dictionary.feature_dictionary
        for i,node in enumerate(nodes):
            if node.absolute_gains is not None:
                for feature,gain in zip(node.features,node.absolute_gains):
                    gains[fd[feature],i] = gain
        return gains

    def total_absolute_error_matrix(self,nodes):
        all_root_error = np.zeros((len(self.output_features),len(nodes)))
        all_leaf_error = np.zeros((len(self.output_features),len(nodes)))
        for i,node in enumerate(nodes):
            leaf_error,root_error = node.total_error_vs_root()
            all_leaf_error[:,i] = leaf_error
            all_root_error[:,i] = root_error
        return all_root_error-all_leaf_error,all_leaf_error,all_root_error


    def total_local_error_matrix(self,nodes):
        all_parent_error = np.zeros((len(self.output_features),len(nodes)))
        all_node_error = np.zeros((len(self.output_features),len(nodes)))
        for i,node in enumerate(nodes):
            node_error,parent_error = node.total_error_vs_parent()
            all_node_error[:,i] = node_error
            all_parent_error[:,i] = parent_error
        return all_parent_error-all_node_error,all_node_error,all_parent_error


    def local_gain_matrix(self,nodes):
        gains = np.zeros((len(self.output_features),len(nodes)))
        fd = self.truth_dictionary.feature_dictionary
        for i,node in enumerate(nodes):
            for feature,gain in zip(node.features,node.local_gains):
                gains[fd[feature],i] = gain
        return gains

    def level(self,target):
        level = []
        for tree in self.trees:
            level.extend(tree.level(target))
        return level

    def load(location, prefix="/run", ifh="/run.ifh",ofh='run.ofh',clusters='run.clusters',input="input.counts",output="output.counts"):

        combined_tree_files = sorted(glob.glob(location + prefix + "*.compact"))

        print(combined_tree_files)

        raw_forest = []

        for tree_file in combined_tree_files:
            raw_forest.append(json.load(open(tree_file.strip())))

        input = np.loadtxt(location+input)
        output = np.loadtxt(location+output)
        ifh = np.loadtxt(location+ifh,dtype=str)
        ofh = np.loadtxt(location+ofh,dtype=str)

        clusters = np.loadtxt(location+clusters)

        first_forest = Forest(raw_forest[1:],input_features=ifh,output_features=ofh,input=input,output=output,clusters=clusters)

        first_forest.prototype = Tree(raw_forest[0],first_forest)

        sample_encoding = first_forest.node_sample_encoding(first_forest.leaves())

        if np.sum(np.sum(sample_encoding,axis=1) == 0) > 0:
            print("WARNING, UNREPRESENTED SAMPLES")

        feature_encoding = first_forest.node_feature_encoding(first_forest.leaves())

        if np.sum(np.sum(feature_encoding,axis=0) == 0) > 0:
            print("WARNING, UNREPRESENTED FEATURES")

        return first_forest

    def abort_sample_leaves(self,sample):

        leaves = []

        for tree in self.trees:
            leaves.extend(tree.aborting_sample_descent(sample))

        return leaves

    def raw_predict_nodes(self,nodes):

        consolidated_predictions = {}

        for node in nodes:
            node_predictions = node.predict_dictionary()
            for feature,prediction in node_predictions.items():
                if feature not in consolidated_predictions:
                    consolidated_predictions[feature] = []
                consolidated_predictions[feature].append(prediction)

        return consolidated_predictions

    def weighted_predict_nodes(self,nodes):

        consolidated_predictions = {}

        for node in nodes:
            node_predictions = node.predict_weighted_dictionary()
            for feature,prediction in node_predictions.items():
                if feature not in consolidated_predictions:
                    consolidated_predictions[feature] = ([],[])
                consolidated_predictions[feature][0].append(prediction[0])
                consolidated_predictions[feature][1].append(prediction[1])

        return consolidated_predictions

    def raw_prediction_matrix(self,nodes):
        fd = self.truth_dictionary.feature_dictionary
        predictions = np.zeros((len(nodes),len(fd)))
        for i,node in enumerate(nodes):
            for feature,feature_prediction in zip(node.features,node.medians):
                predictions[i,fd[feature]] = feature_prediction
        return predictions

    def feature_weight_matrix(self,nodes):
        fd = self.truth_dictionary.feature_dictionary
        weights = np.zeros((len(nodes),len(fd)))
        for i,node in enumerate(nodes):
            for feature,feature_weight in zip(node.features,node.weights):
                weights[i,fd[feature]] = feature_weight
        return weights

    def nodes_predict_feature(self,nodes,feature):
        predictions = []
        for node in nodes:
            try:
                node_feature_index = node.features.index(feature)
                predictions.append(node.medians[node_feature_index])
            except ValueError:
                predictions.append(None)
        return predictions

    def node_feature_index_table(self,nodes):
        fd = self.truth_dictionary.feature_dictionary
        table = -1 * np.ones((len(nodes),len(fd)),dtype=int)
        for node_index,node in enumerate(nodes):
            for node_feature_index,node_feature in enumerate(node.features):
                forest_feature_index = fd[node_feature]
                table[node_index,forest_feature_index] = node_feature_index
        return table

    def node_predict_assisted(self,nodes,feature_indecies):
        if -1 in weight_indecies:
            print("Tried predict a feature a node doesn't have!")
            raise ValueError
        predictions = np.zeros(len(nodes))
        for node_index,node,feature_index in zip(range(len(nodes)),nodes,feature_indecies):
            predictions[node_index] = node.medians[feature_index]
        return predictions

    def set_feature_weights(self,nodes,weights,feature):
        for node,weight in zip(nodes,weights):
            feature_index = node.features.index(feature)
            node.weights[feature_index] = weight

    def set_feature_weights_assisted(self,nodes,weight_indecies,weights):
        if -1 in weight_indecies:
            print("Tried to set a weight for a feature a node doesn't have!")
            print(weight_indecies)
            raise ValueError
        for node,weight_index,weight in zip(nodes,weight_indecies,weights):
            node.weights[weight_index] = weight


    def weigh_leaves(self,positive=True):

        forest_leaves = self.leaves()
        leaf_sample_encoding = self.node_sample_encoding(forest_leaves)
        # print("SAMPLE ENCODING")
        # print(list(leaf_sample_encoding))
        leaf_feature_encoding = self.node_feature_encoding(forest_leaves)
        # print("FEATURE_ENCODING")
        # print(list(leaf_feature_encoding))
        raw_prediction_matrix = self.raw_prediction_matrix(forest_leaves)
        leaf_feature_index_table = self.node_feature_index_table(forest_leaves)

        # local_generator = self.linear_problem_generator(self.features,forest_leaves,leaf_sample_encoding,leaf_feature_encoding,raw_prediction_matrix,leaf_feature_index_table,negative_weights)

        # constant_repeater = map(lambda x: (x[0],x[1],forest_leaves,leaf_sample_encoding,leaf_feature_encoding,raw_prediction_matrix,leaf_feature_index_table,negative_weights), enumerate(self.features))
        #
        #
        for feature in self.features:
            feature_index = self.truth_dictionary.feature_dictionary[feature]
            # leaf_feature_index = leaf_feature_encoding
            feature_leaf_indecies = np.arange(len(forest_leaves))[leaf_feature_encoding[:,feature_index]]
            feature_leaves = [forest_leaves[x] for x in feature_leaf_indecies]
            feature_leaf_sample_encoding = leaf_sample_encoding.T[feature_leaf_indecies].T.astype(dtype=float)
            leaf_predictions = raw_prediction_matrix[feature_leaf_indecies][:,feature_index]
            for i,prediction in enumerate(leaf_predictions):
                feature_leaf_sample_encoding[:,i] *= prediction
            leaf_feature_indecies = leaf_feature_index_table[feature_leaf_indecies][:,feature_index]

            truth = self.output[:,feature_index]

            weights = Ridge(alpha=5).fit(feature_leaf_sample_encoding,truth).coef_
            # weights = NMF().fit()


            if positive:
                weights[weights < 0] = 0

            sum = np.sum(weights)

            if sum > .01:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(weights.shape)

            # feature,weights = self.solve_linear((feature_index,feature,feature_leaf_sample_encoding,truth,positive))

            # print("SOLVING WEIGHTS")
            # print(feature)
            # print(list(feature_leaf_indecies))
            # print(list(leaf_feature_encoding[:,feature_index]))
            # print(leaf_feature_indecies)
            # print(leaf_feature_)
            # print(list(feature_leaf_sample_encoding))
            # print(feature_leaf_sample_encoding.shape)
            # print(leaf_predictions)
            # print(truth)
            # print(weights)

            self.set_feature_weights(feature_leaves,weights,feature)

            # linear_problems.append((len(linear_problems),feature,feature_leaf_sample_encoding,truth))

        # print("Linear problems formulated")
        #
        # for feature,weights in Pool().imap_unordered(self.solve_linear,local_generator,chunksize=1):
        #     print("Solved weights:" + feature)
        #     feature_index = self.truth_dictionary.feature_dictionary[feature]
        #     feature_leaf_indecies = np.arange(len(forest_leaves))[leaf_feature_encoding[:,feature_index]]
        #     feature_leaves = [forest_leaves[x] for x in feature_leaf_indecies]
        #     self.set_feature_weights(feature_leaves,weights,feature)

        plt.figure()
        plt.hist(self.feature_weight_matrix(forest_leaves).flatten(),bins=50,log=True)
        plt.show()

    def abort_predict_sample(self,sample):

        fd = self.truth_dictionary.feature_dictionary

        leaves = self.abort_sample_leaves(sample)

        consolidated_predictions = self.raw_predict_nodes(leaves)

        single_prediction = np.zeros(len(fd))

        for feature,predictions in consolidated_predictions.items():

            single_prediction[fd[feature]] = np.mean(predictions)

        return single_prediction

    def abort_weighted_predict_sample(self,sample):

        leaves = self.abort_sample_leaves(sample)

        return self.weighted_node_vector_prediction(leaves)

    def weighted_node_vector_prediction(self,nodes):
        fd = self.truth_dictionary.feature_dictionary
        consolidated_predictions = self.weighted_predict_nodes(nodes)
        single_prediction = np.zeros(len(fd))

        for feature,entry in consolidated_predictions.items():
            predictions,weights = entry
            prediction = sum([p * w for p,w in zip(predictions,weights)]) / sum(weights)
            if np.isfinite(prediction):
                single_prediction[fd[feature]] = prediction
            else:
                single_prediction[fd[feature]] = 0

        return single_prediction

    def raw_node_prediction(self,nodes):
        fd = self.truth_dictionary.feature_dictionary
        single_prediction = np.zeros(len(fd))
        raw_predictions = self.raw_predict_nodes(nodes)
        for feature,predictions in raw_predictions.items():
            single_prediction[fd[feature]] = np.mean(predictions)

        return predictions

    def predict_matrix(self,matrix,features=None,samples=None,weighted=True):

        if features is None:
            features = self.features
        if samples is None:
            samples = self.samples

        predictions = np.zeros((len(matrix),len(self.features)))

        for i,row in enumerate(matrix):
            sample = {feature:value for feature,value in zip(features,row)}
            if weighted:
                predictions[i] = self.abort_weighted_predict_sample(sample)
            else:
                predictions[i] = self.abort_predict_sample(sample)

        return predictions

    def predict_vector_leaves(self,vector,features=None):
        if features is None:
            features = self.features
        sample = {feature:value for feature,value in zip(features,vector)}
        return self.abort_sample_leaves(sample)

    def cluster_samples_simple(self,override=False,*args,**kwargs):

        counts = self.output

        if hasattr(self,'sample_clusters') and not override:
            print("Clustering has already been done")
            return self.sample_labels
        else:
            self.sample_labels = np.array(sdg.fit_predict(counts,*args,**kwargs))

        cluster_set = set(self.sample_labels)
        clusters = []
        for cluster in cluster_set:
            cells = np.arange(len(self.sample_labels))[self.sample_labels == cluster]
            clusters.append(SampleCluster(self,cells,cluster))

        self.sample_clusters = clusters

        return self.sample_labels

    def cluster_samples_encoding(self,override=False,*args,**kwargs):

        leaves = self.leaves()
        encoding = self.node_sample_encoding(leaves)

        if hasattr(self,'sample_clusters') and not override:
            print("Clustering has already been done")
        else:
            self.sample_labels = np.array(sdg.fit_predict(encoding,*args,**kwargs))

        cluster_set = set(self.sample_labels)
        clusters = []
        for cluster in cluster_set:
            cells = np.arange(len(self.sample_labels))[self.sample_labels == cluster]
            clusters.append(SampleCluster(self,cells,cluster))

        self.sample_clusters = clusters

        return self.sample_labels

    def cluster_samples_coocurrence(self,override=False,*args,**kwargs):
        leaves = self.leaves()
        encoding = self.node_sample_encoding(leaves)
        coocurrence = coocurrence_matrix(encoding)

        if hasattr(self,'sample_clusters') and not override:
            print("Clustering has already been done")
            return self.sample_labels
        else:
            self.sample_labels = np.array(sdg.fit_predict(coocurrence,precomputed=True,*args,**kwargs))

        cluster_set = set(self.sample_labels)
        clusters = []
        for cluster in cluster_set:
            cells = np.arange(len(self.sample_labels))[self.sample_labels == cluster]
            clusters.append(SampleCluster(self,cells,cluster))

        self.sample_clusters = clusters

        return self.sample_labels


    def cluster_leaf_samples(self,type="fuzzy",override=False,*args,**kwargs):

        leaves = self.leaves()
        encoding = self.node_sample_encoding(leaves).T

        if hasattr(self,'leaf_clusters') and not override:
            print("Clustering has already been done")
            return self.leaf_labels
        else:
            self.leaf_labels = np.array(sdg.fit_predict(encoding,*args,**kwargs))

        cluster_set = set(self.leaf_labels)

        clusters = []

        for cluster in cluster_set:
            leaf_index = np.arange(len(self.leaf_labels))[self.leaf_labels == cluster]
            clusters.append(NodeCluster(self,[leaves[i] for i in leaf_index],cluster))

        self.leaf_clusters = clusters
        for leaf,label in zip(leaves,self.leaf_labels):
            leaf.set_cluster(label)

        return self.leaf_labels

    def cluster_leaf_gains(self,type="fuzzy",override=False,*args,**kwargs):

        leaves = self.leaves()
        gains = self.absolute_gain_matrix(leaves)

        if hasattr(self,'leaf_clusters') and not override:
            print("Clustering has already been done")
            return self.leaf_labels
        else:
            self.leaf_labels = np.array(sdg.fit_predict(gains,*args,**kwargs))

        cluster_set = set(self.leaf_labels)

        clusters = []

        for cluster in cluster_set:
            leaf_index = np.arange(len(self.leaf_labels))[self.leaf_labels == cluster]
            clusters.append(NodeCluster(self,[leaves[i] for i in leaf_index],cluster))

        self.leaf_clusters = clusters
        for leaf,label in zip(leaves,self.leaf_labels):
            leaf.set_cluster(label)

        return self.leaf_labels

    def cluster_leaf_predictions(self,override=False,*args,**kwargs):

        leaves = self.leaves()
        predictions = self.raw_prediction_matrix(leaves)

        if hasattr(self,'leaf_clusters') and not override:
            print("Clustering has already been done")
            return self.leaf_labels
        else:
            self.leaf_labels = np.array(sdg.fit_predict(predictions,*args,**kwargs))

        cluster_set = set(self.leaf_labels)

        clusters = []

        for cluster in cluster_set:
            leaf_index = np.arange(len(self.leaf_labels))[self.leaf_labels == cluster]
            clusters.append(NodeCluster(self,[leaves[i] for i in leaf_index],cluster))

        self.leaf_clusters = clusters
        for leaf,label in zip(leaves,self.leaf_labels):
            leaf.set_cluster(label)

        return self.leaf_labels

    def node_total_predictions(self,nodes):
        meta_predictions = np.zeros((len(nodes),self.dim[1]))
        for i,node in enumerate(nodes):
            meta_predictions[i] = node.total_feature_medians()
        return meta_predictions

    def cluster_leaf_total_predictions(self,override=False,*args,**kwargs):

        if hasattr(self,'leaf_clusters') and not override:
            print("Clustering has already been done")
            return self.leaf_labels
        else:
            leaves = self.leaves()
            meta_predictions = self.node_total_predictions(leaves)
            self.leaf_labels = np.array(sdg.fit_predict(meta_predictions,*args,**kwargs))

        cluster_set = set(self.leaf_labels)

        clusters = []

        for cluster in cluster_set:
            leaf_index = np.arange(len(self.leaf_labels))[self.leaf_labels == cluster]
            clusters.append(NodeCluster(self,[leaves[i] for i in leaf_index],cluster))

        self.leaf_clusters = clusters
        for leaf,label in zip(leaves,self.leaf_labels):
            leaf.set_cluster(label)

        return self.leaf_labels

    def cluster_features(self,*args,**kwargs):
        gain_matrix = self.absolute_gain_matrix(self.leaves())
        return sdg.fit_predict(gain_matrix,*args,**kwargs)

    def plot_counts(self,no_plot=False):

        if not no_plot:
            cell_sort = dendrogram(linkage(encoding,metric='cos',method='average'),no_plot=True)['leaves']
            leaf_sort = dendrogram(linkage(encoding.T,metric='cos',method='average'),no_plot=True)['leaves']

            plt.figure(figsize=(10,10))
            plt.imshow(encoding[cell_sort].T[leaf_sort].T,cmap='binary')
            plt.show()

        return cell_sort,leaf_sort,self.ouput_counts



    def interpret_splits(self,override=False,no_plot=False,*args,**kwargs):

        nodes = self.nodes(root=False)

        gain_matrix = self.local_gain_matrix(nodes).T+1

        cluster_set = set(self.split_labels)
        clusters = []
        for cluster in cluster_set:
            split_index = np.arange(len(self.split_labels))[self.split_labels == cluster]
            clusters.append(NodeCluster(self,[nodes[i] for i in split_index],cluster))

        # split_order = np.argsort(self.split_labels)
        split_order = dendrogram(linkage(gain_matrix,metric='cos',method='average'),no_plot=True)['leaves']
        feature_order = dendrogram(linkage(gain_matrix.T+1,metric='cosine',method='average'),no_plot=True)['leaves']

        image = gain_matrix[split_order].T[feature_order].T
        neg = image < 0
        pos = image > 0
        image[neg] = -1 * np.log(np.abs(image[neg]) + 1)
        image[pos] = np.log(image[pos] + 1)

        median = np.median(image)
        range = np.max(image) - median


        plt.figure(figsize=(10,10))
        plt.imshow(image,aspect='auto',cmap='bwr',vmin=median-range,vmax=median+range)
        plt.colorbar()
        plt.show()

        self.split_clusters = clusters

        return self.split_labels,image


    def filter_cells(cells,prerequisite):
        filtered = []
        feature,split,direction = prerequisite
        if direction == '<':
            for cell in cells:
                if self.truth_dictionary.look(cell,feature) < split:
                    filtered.append(cell)
        if direction == '>':
            for cell in cells:
                if self.truth_dictionary.look(cell,feature) > split:
                    filtered.append(cell)
        return filtered

    def plot_cell_clusters(self,colorize=True):
        if not hasattr(self,'leaf_clusters'):
            print("Warning, leaf clusters not detected")
            return None
        if not hasattr(self,'sample_clusters'):
            print("Warning, cell clusters not detected")
            return None

        tc = self.tsne(no_plot=True)

        cluster_tc = np.zeros((len(self.sample_clusters),2))

        for i,cluster in enumerate(self.sample_clusters):
            cluster_cell_mask = self.sample_labels == cluster.id
            mean_coordinates = np.mean(tc[cluster_cell_mask],axis=0)
            cluster_tc[i] = mean_coordinates

        combined_coordinates = np.zeros((self.output.shape[0]+len(self.sample_clusters),2))

        combined_coordinates[0:self.output.shape[0]] = tc

        combined_coordinates[self.output.shape[0]:] = cluster_tc

        highlight = np.ones(combined_coordinates.shape[0])
        highlight[len(self.sample_labels):] = [len(cluster.samples) for cluster in self.sample_clusters]
        # for i,cluster in enumerate(self.sample_clusters):
        #
        #     highlight[self.counts.shape[0] + i:] = len(cluster.samples/10)

        combined_labels = np.zeros(self.output.shape[0]+len(self.sample_clusters))
        if colorize:
            combined_labels[0:len(self.sample_labels)] = self.sample_labels
            combined_labels[len(self.sample_labels):] = [cluster.id for cluster in self.sample_clusters]

        cluster_names = [cluster.id for cluster in self.sample_clusters]
        cluster_coordiantes = combined_coordinates[len(self.sample_labels):]

        plt.figure(figsize=(5,5))
        plt.title("TSNE-Transformed Cell Coordinates")
        plt.scatter(combined_coordinates[:,0],combined_coordinates[:,1],s=highlight,c=combined_labels,cmap='rainbow')
        for cluster,coordinates in zip(cluster_names,cluster_coordiantes):
            plt.text(*coordinates,cluster,verticalalignment='center',horizontalalignment='center')
        plt.savefig("./tmp.delete.png",dpi=500)

    # def cluster_distances(self):
    #     for tree in self.trees:
    #         pass

    def sample_cluster_coordinate_matrix(self):
        coordinates = np.zeros((len(self.sample_clusters),len(self.features)))
        for i,sample_cluster in enumerate(self.sample_clusters):
            coordinates[i] = sample_cluster.median_feature_values()
        return coordinates

    def tsne(self,no_plot=False,override=False,**kwargs):
        if not hasattr(self,'tsne_coordinates') or override:
            self.tsne_coordinates = TSNE().fit_transform(self.output)

        if not no_plot:
            plt.figure()
            plt.title("TSNE-Transformed Cell Coordinates")
            plt.scatter(self.tsne_coordinates[:,0],self.tsne_coordinates[:,1],s=.1,**kwargs)
            plt.show()

        return self.tsne_coordinates

    def tsne_encoding(self,no_plot=False,override=False,**kwargs):
        if not hasattr(self,'tsne_coordinates') or override:
            self.tsne_coordinates = TSNE().fit_transform(self.node_sample_encoding(self.leaves()))

        if not no_plot:
            plt.figure()
            plt.title("TSNE-Transformed Cell Coordinates")
            plt.scatter(self.tsne_coordinates[:,0],self.tsne_coordinates[:,1],s=.1,**kwargs)
            plt.show()

        return self.tsne_coordinates

    def average_prereq_freq_level(self,nodes):
        prereq_dict = {}
        for node in nodes:
            for level,prerequisite in enumerate(node.prerequisites):
                prereq_feature, split, prereq_sign = prerequisite
                if prereq_feature not in prereq_dict:
                    prereq_dict[prereq_feature] = [0,0]
                prereq_dict[prereq_feature][0] += level
                prereq_dict[prereq_feature][1] += 1
        for prereq_feature in prereq_dict:
            prereq_dict[prereq_feature][0] /= prereq_dict[prereq_feature][1]
        sorted_prereqs = sorted(list(prereq_dict.items()),key=lambda prereq: prereq[1][0])
        return sorted_prereqs

    def prereq_summary(self):

        leaves = self.leaves()
        prereqs = self.average_prereq_freq_level(leaves)

        # prereqs = [prereq for prereq in prereqs if prereq[0][:2] != "CG"]
        prereqs = sorted(prereqs,key=lambda prereq: prereq[1][1])[::-1]

        prereq_features = [prereq[0] for prereq in prereqs]
        prereq_levels = [prereq[1][0] for prereq in prereqs]
        prereq_frequencies = [prereq[1][1] for prereq in prereqs]

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_axes([.6,.025,.2,.95])
        ax.set_title("Prerequisite/Level Distribution")
        ax.scatter(prereq_levels[:50],np.arange(49,-1,-1),s=prereq_frequencies[:50])
        ax.set_xlabel("Average Level of Decision")
        # ax.set_xlim(max(prereq_levels[:20])*1.1,-0.01)
        ax.set_yticks(np.arange(49,-1,-1))
        ax.set_yticklabels(prereq_features[:50])
        # ax.grid(axis='y')

    def node_feature_summary(self,nodes):

        feature_counts = count_list_elements([n.feature['name'] for n in nodes])

        feature_counts = list(feature_counts.items())

        # return sorted(feature_counts)[::-1]
        return sorted(feature_counts,key=lambda x: x[1])[::-1]

    def find_node_cluster_divergence(self,c1,c2):
        c1_index = [c.id for c in self.leaf_clusters].index(c1)
        c1_leaves = self.leaf_clusters[c1_index].nodes
        divergent_nodes = []
        distances = []
        for i,leaf in enumerate(c1_leaves):
            divergence = leaf.find_cluster_divergence(c2)
            divergent_nodes.append(divergence)
            distances.append(leaf.level - divergence.level)
        average_distance = np.mean(distances)
        if np.is_nan(average_distance):
            average_distance = np.mean([l.level for l in c1_leaves])
        return average_distance,divergent_nodes,distances

    def node_cluster_distance_matrix(self):
        n = len(self.leaf_clusters)
        distances = np.zeros((n,n))
        for i in range(len(self.leaf_clusters)):
            for j in range(len(self.leaf_clusters)):
                average_distance,_,_ = self.find_node_cluster_divergence(self.leaf_clusters[i].id,self.leaf_clusters[j].id)
                distances[i,j] += average_distance
                distances[j,i] += average_distance
        return distances

    def node_divergence_encoding(self,nodes):
        encoding = np.zeros((len(nodes),len(self.leaf_clusters),len(self.leaf_clusters)))
        for i,node in enumerate(nodes):
            encoding[i] = node.cluster_divergence_encoding()
        return encoding

    def cluster_divergence(self,**kwargs):
        leaves = self.leaves()
        leaf_clusters = self.leaf_clusters
        stems = self.stems()
        # roots = [t.root for t in self.trees]
        # nodes = roots + stems
        nodes = stems
        encoding = self.node_divergence_encoding(nodes)
        flat_encoding = np.array([ed.flatten() for ed in encoding])
        # reduced = PCA(n_components=10).fit_transform(flat_encoding)
        labels = np.array(sdg.fit_predict(flat_encoding,**kwargs))
        for i,label in enumerate(labels):
            labels[i] = label + len(leaf_clusters)

        cluster_set = set(labels)

        clusters = []

        for cluster in cluster_set:
            cluster_node_index = np.arange(len(labels))[labels == cluster]
            cluster_nodes = [nodes[i] for i in cluster_node_index]
            print(np.sum(labels == cluster))
            print(len(cluster_nodes))
            clusters.append(NodeCluster(self,cluster_nodes,cluster))

        self.divergence_clusters = clusters
        self.divergence_labels = labels

        # for node,label in zip(nodes,labels):
        #     node.set_cluster(label)

        return nodes,labels,flat_encoding

    def reset_clusters(self):
        try:
            del self.sample_clusters
            del self.sample_labels
        except:
            pass
        try:
            del self.leaf_clusters
            del self.leaf_labels
        except:
            pass
        try:
            del self.split_clusters
            del self.split_labels
        except:
            pass

        for node in self.nodes():
            node.child_clusters = ([],[])
            if hasattr(node,'cluster'):
                del node.cluster

    def plot_sample_feature_split(self,gradient,plot_n=20):

        left = gradient < .5
        right = gradient >= .5

        sample_sort = np.argsort(gradient)

        left_counts = self.output[left]
        right_counts = self.output[right]

        left_mean_features = np.mean(left_counts,axis=0)
        right_mean_features = np.mean(right_counts,axis=0)

        sort_up_left = np.argsort(left_mean_features - right_mean_features)
        sort_up_right = np.argsort(right_mean_features - left_mean_features)

        features_up_left = self.output_features[sort_up_left]
        features_up_right = self.output_features[sort_up_right]

        plt.figure(figsize=(5,8))
        plt.suptitle("Divergence of Features",fontsize=20)
        ax1 = plt.subplot(211)
        ax1.imshow(self.output[sample_sort].T[sort_up_right][-plot_n:],aspect='auto')
        ax1.set_yticks(np.arange(plot_n))
        ax1.set_yticklabels(features_up_right[-plot_n:],fontsize=14)
        ax1.tick_params(axis='x',labelbottom=False)
        ax2 = plt.subplot(212)
        ax2.imshow(self.output[sample_sort].T[sort_up_left][-plot_n:],aspect='auto')
        ax2.set_yticks(np.arange(plot_n))
        ax2.set_yticklabels(features_up_left[-plot_n:],fontsize=14)
        ax2.tick_params(axis='y',labelleft=False,labelright=True)
        plt.show()


class TruthDictionary:

    def __init__(self,counts,header,samples=None):

        self.counts = counts
        self.header = header
        self.feature_dictionary = {}

        self.sample_dictionary = {}
        for i,feature in enumerate(header):
            self.feature_dictionary[feature.strip('""').strip("''")] = i
        if samples is None:
            samples = map(lambda x: str(x),range(counts.shape[0]))
        for i,sample in enumerate(samples):
            self.sample_dictionary[sample.strip("''").strip('""')] = i

    def look(self,sample,feature):
#         print(feature)
        return(self.counts[self.sample_dictionary[sample],self.feature_dictionary[feature]])

class SampleCluster:

    def __init__(self,forest,samples,id):
        self.id = id
        self.samples = samples
        self.forest = forest

    def median_feature_values(self):
        return np.median(self.forest.counts[self.samples],axis=0)

    def increased_features(self,n=50,plot=True):
        initial_medians = self.forest.weighted_node_vector_prediction([self.forest.prototype.root])
        current_medians = self.median_feature_values()

        difference = current_medians - initial_medians
        feature_order = np.argsort(difference)
        ordered_features = np.array(self.forest.features)[feature_order]
        ordered_difference = difference[feature_order]

        if plot:
            plt.figure(figsize=(10,8))
            plt.title("Upregulated Genes")
            plt.scatter(np.arange(n),ordered_difference[-n:])
            plt.xlim(0,n)
            plt.xlabel("Gene Symbol")
            plt.ylabel("Increase (LogTPM)")
            plt.xticks(np.arange(n),ordered_features[-n:],rotation=45,verticalalignment='top',horizontalalignment='right')
            plt.show()

        return ordered_features,ordered_difference
        #
        # initial_means = np.mean(self.forest.counts,axis=0)
        # current_means = np.mean(self.forest.counts[self.samples],axis=0)
        #
        # difference = current_means - initial_means
        # feature_order = np.argsort(difference)
        # ordered_features = np.array(self.forest.features)[feature_order]
        # ordered_difference = difference[feature_order]
        #
        # if plot:
        #     plt.figure()
        #     plt.title("Upregulated Genes")
        #     plt.scatter(np.arange(n),ordered_difference[-n:])
        #     plt.xlim(0,n)
        #     plt.xlabel("Gene Symbol")
        #     plt.ylabel("Frequency")
        #     plt.xticks(np.arange(n),ordered_features[-n:],rotation='vertical')
        #     plt.show()
        #
        # return ordered_features,ordered_difference


    def decreased_features(self,n=50,plot=True):
        initial_medians = self.forest.weighted_node_vector_prediction([self.forest.prototype.root])
        current_medians = self.median_feature_values()

        difference = current_medians - initial_medians
        feature_order = np.argsort(difference)
        ordered_features = np.array(self.forest.features)[feature_order]
        ordered_difference = difference[feature_order]

        if plot:
            plt.figure(figsize=(10,2))
            plt.title("Upregulated Genes")
            plt.scatter(np.arange(n),ordered_difference[:n])
            plt.xlim(0,n)
            plt.xlabel("Gene Symbol")
            plt.ylabel("Frequency")
            plt.xticks(np.arange(n),ordered_features[:n],rotation=45,verticalalignment='top',horizontalalignment='right')
            plt.show()

        return ordered_features,ordered_difference

    def leaf_encoding(self):
        leaves = self.forest.leaves()
        encoding = self.forest.node_sample_encoding(leaves)
        encoding = encoding[self.samples]
        return encoding

    def leaf_counts(self):
        encoding = self.leaf_encoding()
        return np.sum(encoding,axis=0)

    def leaf_cluster_frequency(self,plot=True):
        leaf_counts = self.leaf_counts()
        leaf_cluster_labels = self.forest.leaf_labels
        leaf_clusters = sorted(list(set(leaf_cluster_labels)))
        leaf_cluster_counts = []
        for leaf_cluster in leaf_clusters:
            cluster_mask = leaf_cluster_labels == leaf_cluster
            leaf_cluster_counts.append(np.sum(leaf_counts[cluster_mask]))
        if plot:
            plt.figure()
            plt.title(f"Distribution of Leaf Clusters in Cell Cluster {self.id}")
            plt.bar(np.arange(len(leaf_clusters)),leaf_cluster_counts,)
            plt.ylabel("Frequency")
            plt.xlabel("Leaf Cluster")
            plt.xticks(np.arange(len(leaf_clusters)),leaf_clusters)
            plt.show()

        return leaf_clusters,leaf_cluster_counts



class NodeCluster:

    def __init__(self,forest,nodes,id):
        self.id = id
        self.nodes = nodes
        self.forest = forest

    def encoding(self):
        return self.forest.node_sample_encoding(self.nodes)

    def mean_absolute_feature_gains(self):
        mean_gains = np.zeros(len(self.forest.features))
        node_gains = [node.absolute_gain_dictionary() for node in self.nodes]
        stacked_gains = stack_dictionaries(node_gains)
        print(list(stacked_gains.items())[:10])
        for i,feature in enumerate(self.forest.features):
            mean_gains[i] = np.mean(stacked_gains[feature])
        return mean_gains

    def ranked_feature_error_gain(self):
        nodes = self.nodes
        total_error_gain_matrix = self.forest.total_absolute_error_matrix(nodes)[0].T
        average_gains = np.mean(total_error_gain_matrix,axis=0)
        gain_rankings = np.argsort(average_gains)
        sorted_gains = average_gains[gain_rankings]
        sorted_features = self.forest.features[gain_rankings]
        return sorted_features,sorted_gains

    def ranked_mean_gains(self):
        mean_gains = self.mean_absolute_feature_gains()
        gain_order = np.argsort(mean_gains)
        sorted_features = np.array(self.forest.features)[gain_order]
        sorted_gains = mean_gains[gain_order]

        plt.figure(figsize=(10,2))
        plt.title("Features Gaining Information")
        plt.scatter(np.arange(50),sorted_gains[-50:])
        plt.xlim(0,50)
        plt.xlabel("Gene Symbol")
        plt.ylabel("Gain")
        plt.xticks(np.arange(50),sorted_features[-50:],rotation='vertical')
        plt.show()

        return sorted_features,sorted_gains

    def biological_cluster_summary(self):
        levels = [node.level for node in self.nodes]

        fig = plt.figure(figsize=(20,10))

        # fig.suptitle(f"Summary of Leaf Cluster {self.id}")

        ax_levels = fig.add_axes([.875,.025,.1,.2])
        ax_levels.set_title("Leaf Levels")
        ax_levels.set_ylabel("Frequency")
        ax_levels.hist(levels)

        leaf_size = [len(node.samples) for node in self.nodes]

        ax_leaf_size = fig.add_axes([.875,.275,.1,.2])
        ax_leaf_size.set_title("Leaf Sizes")
        ax_leaf_size.hist(leaf_size)
        ax_leaf_size.set_ylabel("Frequency")

        ordered_features,ordered_difference = self.increased_features(plot=False)

        range = max(np.abs(np.min(ordered_difference.flatten())),np.max(ordered_difference)) * 1.1

        ax_downregulated = fig.add_axes([.025,.775,.2,.2])
        ax_downregulated.set_title("Downregulated Genes",fontsize=20)
        ax_downregulated.set_ylabel("Mean downregulation (Log TPM)",fontsize=10)
        ax_downregulated.bar(np.arange(10),ordered_difference[:10])
        ax_downregulated.set_ylim(-range,range)
        ax_downregulated.set_xticks(np.arange(10))
        ax_downregulated.set_xticklabels(ordered_features[:10],rotation=45,verticalalignment='top',horizontalalignment='right',fontsize=12)

        ax_upregulated = fig.add_axes([.25,.775,.2,.2])
        ax_upregulated.set_title("Upregulated Genes",fontsize=20)
        ax_upregulated.set_ylabel("Mean upregulation (Log TPM)",labelpad=10,fontsize=10)
        ax_upregulated.yaxis.set_label_position('right')
        ax_upregulated.bar(np.arange(10),ordered_difference[-10:])
        ax_upregulated.set_ylim(-range,range)
        ax_upregulated.set_xticks(np.arange(10))
        ax_upregulated.set_xticklabels(ordered_features[-10:],rotation=45,verticalalignment='top',horizontalalignment='right',fontsize=12)

        ordered_prerequisites,prerequisite_counts = self.prerequisite_frequency(plot=False)

        ax_prerequisites = fig.add_axes([.025,.41,.45,.2])
        ax_prerequisites.set_title("Prerequisites By Frequency",fontsize=20)
        ax_prerequisites.bar(np.arange(10),prerequisite_counts[-10:])
        ax_prerequisites.set_ylabel("Frequency",fontsize=15)
        ax_prerequisites.set_xticks(np.arange(10))
        ax_prerequisites.set_xticklabels(ordered_prerequisites[-10:],rotation=45,verticalalignment='top',horizontalalignment='right',fontsize=12)

        cell_clusters,cell_cluster_frequency = self.cell_cluster_frequency(plot=False)

        ax_cluster_frequency = fig.add_axes([.025,.025,.45,.2])
        ax_cluster_frequency.set_title("Leaf Cluster/Cell Cluster Relation",fontsize=20)
        ax_cluster_frequency.bar(np.arange(len(cell_clusters)),cell_cluster_frequency)
        ax_cluster_frequency.set_xticks(np.arange(len(cell_clusters)))
        ax_cluster_frequency.set_xticklabels(cell_clusters,fontsize=15)
        ax_cluster_frequency.set_xlabel("Cell Clusters",fontsize=15)
        ax_cluster_frequency.set_ylabel("Frequency",fontsize=15)

        prereqs = self.average_prereq_freq_level(plot=False)

        prereqs = [prereq for prereq in prereqs if prereq[0][:2] != "CG"]
        prereqs = sorted(prereqs,key=lambda prereq: prereq[1][1])[::-1]

        prereq_features = [prereq[0] for prereq in prereqs]
        prereq_levels = [prereq[1][0] for prereq in prereqs]
        prereq_frequencies = [prereq[1][1] * 10 for prereq in prereqs]

        ax_path = fig.add_axes([.6,.025,.2,.95])
        ax_path.set_title(f"The Path to Cluster {self.id}",fontsize=20)
        ax_path.scatter(prereq_levels[:50],np.arange(49,-1,-1),s=prereq_frequencies[:50])
        ax_path.set_xlabel("Average Level of Decision",fontsize=15)
        # ax_path.set_xlim(max(prereq_levels[:20])*1.1,-0.01)
        ax_path.set_yticks(np.arange(49,-1,-1))
        ax_path.set_yticklabels(prereq_features[:50],fontsize=14)
        # ax_path.grid(axis='y')

        plt.show()

    def cell_cluster_frequency(self,plot=True):
        cell_cluster_labels = self.forest.sample_labels
        cell_counts = self.cell_counts()
        cell_clusters = sorted(list(set(cell_cluster_labels)))
        cluster_counts = []
        for cluster in cell_clusters:
            cluster_mask = cell_cluster_labels == cluster
            cluster_counts.append(np.sum(cell_counts[cluster_mask]))

        if plot:
            plt.figure()
            plt.title("Frequency of cell clusters in leaf cluster")
            plt.bar(np.arange(len(cell_clusters)),cluster_counts,tick_labels=cell_clusters)
            plt.ylabel("Frequency")
            plt.show()

        return cell_clusters,cluster_counts


    def cell_counts(self):
        encoding = self.encoding()
        return np.sum(encoding,axis=1)

    def plot_cell_counts(self,**kwargs):
        counts = self.cell_counts()
        plt.figure(figsize=(15,10))
        plt.scatter(self.forest.tsne(no_plot=True)[:,0],self.forest.tsne(no_plot=True)[:,1],c=counts,**kwargs)
        plt.colorbar()
        plt.show()

    def cell_frequency(self):
        encoding = self.encoding()
        return np.sum(encoding,axis=1)/np.sum(encoding.flatten())

    def prerequisites(self):
        prerequisite_dictionary = {}
        for node in self.nodes:
            for prerequisite,split,sign in node.prerequisites:
                if prerequisite not in prerequisite_dictionary:
                    prerequisite_dictionary[prerequisite] = []
                prerequisite_dictionary[prerequisite].append((split,sign))
        return prerequisite_dictionary

    def prerequisites_by_level(self):
        levels = []
        max_depth = max([node.level for node in self.nodes])
        for level in range(max_depth):
            levels.append([])
            for node in self.nodes:
                try:
                    levels[-1].append(node.prerequisites[level])
                except:
                    pass
        return levels

    def average_prereq_freq_level(self,plot=True):
        prereqs = self.forest.average_prereq_freq_level(self.nodes)

        if plot:
            sorted_prereqs = sorted(prereqs,key=lambda prereq: prereq[1][1])[::-1]
            prereq_features = [prereq[0] for prereq in sorted_prereqs]
            prereq_levels = [prereq[1][0] for prereq in sorted_prereqs]
            prereq_frequencies = [prereq[1][1] * 10 for prereq in sorted_prereqs]

            plt.figure(figsize=(4,10))
            plt.title(f"The Path to Cluster {self.id}",fontsize=15)
            plt.scatter(prereq_levels[:30],np.arange(29,-1,-1),s=prereq_frequencies[:30])
            plt.xlabel("Average Level of Decision",fontsize=15)
            plt.yticks(np.arange(29,-1,-1),prereq_features[:30],fontsize=15)
            plt.show()

        return prereqs

    def prerequisite_frequency(self,n=50,plot=True):
        prerequisites = list(self.prerequisites().items())
        prerequisites.sort(key=lambda x: len(x[1]))
        prerequisite_counts = [len(x[1]) for x in prerequisites]
        prerequisite_labels = [x[0] for x in prerequisites]

        if plot:
            plt.figure(figsize=(10,2))
            plt.title("Prerequisites By Frequency")
            plt.scatter(np.arange(n),prerequisite_counts[-n:])
            plt.xlim(0,n)
            plt.xlabel("Gene Symbol")
            plt.ylabel("Frequency")
            plt.xticks(np.arange(n),prerequisite_labels[-n:],rotation='vertical')
            plt.show()

        return prerequisite_labels,prerequisite_counts

    # def mean_feature_predictions(self):
    #     return self.forest.weighted_node_prediction(self.nodes)

    def weighted_feature_predictions(self):
        return self.forest.weighted_node_vector_prediction(self.nodes)

    def increased_features(self,n=50,plot=True):
        initial_medians = self.forest.weighted_node_vector_prediction([self.forest.prototype.root])
        leaf_medians = self.weighted_feature_predictions()
        difference = leaf_medians - initial_medians
        feature_order = np.argsort(difference)
        ordered_features = np.array(self.forest.features)[feature_order]
        ordered_difference = difference[feature_order]

        if plot:
            plt.figure(figsize=(10,2))
            plt.title("Upregulated Genes")
            plt.scatter(np.arange(n),ordered_difference[-n:])
            plt.xlim(0,n)
            plt.xlabel("Gene Symbol")
            plt.ylabel("Frequency")
            plt.xticks(np.arange(n),ordered_features[-n:],rotation='vertical')
            plt.show()

        return ordered_features,ordered_difference

    def decreased_features(self,n=50,plot=True):
        initial_medians = self.forest.weighted_node_vector_prediction([self.forest.prototype.root])
        leaf_medians = self.weighted_feature_predictions()
        difference = leaf_medians - initial_medians
        feature_order = np.argsort(difference)
        ordered_features = np.array(self.forest.features)[feature_order]
        ordered_difference = difference[feature_order]

        if plot:
            plt.figure(figsize=(10,2))
            plt.title("Upregulated Genes")
            plt.scatter(np.arange(n),ordered_difference[:n])
            plt.xlim(0,n)
            plt.xlabel("Gene Symbol")
            plt.ylabel("Frequency")
            plt.xticks(np.arange(n),ordered_features[:n],rotation='vertical')
            plt.show()

        return ordered_features,ordered_difference


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
# import community
# import networkx as nx

def numpy_mad(mtx):
    medians = []
    for column in mtx.T:
        medians.append(np.median(column[column!=0]))
    median_distances = np.abs(mtx - np.tile(np.array(medians), (mtx.shape[0],1)))
    mads = []
    for (i,column) in enumerate(median_distances.T):
        mads.append(np.median(column[mtx[:,i]!=0]))
    return np.array(mads)

def nonzero_var_column(mtx):
    nzv = np.zeros(mtx.shape[1])
    for i in range(mtx.shape[1]):
        nzv[i] = np.var(mtx[:,i][mtx[:,i] != 0])
    return nzv

def sample_node_encoding(nodes,samples):
    encoding = np.zeros((len(nodes),samples),dtype=bool)
    for i,node in enumerate(nodes):
        encoding[i] = node.sample_mask()
    sample_encoding = encoding.T
    unrepresented = np.sum(sample_encoding,axis=1) == 0
    if np.sum(unrepresented) > 0:
        sample_encoding[unrepresented] = 1;
    return sample_encoding


def feature_node_index(nodes,feature):
    encoding = np.zeros(len(nodes),dtype=bool)
    for i,node in enumerate(nodes):
        if feature in node.features:
            encoding[i] = True
    return encoding


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
#
# def node_hdbscan_samples(nodes,samples):
#
#     node_encoding = node_sample_encoding(nodes,samples)
#
#     pre_computed_distance = pdist(node_encoding,metric='cityblock')
#
#     clustering_model = HDBSCAN(min_cluster_size=50, metric='precomputed')
#
# #     plt.figure()
# #     plt.title("Dbscan observed distances")
# #     plt.hist(pre_computed_distance,bins=50)
# #     plt.show()
#
#     clusters = clustering_model.fit_predict(scipy.spatial.distance.squareform(pre_computed_distance))
#
# #     clusters = clustering_model.fit_predict(node_encoding)

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
#
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

# def embedded_hdbscan(coordinates):
#
#     clustering_model = HDBSCAN(min_cluster_size=50)
#     clusters = clustering_model.fit_predict(coordinates)
#     return clusters
#
# def sample_hdbscan(nodes,samples):
#
#     node_encoding = node_sample_encoding(nodes,samples)
#     embedding_model = PCA(n_components=100)
#     pre_computed_embedded = embedding_model.fit_transform(node_encoding.T)
#     print("Sample HDBscan Encoding: {}".format(pre_computed_embedded.shape))
# #     pre_computed_distance = coocurrence_distance(node_encoding)
#     pre_computed_distance = scipy.spatial.distance.squareform(pdist(pre_computed_embedded,metric='correlation'))
#     print("Sample HDBscan Distance Matrix: {}".format(pre_computed_distance.shape))
# #     pre_computed_distance[pre_computed_distance == 0] += .000001
#     pre_computed_distance[np.isnan(pre_computed_distance)] = 10000000
#     clustering_model = HDBSCAN(min_samples=3,metric='precomputed')
#     clusters = clustering_model.fit_predict(pre_computed_distance)
#
#     return clusters

def cluster_labels_to_connectivity(labels):
    samples = labels.shape[0]
    clusters = list(set(labels))
    cluster_masks = []
    connectivity = np.zeros((samples,samples),dtype=bool)
    for cluster in clusters:
        cluster_masks.append(labels == cluster)
    for cluster_mask in cluster_masks:
        vertical_mask = np.zeros((samples,samples),dtype=bool)
        horizontal_mask = np.zeros((samples,samples),dtype=bool)
        vertical_mask[:,cluster_mask] = True
        horizontal_mask[cluster_mask] = True
        square_mask = np.logical_and(vertical_mask,horizontal_mask)
        connectivity[square_mask] = True
    return connectivity

def sample_agglomerative(nodes,samples,n_clusters):

    node_encoding = node_sample_encoding(nodes,samples)

    pre_computed_distance = pdist(node_encoding.T,metric='cosine')

    clustering_model = AgglomerativeClustering(n_clusters=n_clusters,affinity='precomputed')

    clusters = clustering_model.fit_predict(scipy.spatial.distance.squareform(pre_computed_distance))

#     clusters = clustering_model.fit_predict(node_encoding)

    return clusters

def stack_dictionaries(dictionaries):
    stacked = {}
    for dictionary in dictionaries:
        for key,value in dictionary.items():
            if key not in stacked:
                stacked[key] = []
            stacked[key].append(value)
    return stacked

def consolidate_entries(keys,dictionaries):
    consolidated = empty_list_dictionary(keys)
    for dictionary in dictionaries:
        for key,value in iter(dictionary):
            if key not in consolidated:
                consolidated[key] = []
            consolidated[entry].append(value)
    return consolidated

def empty_list_dictionary(keys):
    return {key:[] for key in keys}

def count_list_elements(elements):
    dict = {}
    for element in elements:
        if element not in dict:
            dict[element] = 0
        dict[element] += 1
    return dict
