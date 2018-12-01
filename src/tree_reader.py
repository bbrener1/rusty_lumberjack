import matplotlib.pyplot as plt
import numpy as np
import re
import json
import sys
import os
import random
import glob
import pickle

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

# from hdbscan import HDBSCAN

from scipy.cluster import hierarchy as hrc
from scipy.cluster.hierarchy import dendrogram,linkage

from sklearn.decomposition import PCA

from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import jaccard_similarity_score
jaccard_index = jaccard_similarity_score

from multiprocessing import Pool

import copy

sys.path.append("/Users/boris/haxx/python/")
import gravity_clustering

class Node:

    def __init__(self, node_json,tree,forest,parent=None,prerequisites=None,level=0):
        if prerequisites is None:
            prerequisites = []
        self.tree = tree
        self.forest = forest
        self.parent = parent
        self.level = level
        self.feature = node_json['feature']
        self.split = node_json['split']
        self.features = node_json['features']
        self.samples = node_json['samples']
        self.medians = node_json['medians']
        self.dispersions = node_json['dispersions']
        self.weights = np.ones(len(self.features),dtype=float)
        self.local_gains = node_json['local_gains']
        self.absolute_gains = node_json['absolute_gains']
        self.children = []
        self.prerequisites = prerequisites
        if len(node_json['children']) > 0:
            self.children.append(Node(node_json['children'][0],self.tree,self.forest,self,prerequisites = prerequisites + [(self.feature,self.split,'<')],level=level+1))
            self.children.append(Node(node_json['children'][1],self.tree,self.forest,self,prerequisites = prerequisites + [(self.feature,self.split,'>')],level=level+1))

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

    def stems(self):
        stems = []
        for child in self.children:
            stems.extend(child.nodes())
        for child in self.children:
            if len(child.children) > 0:
                stems.append(child)
        return nodes

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

    def depth(self,d=0):
        for child in self.children:
            d = max(child.depth(d+1),d)
        return d

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
        sample_index = np.zeros(len(truth_dictionary.sample_dictionary),dtype=bool)
        for sample in self.samples:
            sample_index[truth_dictionary.sample_dictionary[sample]] = True
        # if np.sum(sample_index) == 0:
        #     print(self.samples)
        #     print(self.prerequisites)
        #     raise IndexError
        return sample_index

    def feature_sample_boolean_index(self,truth_dictionary=None):
        return self.sample_index(truth_dictionary),self.feature_index(truth_dictionary)

    def node_counts(self,counts=None,truth_dictionary=None):
        if counts is None:
            counts = self.forest.counts
        return counts[self.sample_index()].T[self.feature_index()].T

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

    def feature_leaves(self,feature):
        leaves = []
        for tree in self.trees:
            if feature in tree.root.features:
                leaves.extend(tree.leaves)
        return leaves

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

    def local_gain_matrix(self,nodes):
        gains = np.zeros((len(self.features),len(nodes)))
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

    def load(location, prefix="/run.*.compact", header="/run.prediction_header",truth="run.prediction_truth"):

        combined_tree_files = sorted(glob.glob(location + prefix))

        print(combined_tree_files)

        raw_forest = []

        for tree_file in combined_tree_files:
            raw_forest.append(json.load(open(tree_file.strip())))

        counts = np.loadtxt(location+truth)
        header = np.loadtxt(location+header,dtype=str)

        first_forest = Forest(raw_forest[1:],counts,features=header)

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

            truth = self.counts[:,feature_index]

            weights = Ridge(alpha=5).fit(feature_leaf_sample_encoding,truth).coef_

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

    def cluster_samples(self,override=False,*args,**kwargs):


        leaves = self.leaves()
        encoding = self.node_sample_encoding(leaves)

        if hasattr(self,'sample_clusters') and not override:
            print("Clustering has already been done")
        else:
            self.sample_labels = np.array(gravity_clustering.fit_predict(encoding,"fuzzy",*args,**kwargs))

        cluster_set = set(self.sample_labels)
        clusters = []
        for cluster in cluster_set:
            cells = np.arange(len(self.sample_labels))[self.sample_labels == cluster]
            clusters.append(SampleCluster(self,cells,cluster))

        self.sample_clusters = clusters

        return self.sample_labels

    def cluster_leaves(self,override=False,*args,**kwargs):

        leaves = self.leaves()
        encoding = self.node_sample_encoding(leaves).T

        if hasattr(self,'leaf_clusters') and not override:
            print("Clustering has already been done")
        else:
            self.leaf_labels = np.array(gravity_clustering.fit_predict(encoding,"fuzzy",*args,**kwargs))

        cluster_set = set(self.leaf_labels)

        clusters = []

        for cluster in cluster_set:
            leaf_index = np.arange(len(self.leaf_labels))[self.leaf_labels == cluster]
            clusters.append(NodeCluster(self,[leaves[i] for i in leaf_index],cluster))

        self.leaf_clusters = clusters

        return self.leaf_labels

    def cluster_features(self,*args,**kwargs):
        gain_matrix = self.absolute_gain_matrix(self.leaves())
        return gravity_clustering.fit_predict(gain_matrix,"fuzzy",*args,**kwargs)

    def plot_counts(self,no_plot=False):

        if not no_plot:
            cell_sort = dendrogram(linkage(encoding,metric='cos',method='average'),no_plot=True)['leaves']
            leaf_sort = dendrogram(linkage(encoding.T,metric='cos',method='average'),no_plot=True)['leaves']

            plt.figure(figsize=(10,10))
            plt.imshow(encoding[cell_sort].T[leaf_sort].T,cmap='binary')
            plt.show()

        return cell_sort,leaf_sort,self.counts

    def cluster_splits(self,override=False,no_plot=False,*args,**kwargs):

        nodes = self.nodes(root=False)

        gain_matrix = self.local_gain_matrix(nodes).T+1

        if hasattr(self,'split_clusters') and not override:
            print("Clustering has already been done")
        else:
            self.split_labels = np.array(gravity_clustering.fit_predict(gain_matrix,"fuzzy",*args,**kwargs))

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

    def plot_cell_clusters(self):
        if not hasattr(self,'leaf_clusters'):
            print("Warning, leaf clusters not detected")
            return None
        if not hasattr(self,'sample_clusters'):
            print("Warning, cell clusters not detected")
            return None

        cluster_coordinates = np.zeros((len(self.cell_clusters)),len(self.features))
        for i,cluster in enumerate(self.cell_clusters):
            cluster_coordinates[i] = cluster.median_feature_values()

        combined_coordinates = np.zeros((self.counts.shape[0]+len(cell_clusters),self.counts.shape[1]))

        combined_coordinates[0] = self.counts

        combined_coordinates[self.counts.shape[0]] = cluster_coordinates

        transformed = TSNE().fit_transform(combined_coordinates)

        highlight = np.ones(combined_coordinates.shape[0]) * 0.1
        highlight[-1 * len(self.cell_clusters):] = 5

        combined_labels = np.zeros(self.counts.shape[0]+len(cell_clusters))
        combined_labels[0] = self.cell_labels
        combined_labels[len(self.cell_labels)] = np.arange(len(self.cell_clusters))

        plt.figure()
        plt.title("TSNE-Transformed Cell Coordinates")
        plt.scatter(transformed[:,0],transformed[:,1],s=highlight,c=combined_labels)
        plt.show()


    def tsne(self,no_plot=False):
        if not hasattr(self,'tsne'):
            self.tsne = TSNE().fit_transform(self.counts)

        if not no_plot:
            plt.figure()
            plt.title("TSNE-Transformed Cell Coordinates")
            plt.scatter(self.tsne[:,0],self.tsne[:,1],s=.1)
            plt.show()

        return self.tsne

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

    def __init__(self,forest,cells,id):
        self.id = id
        self.samples = cells
        self.forest = forest

    def median_feature_values(self):
        return np.median(self.forest.counts[self.samples],axis=0)

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

    def cluster_summary(self):
        levels = [node.level for node in self.nodes]

        plt.figure()
        plt.title("Levels")
        plt.hist(levels)
        plt.show()

        leaf_size = [len(node.samples) for node in self.nodes]

        plt.figure()
        plt.title("Leaf sizes")
        plt.hist(leaf_size)
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

    def prerequisite_frequency(self):
        prerequisites = list(self.prerequisites().items())
        prerequisites.sort(key=lambda x: len(x[1]))
        prerequisite_counts = [len(x[1]) for x in prerequisites][-50:]
        prerequisite_labels = [x[0] for x in prerequisites][-50:]

        plt.figure(figsize=(10,2))
        plt.title("Prerequisites By Frequency")
        plt.scatter(np.arange(len(prerequisite_counts)),prerequisite_counts)
        plt.xlim(0,50)
        plt.xlabel("Gene Symbol")
        plt.ylabel("Frequency")
        plt.xticks(np.arange(50),prerequisite_labels,rotation='vertical')
        plt.show()

        return prerequisites

    # def mean_feature_predictions(self):
    #     return self.forest.weighted_node_prediction(self.nodes)

    def weighted_feature_predictions(self):
        return self.forest.weighted_node_vector_prediction(self.nodes)

    def increased_features(self):
        initial_medians = self.forest.weighted_node_vector_prediction([self.forest.prototype.root])
        leaf_medians = self.weighted_feature_predictions()
        difference = leaf_medians - initial_medians
        feature_order = np.argsort(difference)
        ordered_features = np.array(self.forest.features)[feature_order]
        ordered_difference = difference[feature_order]

        plt.figure(figsize=(10,2))
        plt.title("Upregulated Genes")
        plt.scatter(np.arange(50),ordered_difference[-50:])
        plt.xlim(0,50)
        plt.xlabel("Gene Symbol")
        plt.ylabel("Frequency")
        plt.xticks(np.arange(50),ordered_features[-50:],rotation='vertical')
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
        encoding[i] = node.sample_index()
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
# def hacked_louvain(nodes,samples):
#
#     node_encoding = node_sample_encoding(nodes,len(samples))
#     print("Louvain Encoding: {}".format(node_encoding.shape))
#     pre_computed_distance = coocurrence_distance(node_encoding.T)
#     print("Louvain Distances: {}".format(pre_computed_distance.shape))
#     sample_graph = nx.from_numpy_matrix(pre_computed_distance)
#     print("Louvain Cast To Graph")
#     least_spanning_tree = nx.minimum_spanning_tree(sample_graph)
#     print("Louvain Least Spanning Tree constructed")
#     part_dict = community.best_partition(least_spanning_tree)
#     print("Louvain Partition Done")
#     clustering = np.zeros(len(part_dict))
#     for i,sample in enumerate(samples):
#         clustering[i] = part_dict[int(sample)]
#     print("Louvain: {}".format(clustering.shape))
#     return clustering

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
