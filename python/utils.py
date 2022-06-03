"""Data processing utilities."""

import json
import math
from texttable import Texttable
import dgl
import torch
import numpy as np
from scipy import stats 


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def process_pair(path):
    """
    Reading a json file with a pair of graphs.
    :param path: Path to a JSON file.
    :return data: Dictionary with data, also containing processed DGL graphs.
    """
    data = json.load(open(path))
    edges_1 = data["graph_1"] + [[y, x] for x, y in data["graph_1"]]

    edges_2 = data["graph_2"] + [[y, x] for x, y in data["graph_2"]]

    edges_1 = np.array(edges_1, dtype=np.int64);
    edges_2 = np.array(edges_2, dtype=np.int64);
    G_1 = dgl.DGLGraph((edges_1[:,0], edges_1[:,1]));
    G_2 = dgl.DGLGraph((edges_2[:,0], edges_2[:,1]));
    
    data['G_1'] = G_1;
    data['G_2'] = G_2;
    return data

def calculate_sigmoid_loss(prediction, target):
    """
    Calculating the squared loss on the normalized GED.
    :param prediction: Predicted log value of GED.
    :param target: Factual log transofmed GED.
    :return score: Squared error.
    """
    prediction = prediction
    target = target
    score = (prediction-target)**2
    return score

def calculate_loss(prediction, target):
    """
    Calculating the squared loss on the normalized GED.
    :param prediction: Predicted log value of GED.
    :param target: Factual log transofmed GED.
    :return score: Squared error.
    """
    prediction = -math.log(prediction)
    target = -math.log(target)
    score = (prediction-target)**2
    return score

def calculate_sigmoid_loss(prediction, data):
    """
    Calculating the squared loss on the sigmoid space (similarity).
    :param prediction: Predicted log value of GED.
    :param target: Factual log transofmed GED.
    :return score: Squared error.
    """
    if type(data) == list:
        target = np.array([d_instance['target'].detach().numpy() for d_instance in data])
    else:
        target = data["target"].detach().numpy()
    
    score = (prediction-target)**2
    return score

def calculate_normalized_ged(data):
    """
    Calculating the normalized GED for a pair of graphs.
    :param data: Data table.
    :return norm_ged: Normalized GED score.
    """
    norm_ged = data["ged"]/(0.5*(len(data["labels_1"])+len(data["labels_2"])))
    return norm_ged


class Metric():
    def __init__(self, instances, ged_scaling=None, min_ged=None, max_ged=None):
        self.instances = instances;
        self.ged = [];
        self.normalized_ged = [];
        self.normalization_constant = [];
        per_node_dict = {};
        for i,entry in enumerate(instances):
            if entry['id_1'] not in per_node_dict:
                per_node_dict[entry['id_1']] = [];
            if entry['id_2'] not in per_node_dict:
                per_node_dict[entry['id_2']] = [];
            per_node_dict[entry['id_1']].append(i)
            per_node_dict[entry['id_2']].append(i)
            self.ged.append(entry['ged']);
            if ged_scaling is None:
                norm_ged = entry["ged"]/(0.5*(len(entry["labels_1"])+len(entry["labels_2"])));
                self.normalized_ged.append(torch.from_numpy(np.exp(-norm_ged).reshape(1, 1)).view(-1).float());
            elif ged_scaling == 'minmax_scaling_v1':
                assert min_ged is not None and max_ged is not None
                norm_ged = entry["ged"]/(0.5*(len(entry["labels_1"])+len(entry["labels_2"])))
                norm_ged = np.array([(norm_ged - min_ged) / (max_ged - min_ged)])
                self.normalized_ged.append(torch.from_numpy(norm_ged.reshape(1, 1)).view(-1).float());
            elif ged_scaling == 'minmax_scaling_v2':
                assert min_ged is not None and max_ged is not None
                norm_ged = entry["ged"]
                norm_ged = np.array([(norm_ged - min_ged) / (max_ged - min_ged)])
                self.normalized_ged.append(torch.from_numpy(norm_ged.reshape(1, 1)).view(-1).float());
            self.normalization_constant.append(0.5*(len(entry["labels_1"])+len(entry["labels_2"])));
            
        self.ged = np.array(self.ged)
        self.normalized_ged = np.array(self.normalized_ged)
            
        self.per_node_dict_ged = {_:{i:self.ged[i] for i in per_node_dict[_]} for _ in per_node_dict};
        self.per_node_dict_normalized_ged = {_:{i:self.normalized_ged[i] for i in per_node_dict[_]} for _ in per_node_dict};
        self.per_node_dict_ged = {i: sorted(self.per_node_dict_ged[i].items(), key=lambda x: x[1]) for i in self.per_node_dict_ged}
        self.per_node_dict_normalized_ged = {i: sorted(self.per_node_dict_normalized_ged[i].items(), key=lambda x: x[1]) for i in self.per_node_dict_ged}
        
        self.per_node_dict_ged_sorted_ids = {i:np.array([_[0] for _ in self.per_node_dict_ged[i]]) for i in self.per_node_dict_ged};
        self.per_node_dict_ged_sorted = {i:np.array([_[1] for _ in self.per_node_dict_ged[i]]) for i in self.per_node_dict_ged};
        self.per_node_dict_ged_sorted_changes = {i:np.array([0]*len(self.per_node_dict_ged[i])) for i in self.per_node_dict_ged};
        self.per_node_dict_normalized_ged_sorted_ids = {i:np.array([_[0] for _ in self.per_node_dict_normalized_ged[i]]) for i in self.per_node_dict_normalized_ged};
        self.per_node_dict_normalized_ged_sorted = {i:np.array([_[1] for _ in self.per_node_dict_normalized_ged[i]]) for i in self.per_node_dict_normalized_ged};
        self.per_node_dict_normalized_ged_sorted_changes = {i:np.array([0]*len(self.per_node_dict_normalized_ged[i])) for i in self.per_node_dict_normalized_ged};
        
        for i in self.per_node_dict_ged_sorted_ids:
            prev_val = -1;
            prev_key = -1;
            for j in range(len(self.per_node_dict_ged_sorted_ids[i])-1, -1, -1):
                if self.ged[self.per_node_dict_ged_sorted_ids[i][j]] != prev_val:
                    prev_key = j;
                    prev_val = self.ged[self.per_node_dict_ged_sorted_ids[i][j]];
                self.per_node_dict_ged_sorted_changes[i][j] = prev_key;
                    
                
        for i in self.per_node_dict_normalized_ged_sorted_ids:
            prev_val = -1;
            prev_key = -1;
            for j in range(len(self.per_node_dict_normalized_ged_sorted_ids[i])-1, -1, -1):
                if self.normalized_ged[self.per_node_dict_normalized_ged_sorted_ids[i][j]] != prev_val:
                    prev_key = j;
                    prev_val = self.normalized_ged[self.per_node_dict_normalized_ged_sorted_ids[i][j]];
                self.per_node_dict_normalized_ged_sorted_changes[i][j] = prev_key;
            
    def precision_with_ties(self, y_true_changes, y_score, k=10):
        # highest rank is 1 so +2 instead of +1
        k = min(k, len(y_true_changes))
        relevant_set = set(range(y_true_changes[k]+1));
        order = np.argsort(y_score);
        prec = 0.0;
        a = 0;
        while a < k and a < len(y_score):
            b = a+1;
            while b < len(y_score) and y_score[order[b]]==y_score[order[a]]:
                b += 1;
            rel_sum = 0.0;
            for i in range(a, b):
                rel = 0;
                if order[i] in relevant_set:
                    rel = 1;
                rel_sum += rel
            
            if k >= b:
                prec += rel_sum;
            else:
                prec += (rel_sum*(k-a))/(b-a);
            a = b;
        return prec/k;
    
    def average_precision_at_k(self, predictions, k=10, unnormalized=False):
        if unnormalized:
            predictions = -np.log(predictions) * self.normalization_constant; #TODO: remove this
            precision_list = [self.precision_with_ties(self.per_node_dict_ged_sorted_changes[i], \
                                                       predictions[self.per_node_dict_ged_sorted_ids[i]], k=k) \
                             for i in self.per_node_dict_ged_sorted_ids];
            return np.mean(precision_list);
        else:
            precision_list = [self.precision_with_ties(self.per_node_dict_normalized_ged_sorted_changes[i], \
                                                       predictions[self.per_node_dict_normalized_ged_sorted_ids[i]], k=k) \
                             for i in self.per_node_dict_normalized_ged_sorted_ids];

            return np.mean(precision_list);    
        
    def mean_average_precision(predictions, unnormalized=False):
        pass;
    def mse(self, predictions, unnormalized=False):
        if unnormalized:
            predictions = -np.log(predictions) * self.normalization_constant;
            se_list = (predictions - self.ged)**2;
            return np.mean(se_list);
        else:
            se_list = (predictions - self.normalized_ged)**2;
            return np.mean(se_list);
    def mae(self, predictions, unnormalized=False):
        if unnormalized:
            predictions = -np.log(predictions) * self.normalization_constant;
            ae_list = np.absolute(predictions - self.ged);
            return np.mean(ae_list);
        else:
            ae_list = np.absolute(predictions - self.normalized_ged);
            return np.mean(ae_list);
        
    def spearman(self, predictions, mode='macro', unnormalized=False):
        if unnormalized:
            predictions = -np.log(predictions) * self.normalization_constant;
            if mode=="macro":
                correlation_list = [stats.spearmanr(self.per_node_dict_ged_sorted[i], \
                                                       predictions[self.per_node_dict_ged_sorted_ids[i]])[0] \
                                    if np.std(predictions[self.per_node_dict_ged_sorted_ids[i]]) > 0.0000001 else 0.0 \
                             for i in self.per_node_dict_ged_sorted_ids];
                return np.mean(correlation_list);
            else:
                if np.std(predictions) < 0.0000001:
                    return 0.0;
                return stats.spearmanr(self.ged, predictions)[0]
        else:
            if mode=="macro":
                correlation_list = [stats.spearmanr(self.per_node_dict_normalized_ged_sorted[i], \
                                                       predictions[self.per_node_dict_normalized_ged_sorted_ids[i]])[0] \
                                    if np.std(predictions[self.per_node_dict_normalized_ged_sorted_ids[i]]) > 0.0000001 else 0.0 \
                             for i in self.per_node_dict_normalized_ged_sorted_ids];
                return np.mean(correlation_list);
            else:
                if np.std(predictions) < 0.0000001:
                    return 0.0;
                return stats.spearmanr(self.normalized_ged, predictions)[0]
        
    def kendalltau(self, predictions, mode='macro', unnormalized=False, nan_policy='propagate', return_all=False):
        if unnormalized:
            predictions = -np.log(predictions) * self.normalization_constant;
            if mode=="macro":
                correlation_list = [stats.kendalltau(self.per_node_dict_ged_sorted[i], \
                                                       predictions[self.per_node_dict_ged_sorted_ids[i]])[0] \
                                    if np.std(predictions[self.per_node_dict_ged_sorted_ids[i]]) > 0.0000001 else 0.0 \
                             for i in self.per_node_dict_ged_sorted_ids];
                return np.mean(correlation_list);
            else:
                if np.std(predictions) < 0.0000001:
                    return 0.0;
                return stats.kendalltau(self.ged, predictions, nan_policy=nan_policy)[0]
        else:
            if mode=="macro":
                correlation_list = [stats.kendalltau(self.per_node_dict_normalized_ged_sorted[i], \
                                                       predictions[self.per_node_dict_normalized_ged_sorted_ids[i]])[0] \
                                    if np.std(predictions[self.per_node_dict_normalized_ged_sorted_ids[i]]) > 0.0000001 else 0.0 \
                             for i in self.per_node_dict_normalized_ged_sorted_ids];
                if return_all:
                    return correlation_list;
                else:
                    return np.mean(correlation_list);
            else:
                if np.std(predictions) < 0.0000001:
                    return 0.0;
                return stats.kendalltau(self.normalized_ged, predictions, nan_policy=nan_policy)[0]

            
class MetricV1():
    def __init__(self, instances):
        self.instances = instances;
        self.ged = [];
        per_node_dict = {};
        for i,entry in enumerate(instances):
            if entry['id_1'] not in per_node_dict:
                per_node_dict[entry['id_1']] = [];
            if entry['id_2'] not in per_node_dict:
                per_node_dict[entry['id_2']] = [];
            per_node_dict[entry['id_1']].append(i)
            per_node_dict[entry['id_2']].append(i)
            self.ged.append(entry['target']);
            
        self.ged = np.array(self.ged)
            
        self.per_node_dict_ged = {_:{i:self.ged[i] for i in per_node_dict[_]} for _ in per_node_dict};
        self.per_node_dict_ged = {i: sorted(self.per_node_dict_ged[i].items(), key=lambda x: x[1]) for i in self.per_node_dict_ged}
        
        self.per_node_dict_ged_sorted_ids = {i:np.array([_[0] for _ in self.per_node_dict_ged[i]]) for i in self.per_node_dict_ged};
        self.per_node_dict_ged_sorted = {i:np.array([_[1] for _ in self.per_node_dict_ged[i]]) for i in self.per_node_dict_ged};
        self.per_node_dict_ged_sorted_changes = {i:np.array([0]*len(self.per_node_dict_ged[i])) for i in self.per_node_dict_ged};
        
        for i in self.per_node_dict_ged_sorted_ids:
            prev_val = -1;
            prev_key = -1;
            for j in range(len(self.per_node_dict_ged_sorted_ids[i])-1, -1, -1):
                if self.ged[self.per_node_dict_ged_sorted_ids[i][j]] != prev_val:
                    prev_key = j;
                    prev_val = self.ged[self.per_node_dict_ged_sorted_ids[i][j]];
                self.per_node_dict_ged_sorted_changes[i][j] = prev_key;
                    
    def precision_with_ties(self, y_true_changes, y_score, k=10):
        # highest rank is 1 so +2 instead of +1
        k = min(k, len(y_true_changes))
        relevant_set = set(range(y_true_changes[k]+1));
        order = np.argsort(y_score);
        prec = 0.0;
        a = 0;
        while a < k and a < len(y_score):
            b = a+1;
            while b < len(y_score) and y_score[order[b]]==y_score[order[a]]:
                b += 1;
            rel_sum = 0.0;
            for i in range(a, b):
                if order[i] in relevant_set:
                    rel_sum += 1;
            if k >= b:
                prec += rel_sum;
            else:
                prec += (rel_sum*(k-a))/(b-a);
            a = b;
        return prec/k;
    
    def average_precision_at_k(self, predictions, k=10):
        precision_list = [self.precision_with_ties(self.per_node_dict_ged_sorted_changes[i], 
                                                   predictions[self.per_node_dict_ged_sorted_changes[i]], 
                                                   k=k)
                             for i in self.per_node_dict_ged_sorted_changes];
        return np.mean(precision_list);    
        
    def mean_average_precision(predictions, unnormalized=False):
        pass;
    def mse(self, predictions, unnormalized=False):
        if unnormalized:
            predictions = -np.log(predictions) * self.normalization_constant;
            se_list = (predictions - self.ged)**2;
            return np.mean(se_list);
        else:
            se_list = (predictions - self.normalized_ged)**2;
            return np.mean(se_list);
    def mae(self, predictions, unnormalized=False):
        if unnormalized:
            predictions = -np.log(predictions) * self.normalization_constant;
            ae_list = np.absolute(predictions - self.ged);
            return np.mean(ae_list);
        else:
            ae_list = np.absolute(predictions - self.normalized_ged);
            return np.mean(ae_list);
        
    def spearman(self, predictions, mode='macro', unnormalized=False):
        if unnormalized:
            predictions = -np.log(predictions) * self.normalization_constant;
            if mode=="macro":
                correlation_list = [stats.spearmanr(self.per_node_dict_ged_sorted[i], \
                                                       predictions[self.per_node_dict_ged_sorted_ids[i]])[0] \
                                    if np.std(predictions[self.per_node_dict_ged_sorted_ids[i]]) > 0.0000001 else 0.0 \
                             for i in self.per_node_dict_ged_sorted_ids];
                return np.mean(correlation_list);
            else:
                if np.std(predictions) < 0.0000001:
                    return 0.0;
                return stats.spearmanr(self.ged, predictions)[0]
        else:
            if mode=="macro":
                correlation_list = [stats.spearmanr(self.per_node_dict_normalized_ged_sorted[i], \
                                                       predictions[self.per_node_dict_normalized_ged_sorted_ids[i]])[0] \
                                    if np.std(predictions[self.per_node_dict_normalized_ged_sorted_ids[i]]) > 0.0000001 else 0.0 \
                             for i in self.per_node_dict_normalized_ged_sorted_ids];
                return np.mean(correlation_list);
            else:
                if np.std(predictions) < 0.0000001:
                    return 0.0;
                return stats.spearmanr(self.normalized_ged, predictions)[0]
        
    def kendalltau(self, predictions, mode='macro', unnormalized=False, nan_policy='propagate'):
        if unnormalized:
            predictions = -np.log(predictions) * self.normalization_constant;
            if mode=="macro":
                correlation_list = [stats.kendalltau(self.per_node_dict_ged_sorted[i], \
                                                       predictions[self.per_node_dict_ged_sorted_ids[i]])[0] \
                                    if np.std(predictions[self.per_node_dict_ged_sorted_ids[i]]) > 0.0000001 else 0.0 \
                             for i in self.per_node_dict_ged_sorted_ids];
                return np.mean(correlation_list);
            else:
                if np.std(predictions) < 0.0000001:
                    return 0.0;
                return stats.kendalltau(self.ged, predictions, nan_policy=nan_policy)[0]
        else:
            if mode=="macro":
                correlation_list = [stats.kendalltau(self.per_node_dict_normalized_ged_sorted[i], \
                                                       predictions[self.per_node_dict_normalized_ged_sorted_ids[i]])[0] \
                                    if np.std(predictions[self.per_node_dict_normalized_ged_sorted_ids[i]]) > 0.0000001 else 0.0 \
                             for i in self.per_node_dict_normalized_ged_sorted_ids];
                return np.mean(correlation_list);
            else:
                if np.std(predictions) < 0.0000001:
                    return 0.0;
                return stats.kendalltau(self.normalized_ged, predictions, nan_policy=nan_policy)[0]
            

def summarize_results(test_results):
    keys = sorted(test_results[0].__dict__.keys())
    maxlen = max([len(k) for k in keys]) + 1
    for key in keys:
        key_results = [r.__dict__[key] for r in test_results if r is not None]
        avg = np.mean(key_results)
        std = np.std(key_results)
        
        key = key if len(key) >= maxlen else '{}{}'.format(key, ' '*(maxlen-len(key)))
        print(f'{key}: \t{avg:.05f} +/- {std:.06f}')
    
        