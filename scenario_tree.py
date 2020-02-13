import numpy as np
import pandas as pd
from copy import copy


def traverse(root, func, **kwargs):
    # inform function of entering and exiting
    func(root, 'enter', **kwargs)
    if root.children is not None:
        for i in range(len(root.children)):
            traverse(root.children[i], func, **kwargs)
    func(root, 'exit', **kwargs)


def get_node_list(root):
    node_list = []

    def func(node, enter_exit, node_list):
        if enter_exit == 'enter':
            node_list.append(node)

    traverse(root, func, node_list=node_list)
    return node_list


def get_rect_tree_params(root, cx, cy, l, mg, sep):
    root.cx = cx
    root.cy = cy
    root.l = l

    if root.children:
        N = len(root.children)
        mg_a = mg * l / (N - 1)

        s_now = cy - l / 2
        cx_now = cx + sep

        for c, p in zip(root.children, root.probs):
            l_now = p * l * (1 - mg)
            e_now = s_now + l_now
            cy_now = (s_now + e_now) / 2
            get_rect_tree_params(c, cx_now, cy_now, l_now, mg, sep)

            s_now = e_now + mg_a


def get_level(root, lv):
    root.lv = lv
    if root.children:
        for c in root.children:
            get_level(c, lv + 1)


def get_cond_prob(root, cond_prob):
    root.cond_prob = cond_prob
    if root.children:
        for c, p in zip(root.children, root.probs):
            get_cond_prob(c, p)


def get_path(node, key=None):
    c = node
    p = [c]
    while c.father is not None:
        p.append(c.father)
        c = c.father
    p = p[::-1]
    if key is None:
        return p
    else:
        return list(map(lambda x: x[key], p))


def get_tree_matrix(leaf_list):
    mat = np.asarray([get_path(l) for l in leaf_list])
    return mat


def get_full_structure(root):
    node_list = get_node_list(root)
    leaf_list = [nd for nd in node_list if nd.children is None]
    nonleaf_list = [nd for nd in node_list if nd.children is not None]
    mat = get_tree_matrix(leaf_list)
    return {'node_list': np.asarray(node_list),
            'leaf_list': np.asarray(leaf_list),
            'nonleaf_list': np.asarray(nonleaf_list),
            'mat': np.asarray(mat)}


def assign_value_to_node_list(node_list, values, key='value'):
    # temporary implementation.
    for i in range(len(node_list)):
        node_list[i].__dict__[key] = values[i]
    return node_list


def get_value_of_node_list(node_list, key='value'):
    return [node_list[i].__dict__[key] for i in range(len(node_list))]


def propagate_weights(root):
    if root.children is None:
        if root.weight is None:
            raise RuntimeError("leaf weight is None, cannot propagate")
    else:
        for c in root.children:
            propagate_weights(c)
        children_weights = np.asarray([cd.weight for cd in root.children])
        root.weight = np.sum(children_weights)
        root.probs = children_weights / root.weight


def calc_weight_from_probs(root):
    if root.weight is None:
        raise RuntimeError("root weight is none")
    if root.children is None:
        return
    if root.probs is None:
        raise RuntimeError("root probs is none")

    for k in range(len(root.children)):
        root.children[k].weight = root.weight * root.probs[k]

    for c in root.children:
        calc_weight_from_probs(c)


def propagate_father(root):
    if root.children is None:
        return
    else:
        for c in root.children:
            c.father = root
            propagate_father(c)


def propagate_martingale_values(root):
    if root.children is None:
        if root.value is None:
            raise RuntimeError("leaf value is none, cannot propagate")
    else:
        for c in root.children:
            propagate_martingale_values(c)

        children_values = np.asarray([cd.value for cd in root.children])
        root.value = np.dot(root.probs, children_values)


def duplicate_tree(root): 
    new_root = ScenarioNode(**{
        'index': root.index,
        'leaf_index': root.leaf_index,
        'value': root.value, 
        'probs': copy(root.probs),
        'weight': root.weight,
    })
    
    if root.children is not None:
        new_root.children = []
        for c in root.children:
            new_c = duplicate_tree(c)
            new_c.father = new_root
            new_root.children.append(new_c)
            
    return new_root
    
        
class ScenarioNode:
    def __init__(self, index=None, leaf_index=None, value=None, probs=None,
                 children=None, weight=None, father=None):
        self.index = index
        self.leaf_index = leaf_index
        self.value = value
        self.probs = probs
        self.weight = weight
        self.children = children
        self.father = father


class ScenarioTree:
    def __init__(self, root, duplicate=True):
        if duplicate:
            self.root = duplicate_tree(root)
        else:
            self.root = root
        
        self.propagate_father()

        self.strc = get_full_structure(self.root)
        self.node_list = self.strc['node_list']
        self.leaf_list = self.strc['leaf_list']
        self.nonleaf_list = self.strc['nonleaf_list']
        self.mat = self.strc['mat']

        # Size of Tree
        self.N = len(self.node_list)
        # Size of Leaves, Time Stages:
        self.L, self.T = self.mat.shape

        # Assign Node Indices
        self.assign(np.arange(self.N), key='index')
        # Assign Leaf Indices
        self.assign_leaf(np.arange(self.N), key='leaf_index')

    def __len__(self):
        return self.N

    def assign(self, values, key='value'):
        assign_value_to_node_list(self.node_list, values, key=key)
        return self

    def assign_leaf(self, values, key='value'):
        assign_value_to_node_list(self.leaf_list, values, key=key)
        return self

    def assign_nonleaf(self, values, key='value'):
        assign_value_to_node_list(self.nonleaf_list, values, key=key)
        return self

    def propagate_weights(self, weights=None):
        if weights is not None:
            self.assign_leaf(weights, key='weight')
        propagate_weights(self.root)

    def propagate_martingale_values(self, values=None):
        if values is not None:
            self.assign_leaf(values, key='value')
        propagate_martingale_values(self.root)

    def propagate_father(self):
        propagate_father(self.root)

    def get_mat_values(self):
        return np.asarray(list(map(lambda y:
                                   list(map(lambda x: x.value, y)), self.mat)))

    def get_distance_matrix(self, sce, cost_func):
        return np.asarray([[cost_func(get_value_of_node_list(i),
                                      get_value_of_node_list(j))
                            for j in sce.mat]
                           for i in self.mat])

    def plot(self):
        pd.DataFrame(self.get_mat_values().T).plot(legend=False, grid=True)


def construct_markov_tree(root, T, values, trans_df):
    root.children = [ScenarioNode(value=v) for v in values]
    root.probs = trans_df[root.value].values

    if T > 1:
        for c in root.children:
            construct_markov_tree(c, T - 1, values, trans_df)


class MarkovScenarioTree(ScenarioTree):
    def __init__(self, T, values, trans_mat, root_value_k=0):
        self.T = T
        self.values = values

        self.trans_mat = trans_mat
        self.trans_df = pd.DataFrame(trans_mat.T,
                                     columns=values,
                                     index=values)
        root = ScenarioNode(value=values[root_value_k])
        construct_markov_tree(root, T, values, self.trans_df)

        # with markov structure, we can calculate already the weights
        root.weight = 1.0
        calc_weight_from_probs(root)

        super(MarkovScenarioTree, self).__init__(root, duplicate=False)


class IndependentScenarioTree(MarkovScenarioTree):
    def __init__(self, T, values, root_value_k=0):
        N = len(values)
        trans_mat = np.full((N, N), 1 / N)
        super(IndependentScenarioTree, self).__init__(T=T, values=values, trans_mat=trans_mat,
                                                      root_value_k=root_value_k)
