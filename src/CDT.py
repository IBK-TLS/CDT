"""
Composition-based decision tree for  anomaly detection
-------------------------------
CDT detector.

:authors: Ines Ben Kraiem & Geoffrey Roman-Jimenez

:copyright:
    Copyright 2020 SIG Research Group, IRIT, Toulouse-France.
    

"""



import numpy as np
import uuid
import itertools
import copy
from helper import convert_label_RCIS

def gini_impurity(classes, nclasses):
    # calculation of probabilities (or fractions of observations)
    prob = [0.0 for _ in range(nclasses)] 
    N = len(classes)
    for obs in classes:
        prob[obs] += 1/N
    return sum([ p* (1-p) for p in prob])

  



class node_tree():
    def __init__(self, observations, classes, gini, parent, split_rule):
        """ Definition of node. 
        
         Parameters
        ----------
        observations : set 
            The set of observations considered in this node
        classes : list 
            The list of classes corresponding to observations.
        gini : float
            gini index of the current node
        parent : object
            node-parent of the current node
        split_rule : .....
        
        """
        self.observations = observations
        self.classes = classes
        self.gini = gini
        self.parent = parent
        self.split_rule = split_rule
        self.id = uuid.uuid1()
        
    def dict(self):
        d = {"observations": self.observations,  "classes": self.classes, 
             "gini": self.gini, "id":self.id, "parent": self.parent, 
             "split_rule": self.split_rule}
        return d

def list_of_all_possible_composition(observations):

    # Calculate all the compositions of the observations that have a class 'anomaly'.
    listofcomposition = []
    for o in observations:
        for i, j in itertools.combinations(range(len(o) + 1), 2):
            if len(o[i:j])>1 and not o[i:j] in listofcomposition:
                listofcomposition.append(o[i:j])
    return sorted(listofcomposition, key=len)
    
def islistinlist(s, l):
    if len(s) > len(l):
        return False
    else:
        return True in [ s == sl for sl in  [ l[index:index+len(s)] for index in range(len(l)-len(s)+1) ] ]



class composition_tree():
    def __init__(self, nclasses=2, iteration_max=10000, epsilon = 1e-6, inter_type=0):
        self.nclasses = nclasses
        self.queue = []
        self.tree = []
        self.root = None
        self.epsilon = epsilon
        self.iteration_max = iteration_max
        self.inter_type = inter_type
        self.window_size = None
        self.nblabels = None

    def split(self, node):
        # Split the node in [node true, node false] by maximizing the gain of Gini.
        observations = node.observations
        classes = node.classes
        parent = node.id
        gini_origin = node.gini
        gini_true = 0
        gini_false = 0
        best_composition = None
        observations_true, observations_false = [], []
        classes_true, classes_false = [], []
        
        observations_with_anomaly = [o for o, c in zip(observations, classes) if c!=0]
        classes_with_anomaly = [c for c in classes if c!=0]
        gain_gini_max = 0
        
        for composition in list_of_all_possible_composition(observations_with_anomaly):
        # split the nodes according to the presence or not of the composition
            _classes_true = [ c for o, c in zip(observations, classes) 
                            if islistinlist(composition, o) ]
            _gini_true = gini_impurity(_classes_true, self.nclasses)
            
            _classes_false = [ c for o, c in zip(observations, classes) 
                            if not islistinlist(composition, o) ]
            _gini_false = gini_impurity(_classes_false, self.nclasses)

            N, N_true, N_false = len(classes), len(_classes_true), len(_classes_false)
            
            gain_gini = gini_origin-(((N_true/N)*_gini_true)+((N_false/N)*_gini_false))
            if gain_gini > gain_gini_max:
                gain_gini_max = gain_gini
                gini_true = _gini_true
                gini_false = _gini_false
                best_composition = composition
                observations_true = [ o for o in observations 
                                     if islistinlist(composition, o) ]
                observations_false = [ o for o in observations 
                                     if not islistinlist(composition, o) ]
                classes_true = _classes_true
                classes_false = _classes_false
                
            
             
        gain_gini = gain_gini_max
        split_rule_true = {"composition": best_composition, "condition": True}
        split_rule_false = {"composition": best_composition, "condition": False }
        node_true = node_tree(observations_true, classes_true, 
                              gini_true, parent, split_rule_true)
        node_false = node_tree(observations_false, classes_false, 
                               gini_false, parent, split_rule_false)
        
        return [node_true, node_false], gain_gini 
    
    def fit(self, observations, classes):
        """ Fit CDT to the time series data.
            

        Parameters
        ----------
        observations : dictionary {number: np.array}
            list containing the windowed labeled time series data.
        classes : dictionary {number: np.array}
            list containing the classes corresponding to each observation (window).
        
        
        """
        
        gini = gini_impurity(classes, self.nclasses)
        
        self.root = node_tree(observations, classes, gini, 0, None)
        self.window_size = min([len(o) for o in observations])
        self.nblabels = len(set([ l for o in observations for l in o]))
        self.queue =  [self.root]
        self.tree = [self.root]
        
        # Tree construction 
        n=0
        while not len(self.queue) == 0 and n < self.iteration_max:
            node = self.queue.pop(0)
            splitted_nodes, gain_gini = self.split(node)
            for _node in splitted_nodes:
                if len(_node.classes) > 0 and _node.gini > self.epsilon:
                    self.queue.append(_node)
                
                if len(_node.classes) > 0:
                    self.tree.append(_node)
            n+=1  
    
    def rules_per_class(self):
        leaves = self.get_leaves()
        branches = [(l.classes, self.get_branch(l)) for l in leaves]
        rules_per_class = [[] for _ in range(self.nclasses)]
        for i, (classes, branch) in enumerate(branches):
            if not len(classes) == 0:
                setclasses =[x for i, x in enumerate(classes) if i == classes.index(x)]
                c = max(setclasses, key = classes.count)
                listofrule = [n for n in branch if n.split_rule]
                rules_per_class[c].append(listofrule)
        return rules_per_class
  
    
 
    def get_parent(self, node):
        for _node in self.tree:
            if _node.id == node.parent:
                return _node
    
    def get_childrens(self, node):
        childrens = []
        for _node in self.tree:
            if _node != self.root and _node.parent == node.id:
                
                childrens.append(_node)
        return childrens
    
    def get_leaves(self):
        leaves = []
        for node in self.tree:
            if node.gini <= self.epsilon:
                leaves.append(node)
            else:
                childrens = self.get_childrens(node)
                if len(childrens) == 0:
                    leaves.append(node)
        return leaves 
    
    def get_branch(self, leaf):
        branch = []
        node = leaf
        while node != self.root:
            branch.append(node)
            node = self.get_parent(node)
        return branch

    def class_of_node(self, node): # vote majoritaire dans une feuille impure
        setclasses =[x for i, x in enumerate(node.classes) if i == node.classes.index(x)]
        c = max( setclasses, key = node.classes.count)
        return c
    
    def which_leaf(self, observation):
        """ Classify a new observation.

        Parameters
        ----------
        observation : list 
            a windowed observation from a test datasets.
            
        Returns
        -------
        leaf : .....
        
        class_of_leaf: 0 (normal) or 1 (anomaly)
        """
        _leaf = self.root
        childrens = self.get_childrens(_leaf)
        while not len(childrens) == 0:
            for children in childrens:
                rule = children.split_rule
                checkrule = rule["condition"] is islistinlist(rule["composition"], observation)
                if checkrule:
                    _leaf = children
            childrens = self.get_childrens(_leaf)
        class_of_leaf = self.class_of_node(_leaf) 
        leaf = _leaf
        return leaf, class_of_leaf

   
   

    def anomaly_rules(self):
        """ extrat and simplify  rules from CDT.

        
        Returns
        -------
        rpc : dictionary 
            rules per class. 
        """
        rpc = self.rules_per_class()
        
        rpc = [[[{"split_rule": {
                                    "composition": [convert_label_RCIS(c) for c in r.split_rule["composition"]], 
                                    "condition": r.split_rule["condition"] }, 
                                    "classes":[r.classes.count(0), r.classes.count(1)] 
                                } 
                for r in rb] for rb in rules] for rules in rpc]
       
        return rpc





