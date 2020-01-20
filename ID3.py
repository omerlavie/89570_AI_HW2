import math
from read_file import all_samples_are_classified
from read_file import create_attributes_outcomes

POSITIVE = "yes"
NEGATIVE = "no"
DEFAULT_CLASSIFICATION = POSITIVE


class Tree:
    """
    This is a class for a decision Tree and it's nodes.

    Attributes:
        attribute (str): The best attribute to divide the tree by.
        attributes_left (list): Attributes left to choose from.
        data (samples): List of all the samples in this node.
        sons (dict); List of all the outcomes of an attribute.
        is_node (bool): If the tree is a node)
    """

    def __init__(self, attribute, attributes_left, data):
        """
        This is a constructor for Tree class.

        Attributes:
            attribute (str): The best attribute to divide the tree by.
            attributes_left (list): Attributes left to choose from.
            data (samples): List of all the samples in this node.
        """
        # Name of node.
        self.attribute = attribute
        # # Branches of node.
        # self.branch = branch
        # Sons of node.
        self.sons = {}
        # Attributes left.
        self.attributes_left = attributes_left
        # Data sampled left.
        self.data = data
        self.is_node = False

    def add_son(self, branch, node):
        """
        The function adds a son to the tree.

        Attributes:
            branch (str): The outcome that this node came from.
            node (tree): The node itself.
        """
        self.sons[branch] = node


class Node:
    """
    This is a class for a decision Tree and it's nodes.

    Attributes:
        samples (list): All the samples in this node.
        classification (str): The classification of the samples in this node.
        branch (samples): List of all the samples in this node.
        is_node (bool): If the node is a node.
    """

    def __init__(self, samples, classification, branch):
        """
        The constructor for Node class.

        Parameters:
            samples (str): All the samples in this node.
            classification (list): The classification of the samples in this node.
            branch (samples): List of all the samples in this node.
        """
        # If the node is a leaf there will be a classification.
        self.classification = classification
        # Data sampled left.
        self.data = samples
        self.branch = branch
        self.is_node = True


def all_entropy(data):
    """
    The function calculates the entropy on an entire dataset.

    Parameters:
       data (samples): The data that entropy will be calculated on .

    Returns:
        The entropy on the entire samples.
    """
    pos = neg = 0
    if len(data) == 0:
        return 0
    for sample in data:
        if sample.classification == POSITIVE:
            pos += 1
        else:
            neg += 1
    pos_prob = pos / len(data)
    neg_prob = neg / len(data)
    if pos == 0:
        return - (neg_prob * math.log2(neg_prob))
    if neg_prob == 0:
        return -pos_prob * math.log2(pos_prob)
    result = -pos_prob * math.log2(pos_prob) - (neg_prob * math.log2(neg_prob))
    return result


def entropy(data, attribute):
    """
    The function calculates the entropy on an dataset and attribute.

    Parameters:
       data (samples): The data that entropy will be calculated on.
       attribute (str, str): The attribute to filter the set with.

    Returns:
        The entropy on the entire samples filtered with the attribute.
    """
    pos = neg = 0
    outcome = attribute[1]
    attribute = attribute[0]
    for sample in data:
        if sample.attributes_data[attribute] == outcome:
            if sample.classification == POSITIVE:
                pos += 1
            else:
                neg += 1
    if pos == 0 or neg == 0:
        return 0
    result = -((pos / (pos + neg)) * math.log2(pos / (pos + neg))) - (
                (neg / (pos + neg)) * math.log2(neg / (pos + neg)))
    return result


def average_information_entropy(data, attribute):
    """
    The function calculates the average information entropy on an dataset and attribute.

    Parameters:
       data (samples): The data that average information entropy will be calculated on.
       attribute (str): The attribute to filter the set with.

    Returns:
        The average information entropy on the entire samples filtered with the attribute.
    """
    attribute_outcomes = create_attributes_outcomes(data)[attribute]
    pos_feature = neg_feature = 0
    total_entropy = 0
    for outcome in attribute_outcomes:
        for sample in data:
            if sample.attributes_data[attribute] == outcome:
                if sample.classification == POSITIVE:
                    pos_feature += 1
                else:
                    neg_feature += 1
        total_entropy += ((pos_feature + neg_feature) / len(data)) * entropy(data, (attribute, outcome))
        pos_feature = neg_feature = 0
    return total_entropy


def information_gain(data, attribute):
    """
    The function calculates the information gain on an dataset and attribute.

    Parameters:
       data (samples): The data that entropy will be calculated on.
       attribute (str, str): The attribute to filter the set with.

    Returns:
        The information gain on the entire samples filtered with the attribute.
    """
    return all_entropy(data) - average_information_entropy(data, attribute)


def get_best_info_gain(data, attributes):
    """
    The function calculates the attribute that will give the best information gain on a dataset.

    Parameters:
       data (samples): The data.
       attributes (str): List of attributes.

    Returns:
        The function returns the attribute that will give the highest information gain.
    """
    max_info_gain_attr = None
    max_info_gain = 0
    for attribute in attributes:
        ig = information_gain(data, attribute)
        if ig > max_info_gain:
            max_info_gain = ig
            max_info_gain_attr = attribute
    if max_info_gain_attr == None:
        try:
            return attributes[0]
        except:
            print("stop")
    return max_info_gain_attr


def filter_samples(samples, attribute, outcome):
    """
    The function filter samples by an outcome of an attribute.

    Parameters:
       samples (samples): The data.
       attribute (str): An attribute.
       outcome (str): An outcome of that attribute.

    Returns:
        The function returns a filtered list of all the data set by the outcome.
    """
    filtered_samples = []
    for sample in samples:
        if sample.attributes_data[attribute] == outcome:
            filtered_samples.append(sample)
    return filtered_samples


def build_tree(samples, attributes, outcomes):
    """
    The function builds a tree by a given dataset.

    Parameters:
       samples (samples): The data.
       attributes (str): List of attributes available in this dataset.
       outcomes (dict): Dictionary of all attributes and their outcomes/

    Returns:
        The function returns a decision tree using the ID3 algorithm.
    """
    best_attribute = get_best_info_gain(samples, attributes)
    attributes_left = copy_list(attributes)
    attributes_left.remove(best_attribute)
    tree = Tree(best_attribute, attributes_left, samples)
    outcomes[best_attribute] = sorted(outcomes[best_attribute])
    for outcome in outcomes[best_attribute]:
        subset_samples = filter_samples(samples, best_attribute, outcome)
        samples_classification = all_samples_are_classified(subset_samples)
        if len(subset_samples) == 0:
            tree.add_son(outcome, Node(subset_samples, DEFAULT_CLASSIFICATION, outcome))
        elif len(attributes_left) == 0:
            # No more attributes to divide by are left - meaning there are same samples in the data set with different
            # classification.
            tree.add_son(outcome, Node(subset_samples, most_common_classification(subset_samples), outcome))
        elif samples_classification != 0:
            tree.add_son(outcome, Node(subset_samples, samples_classification, outcome))
        else:
            tree.add_son(outcome, build_tree(subset_samples, attributes_left, outcomes))
    return tree


def most_common_classification(samples):
    """
    The function checks what is the most common classification on a dataset.

    Parameters:
       samples (samples): The data.

    Returns:
        The function returns the most common classification on a dataset.
    """
    pos = neg = 0
    for sample in samples:
        if sample.classification == POSITIVE:
            pos += 1
        else:
            neg += 1
    if pos >= neg:
        return POSITIVE
    else:
        return NEGATIVE


def predict_classification(tree, sample):
    """
    The function predict the classification of a sample.

    Parameters:
        tree (Tree): A decision tree.
        sample (sample): The sample to predict.

    Returns:
        The function returns the prediction of a sample based on the tree given.
    """
    while not tree.is_node:
        tree = tree.sons[sample.attributes_data[tree.attribute]]
    return tree.classification


def copy_list(list_to_copy):
    """
    The function deep copy a list.

    Parameters:
        list_to_copy (list): List to copy.

    Returns:
        The function returns a deep copy of a list.
    """
    new_list = []
    for item in list_to_copy:
        new_list.append(item)
    return new_list


def get_accuracy(training_set, test_set):
    """
    The function calculates the accuracy of the dataset by building a tree using the training_set and getting
    predicions using the test_set.

    Parameters:
        training_set (list): List of samples to build the decision tree on.
        test_set (list): List of sampled to get the accuracy of the tree.

    Returns:
        The function returns the accuracy of the model on the testing_set.
    """
    attributes = create_attributes_outcomes(training_set)
    outcomes = attributes
    tree = build_tree(training_set, list(attributes.keys()), outcomes)
    f = open("output.txt", "w+")
    write_tree(tree, 0, True, f)
    f.close()
    true_pos = true_neg = false_pos = false_neg = 0
    for t in test_set:
        t_prediction = predict_classification(tree, t)
        if t_prediction == t.classification:
            if t_prediction == POSITIVE:
                true_pos += 1
            else:
                true_neg += 1
        else:
            if t_prediction == POSITIVE:
                false_pos += 1
            else:
                false_neg += 1
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_neg + false_pos)
    return accuracy


def write_tree(tree, depth, root, f):
    """
    The function write the tree into a file.

    Parameters:
        tree (Tree): The decision tree.
        depth (int): The depth of the current node.
        root (Boolean): If this is the root of the tree.
        f (file): file descriptor.
    """
    if tree.is_node:
        f.write(":" + tree.classification + "\n")
        return
    elif not root:
        depth += 1
        f.write("\n")
    for son in tree.sons:
        classification = son
        son = tree.sons[son]
        if not root:
            f.write("\t" * depth + '|')
        f.write(tree.attribute + "=" + classification)
        write_tree(son, depth, False, f)
