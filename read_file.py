SPLIT_CHAR = '\t'
POSITIVE = "Yes"
NEGATIVE = "No"


class Sample:
    """
    This is a class for a Sample from a dataset.

    Attributes:
        classifying_field (str): The field in the dataset used to classify data.
        classification (str): The classification of the sample.
        attributes_data (dict): A dictionary of the attributes to outcomes of sample.
        val (int); Hemming distance from this node to the selected node.
        selected_node (Sample): The selected node.
    """
    def __init__(self, attributes, line):
        """
        The constructor for Sample class.

        Parameters:
           attributes (str): All available attributes in this dataset.
           line (str): The line from database the Sample object will be built from.
        """
        self.classifying_field = None
        self.classification = None
        self.attributes_data = {}
        line = line.split(SPLIT_CHAR)
        i = 0
        for a in attributes:
            # Delete '\n' if there is one at the end of the line.
            if line[i].endswith('\n'):
                line[i] = line[i][:-1]
            self.attributes_data[a] = line[i]
            self.classifying_field = a
            if i == len(attributes) - 1:
                self.classification = line[i]
            i += 1
        del self.attributes_data[self.classifying_field]
        self.val = None
        self.selected_node = None

    def __lt__(self, other):
        """
        The iterator for Sample class.

        Parameters:
           other (Sample): Sample.
        """
        res = self.val < other.val
        return res

    def set_distance_to(self, s):
        """
        The function sets distance from current sample to sample s.

        Parameters:
           s (Sample): Another sample.
        """
        distance = 0
        # Calculate the distance between two samples using Hamming distance.
        # For each attribute that is different from the attribute from s add 1 to val.
        for attribute in self.attributes_data:
            if self.attributes_data[attribute] != s.attributes_data[attribute]:
                distance += 1
        self.val = distance
        self.selected_node = s


def create_samples(file_path):
    """
    The function creates a list of samples from a dataset file.

    Parameters:
       file_path (str): file path to the dataset file.

    Returns:
        A list of samples.
    """
    samples = []
    file = open(file_path, 'r')
    # Read the attributes from file.
    attributes = file.readline()
    attributes = attributes.split(SPLIT_CHAR)
    if attributes[len(attributes) - 1].endswith('\n'):
        attributes[len(attributes) - 1] = attributes[len(attributes) - 1][:-1]
    line = file.readline()
    while line:
        sample = Sample(attributes, line)
        samples.append(sample)
        line = file.readline()
    file.close()
    return samples


def get_classifier(file_path):
    """
    The function gets the classifying field of a dataset file.

    Parameters:
       file_path (str): file path the dataset file.

    Returns:
        Classifier field of dataset file.
    """
    file = open(file_path, 'r')
    classifier = file.readline()
    classifier = classifier.split(SPLIT_CHAR)
    classifier = classifier[len(classifier) - 1]
    if classifier.endswith('\n'):
        classifier = classifier[:-1]
    file.close()
    return classifier


def create_attributes_outcomes(samples_set):
    """
    The function gets the outcomes of all attributes based on a sample_set.

    Parameters:
       samples_set (list): Samples dataset.

    Returns:
        A dictionary of all attributes and all of their outcomes.
    """
    keys = samples_set[0].attributes_data.keys()
    attributes_dict = {}
    for key in keys:
        attributes_dict[key] = []
        for s in samples_set:
            if s.attributes_data[key] not in attributes_dict[key]:
                attributes_dict[key].append(s.attributes_data[key])
    return attributes_dict


def all_samples_are_classified(samples):
    """
    The function checks if all samples in a dataset are classified the same.

    Parameters:
       samples (list): Samples dataset.

    Returns:
        The classification of the class if they are classified, else returns 0.
    """
    # In case there are no samples - return from the function.
    if not samples:
        return
    # Default classification is the classification of the first sample.
    classification = samples[0].classification
    for sample in samples[1:]:
        if sample.classification != classification:
            return 0
    return classification

