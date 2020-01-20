POSITIVE = "yes"    # Positive classification of a sample.
NEGATIVE = "no"     # Negative classification of a sample.


def get_k_nearest_neighbours(sample, training_set, k):
    """
    The function get the k nearest neighbours of a sample.

    Parameters:
        sample (Sample): A sample.
        training_set (list): List of samples in the training set.
        k (int): Number of k nearest neighbours to look for in the training set.

    Returns:
        The function returns the k nearest neighbours to the sample.
    """
    neighbours_list = training_set
    # Calculate the distance from each node to the sample node.
    for neighbour in training_set:
        neighbour.set_distance_to(sample)
    neighbours_list.sort()
    # Return the k nearest neighbours of the sample node.
    return neighbours_list[:k]


def get_accuracy(training_set, test_set, k):
    """
    The function calculates the accuracy of the dataset by building a tree using the training_set and getting
    predicions using the test_set.

    Parameters:
        training_set (list): List of samples to build the decision tree on.
        test_set (list): List of sampled to get the accuracy of the tree.
        k (int): k to use in the KNN algorithm.

    Returns:
        The function returns the accuracy of the model on the testing_set.
    """
    # Counters for number of right and wrong classifications for each classification (POSITIVE, NEGATIVE)
    true_pos = true_neg = false_pos = false_neg = 0
    for t in test_set:
        t_prediction = predict_classification(t, training_set, k)
        # Right classification.
        if t_prediction == t.classification:
            if t_prediction == POSITIVE:
                true_pos += 1
            else:
                true_neg += 1
        # Wrong classification.
        else:
            if t_prediction == POSITIVE:
                false_pos += 1
            else:
                false_neg += 1
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_neg + false_pos)
    return accuracy


def predict_classification(predict_sample, training_set, k):
    """
    The function predict the classification of a sample.

    Parameters:
        predict_sample (Sample): The sample to predict.
        training_set (list): training set to build the model on.
        k (int): The k to use in the KNN algorithm.

    Returns:
        The function returns the prediction of a sample based on the k given.
    """
    positive = negative = 0
    knn = get_k_nearest_neighbours(predict_sample, training_set, k)
    # Count the classification for each neighbour from the k nearest neighbours.
    for i in range(len(knn)):
        if knn[i].classification == POSITIVE:
            positive += 1
        else:
            negative += 1
    # Return the classification the the predict_sample by the most common classification of the k nearest neighbours.
    if positive > negative:
        return POSITIVE
    else:
        return NEGATIVE
