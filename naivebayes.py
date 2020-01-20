POSITIVE = "yes"    # Positive classification of a sample.
NEGATIVE = "no"     # Negative classification of a sample.


def get_prior_probability(training_set):
    """
    The function calculates the prior probability on an entire set.

    Parameters:
       training_set (samples): The training set the calculation is based on.

    Returns:
        The function returns prior probability on the training_set.
    """
    total_pos = total_neg = 0
    training_set_size = len(training_set)
    for sample in training_set:
        if sample.classification == POSITIVE:
            total_pos += 1
        else:
            total_neg += 1
    return (total_pos / training_set_size), total_neg / training_set_size


def conditional_probability(xi, y, training_set):
    """
    The function calculates the conditional probability on the training_set.

    Parameters:
       xi (str): A specific feature.
       y (str): Classification.
       training_set (samples): The training set.

    Returns:
        The function returns conditional probability on the training_set.
    """
    xi_sum = 0
    y_sum = 0
    for sample in training_set:
        if sample.classification == y:
            y_sum += 1
            if sample.attributes_data[xi[0]] == xi[1]:
                xi_sum += 1
    return xi_sum / y_sum


def predict_classification(predict_sample, training_set):
    """
    The function predict the classification of a sample based on the training_set.

    Parameters:
       predict_sample (sample): A sample.
       training_set (list): The training set the prediction is based on.

    Returns:
        The function returns the prediction of the sample.
    """
    prior_pos, prior_neg = get_prior_probability(training_set)
    conditional_prob_pos = 1
    conditional_prob_neg = 1
    for attribute in predict_sample.attributes_data:
        key = predict_sample.attributes_data[attribute]
        xi = (attribute, key)
        conditional_prob_pos *= conditional_probability(xi, POSITIVE, training_set)
        conditional_prob_neg *= conditional_probability(xi, NEGATIVE, training_set)
    probability_pos = prior_pos * conditional_prob_pos
    probability_neg = prior_neg * conditional_prob_neg
    if probability_pos > probability_neg:
        return POSITIVE
    else:
        return NEGATIVE


def get_accuracy(training_set, test_set):
    """
    The function calculates the accuracy of the dataset by building a tree using the training_set and getting
    predictions using the test_set.

    Parameters:
        training_set (list): List of samples to build the decision tree on.
        test_set (list): List of sampled to get the accuracy of the tree.

    Returns:
        The function returns the accuracy of the model on the testing_set.
    """
    # Counters for number of right and wrong classifications for each classification (POSITIVE, NEGATIVE)
    true_pos = true_neg = false_pos = false_neg = 0
    for t in test_set:
        t_prediction = predict_classification(t, training_set)
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
