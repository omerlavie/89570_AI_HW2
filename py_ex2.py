import read_file
import math
import knn
import naivebayes
import ID3

# import time
# K_FOLD_VAL = 5
KNN_K = 5


# FILE_PATH = "dataset.txt"


def divide_data_to_k_arrays(data, k):
    """
    The function divide a list into k lists.

    Parameters:
       data (list): The list to be divided.
       k (int): The number of parts the list should be divided to.

    Returns:
        The data divided into k equal length lists.
    """
    # After separation add the sub array of data into divided_data
    divided_data = []
    # The size of each container should be division_size.
    division_size = math.floor(len(data) / k)
    total_counter = 0
    # Separate data to k groups.
    for i in range(k):
        sub_data = []
        for j in range(division_size):
            sub_data.append(data[total_counter])
            total_counter += 1
        # If I'm in the last iteration and there is left over data we will add it to the last division.
        if i == k - 1:
            num_left = len(data) - total_counter
            for m in range(num_left):
                sub_data.append(data[total_counter + m])
        divided_data.append(sub_data)
    return divided_data


# Enter divided data and the sub-array you would like to get as test array.
def create_k_fold_set(data, m):
    """
    Enter divided data and the sub-array you would like to get as test array.

    Parameters:
       data (list): the data that should be divided to k parth.
       m (int): the index of the list that should be the testing set.

    Returns:
        The training set of size K_FOLD_VAL * k and testing set.
    """
    training_set = []
    test_set = data[m]
    j = 0
    for sub_data in data:
        if j != m:
            training_set += sub_data
        j += 1
    return training_set, test_set


if __name__ == '__main__':
    """
    This code create 3 AI models: Decision Tree, Naive Bayes and KNN on a dataset file and return an output.txt file
    as the tree build with an ID3 algorithm and the accuracy of the predictions using a test.txt file.
    """
    ############# CODE FOR OWN USE, TESTING AND PRODUCING ACCURACY.TXT FILE ##############
    # start_time = time.time()
    # samples = read_file.create_samples(FILE_PATH)
    # Divide the data into k lists for the k-fold cross validation.
    # divided_data = divide_data_to_k_arrays(samples, K_FOLD_VAL)
    # total_id3_acc = total_knn_acc = total_nb_acc = 0
    # temp_id3_acc = temp_knn_acc = temp_nb_acc = 0
    # for i in range(1):
    #     training_set, test_set = create_k_fold_set(divided_data, i)
    #     # ===== ID3 ======
    #     temp_acc_id3 = ID3.get_accuracy(training_set, test_set)
    #     total_id3_acc += temp_acc_id3
    #     print("training set size is: " + str(len(training_set)) + " and test set size is: " + str(len(test_set)))
    #     print("id3 accuracy: " + str(temp_acc_id3) + "\n")
    #     # ===== NAIVE BAYES ======
    #     temp_nb_acc = naivebayes.get_accuracy(training_set, test_set)
    #     total_nb_acc += temp_nb_acc
    #     print("training set size is: " + str(len(training_set)) + " and test set size is: " + str(len(test_set)))
    #     print("nb accuracy: " + str(temp_nb_acc) + "\n")
    #     # ========= KNN ==========
    #     temp_knn_acc = knn.get_accuracy(training_set, test_set, KNN_K)
    #     total_knn_acc += temp_knn_acc
    #     print("training set size is: " + str(len(training_set)) + " and test set size is: " + str(len(test_set)))
    #     print("knn accuracy: " + str(temp_knn_acc) + "\n")
    #
    # knn_acc = total_knn_acc / K_FOLD_VAL
    # id3_acc = total_id3_acc / K_FOLD_VAL
    # nb_acc = total_nb_acc / K_FOLD_VAL
    # f = open("accuracy.txt", "w+")
    # f.write(str(math.floor(id3_acc * 100) / 100) + "\t" +
    #           str(math.floor(knn_acc * 100) / 100) + "\t" + str(math.floor(nb_acc * 100) / 100))
    # f.close()
    # print("=== ID3 total accuracy: " + str(id3_acc) + " ===")
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print("=== NB total accuracy: " + str(nb_acc) + " ===")
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print("=== KNN total accuracy: " + str(knn_acc) + " ===")
    # print("--- %s seconds ---" % (time.time() - start_time))
    ######################################################################################
    train_set = read_file.create_samples("train.txt")
    test_set = read_file.create_samples("test.txt")
    id3_accuracy = ID3.get_accuracy(train_set, test_set)
    nb_accuracy = naivebayes.get_accuracy(train_set, test_set)
    knn_accuracy = knn.get_accuracy(train_set, test_set, KNN_K)
    f = open("output.txt", "a")
    f.write("\n" + str(math.floor(id3_accuracy * 100) / 100) + "\t" +
            str(math.floor(knn_accuracy * 100) / 100) + "\t" + str(math.floor(nb_accuracy * 100) / 100))
    f.close()
