# 89570_AI_HW
Implementation of KNN, ID3 and Naive Bayes algorithms in homeworks #2 in Bar-Ilan university artificial intelligence course (89-570).

## How to run
The program reads the classified data from the ```train.txt``` using file in the relative directory and the classified data from ```test.txt``` to calculate the accuracy of the three algorithms. More detailed explenation about the data format is available in HW_2 directory.

## Output
The program is running the 3 algorithms and writes the decision tree created by the ID3 algorithm concated with the accuracy of the algorithms using a ```train.txt``` file and a ```test.txt``` into an ```output.txt``` file in the following format:
```
Decision_tree
\n
<ID3_accuracy>\t<KNN_accuracy>\t<NaiveBayes_accuracy>
```
