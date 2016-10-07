import time
import numpy as np
from ocr import OCRNeuralNetwork
from sklearn.cross_validation import train_test_split

def test(data_matrix, data_labels, test_indices, nn):
    # avg_sum = 0
    # for j in range(100):
    correct_guess_count = 0
    for i in test_indices:
        test = data_matrix[i]
        prediction = nn.predict(test)
        if data_labels[i] == prediction:
            correct_guess_count += 1
    return correct_guess_count * 1.0 / len(test_indices)
    # avg_sum += (correct_guess_count * 1.0 / len(test_indices))
    # return avg_sum
    # return avg_sum / 100

# Load data
if __name__ == '__main__':
    data_matrix = np.loadtxt(open('data.csv', 'r'), delimiter = ',')
    data_labels = np.loadtxt(open('dataLabels.csv', 'r'))
    data_matrix = data_matrix.tolist()
    data_labels = data_labels.tolist()

    print("Performance")
    print("-----------")

    train_indices, test_indices = train_test_split(range(5000))
    # train_indices = test_indices = list(range(5000))
    test_indices = list(range(5000))
    for i in range(5, 150, 5):
        start_time = time.time()
        nn = OCRNeuralNetwork(i, data_matrix, data_labels, train_indices, False, False)
        used_time = time.time() - start_time
        start_time = time.time()
        Performance = str(test(data_matrix, data_labels, test_indices, nn))

        print("{i} Hidden Nodes: {val} \t Time: {time}".format(i=i, val=Performance, time=used_time))