
DATA_DIR = 'data/'
TEST_DATA_FILENAME = DATA_DIR + 't10k-images-idx3-ubyte'
TEST_LABEL_FILENAME = DATA_DIR + 't10k-labels-idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + 'train-images-idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + 'train-labels-idx1-ubyte'

def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')

def read_images(filename, n_max_images = None):
    images = []
    with open(filename, 'rb') as f:
        _ = f.read(4) # Magic number
        number_of_images = bytes_to_int(f.read(4))
        if n_max_images:
            number_of_images = n_max_images
        number_of_rows = bytes_to_int(f.read(4))
        number_of_columns = bytes_to_int(f.read(4))
        for image_idx in range(number_of_images):
            image = []
            for row_idx in range(number_of_rows):
                row = []
                for column_idx in range(number_of_columns):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images

def read_labels(filename, n_max_labels = None):
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4) # Magic number
        number_of_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            number_of_labels = n_max_labels
        for label_idx in range(number_of_labels):
            label = f.read(1)
            labels.append(label)
    return labels

def flatten_list(l): # flattening the images to a list of
    return [pixel for sublist in l for pixel in sublist] # 1 row and 784 col

def extract_features(X):
    return [flatten_list(sample) for sample in X]

def dist(x, y):
    return sum(
        [
            (bytes_to_int(x_i) - bytes_to_int(y_i)) ** 2 
            for x_i, y_i in zip(x, y)
        ]
    )**(0.5)

def get_training_distances_for_test_sample(X_train, test_sample):
    return [dist(train_sample, test_sample) for train_sample in X_train]

def Knn_function(X_train, y_train, X_test, y_test, k=3):
    y_predict = []
    for test_sample_idx, test_sample in enumerate(X_test):
        training_distances = get_training_distances_for_test_sample(X_train, test_sample)
        sorted_dist_indices = [
            pair[0]
            for pair in sorted(
                enumerate(training_distances),
                key=lambda x: x[1]
            )
        ]
        candidates = [
            bytes_to_int(y_train[idx])
            for idx in sorted_dist_indices[:k]
        ]
        print(f"Point is {bytes_to_int(y_test[test_sample_idx])} and we guest {candidates}")
        y_sample = 5
        y_predict.append(y_sample)
    return y_predict

def main():
    X_train = read_images(TRAIN_DATA_FILENAME)
    y_train = read_labels(TRAIN_LABELS_FILENAME)
    X_test = read_images(TEST_DATA_FILENAME, 5)
    y_test = read_labels(TEST_LABEL_FILENAME)

    X_train = extract_features(X_train) # 28x28 image -> 1x784 list
    X_test = extract_features(X_test) # 28x28 image -> 1x784 list

    result = Knn_function(X_train, y_train, X_test, y_test,3)

if __name__ == '__main__':
    main()