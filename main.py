
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
                image.append(pixel)
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

def main():
    X_train = read_images(TRAIN_DATA_FILENAME, 100)
    y_train = read_labels(TRAIN_LABELS_FILENAME, 100)
    X_test = read_images(TEST_DATA_FILENAME, 100)
    y_test = read_labels(TEST_LABEL_FILENAME, 100)

    X_train = extract_features(X_train) # 28x28 image -> 1x784 list
    X_test = extract_features(X_test) # 28x28 image -> 1x784 list

    

if __name__ == '__main__':
    main()