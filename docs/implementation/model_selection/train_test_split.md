# Model Selection

## Train Test Split

### How It Works

`train_test_split` is used to divide a dataset into two subsets: a training set and a testing set.

My implementation is based on shuffling the indices rather than directly shuffling the original data.

The process consists of the following steps:

1. Determine the total number of samples in the dataset.
2. Create a list of indices corresponding to the position of each sample.
3. Use `random_state` as the seed for the pseudorandom number generator.
4. Shuffle the list of indices using `shuffle`.
5. Determine the number of samples for the training and testing sets based on `test_size`.
6. Split the shuffled indices into two groups.
7. Use the corresponding indices to retrieve the data and labels, creating `X_train`, `X_test`, `y_train`, and `y_test`.

This approach ensures that each input sample remains correctly paired with its corresponding label throughout the shuffling and splitting process.


```python
def train_test_split(X, y, test_size=0.2, random_state=42):
    if len(X) != len(y):
        raise ValueError("X and y do not belong to the same sample set!")
        # Check the sample size of the data
    if len(X) == 0 or len(y) == 0:
        raise ValueError("X or y is empty!")
        # Check if data exists
    if test_size <= 0 or test_size >= 1:
        raise ValueError("Invalid test_size value!")
        # Check if test_size is valid
    seed = random.Random(random_state)
    # Create a data shuffling seed based on random_state (ensures each data split will be identical for a given random_state)
    n = len(X)
    indices = list(range(n))
    # Create an index set consisting of all indices of the dataset
    seed.shuffle(indices)
    # Proceed to shuffle data based on seed and shuffle
    n_train = n - math.ceil(test_size * n)
    # Calculate the size of the training dataset
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    # Split the data index set into train and test
    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]
    # Create X_train, X_test, y_train, y_test sets from the split index sets
    return X_train, X_test, y_train, y_test
```