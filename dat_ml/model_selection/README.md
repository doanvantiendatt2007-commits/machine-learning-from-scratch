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