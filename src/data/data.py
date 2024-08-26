import numpy as np


class NumpyDataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NumpyDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, seed=1):
        if len(dataset.X) < batch_size:
            raise ValueError(f"Dataset needs at least {batch_size} elements")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        n = len(dataset.X)
        n_batches, residual = divmod(n, batch_size)
        total_idx = np.arange(n)
        self.idx = []
        for i in range(n_batches):
            if not shuffle:
                batch_idx = total_idx[:batch_size]
            else:
                np.random.seed(seed)
                batch_idx = np.random.choice(total_idx, size=batch_size, replace=False)
            self.idx.append(batch_idx)
            total_idx = np.setdiff1d(total_idx, batch_idx)
        if residual != 0:
            self.idx.append(total_idx)

        self.current = 0
        self.stop = len(self.idx)  # number of batches

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.stop:
            curr = self.current
            self.current += 1
            idx = self.idx[curr]
            return self.dataset[idx]
        else:
            raise StopIteration