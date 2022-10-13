from torch.utils.data import Sampler, RandomSampler


class MultiSceneSampler(Sampler[int]):
    @staticmethod
    def cumsum(sequence):
        r, s = [0], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets, num_samples_per_dataset, generator=None):
        super().__init__(datasets)
        self.datasets = list(datasets)
        self.num_samples_per_dataset = num_samples_per_dataset

        self.cumulative_sizes = self.cumsum(self.datasets)
        self.rnd_samplers = [RandomSampler(d, generator=generator) for d in self.datasets]

    def __iter__(self):
        # The list of datasets which still have data in them. They are visited sequentially
        active_dsets = list(range(len(self.datasets)))
        active_dset_idx = 0
        # One iterator into the random sampler for each dataset. Data will be fetched from here.
        diters = [iter(s) for s in self.rnd_samplers]

        num_samples_active_dset = 0
        while True:
            try:
                yield next(diters[active_dsets[active_dset_idx]]) + self.cumulative_sizes[active_dsets[active_dset_idx]]
                num_samples_active_dset += 1
                if num_samples_active_dset == self.num_samples_per_dataset:
                    # Reset the active dataset
                    active_dset_idx = (active_dset_idx + 1) % len(active_dsets)
                    num_samples_active_dset = 0
            except StopIteration:
                active_dsets.pop(active_dset_idx)
                if len(active_dsets) == 0:  # Exit while(True) loop
                    break
                # Reset the active dataset
                active_dset_idx = (active_dset_idx + 1) % len(active_dsets)
                num_samples_active_dset = 0

    def __len__(self):
        return self.cumulative_sizes[-1]



class MultiSceneSampler(Sampler[int]):
    @staticmethod
    def cumsum(sequence):
        r, s = [0], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets, num_samples_per_dataset, generator=None):
        super().__init__(datasets)
        self.datasets = list(datasets)
        self.num_samples_per_dataset = num_samples_per_dataset

        self.cumulative_sizes = self.cumsum(self.datasets)
        self.rnd_samplers = [RandomSampler(d, generator=generator) for d in self.datasets]

    def __iter__(self):
        # The list of datasets which still have data in them. They are visited sequentially
        active_dsets = list(range(len(self.datasets)))
        active_dset_idx = 0
        # One iterator into the random sampler for each dataset. Data will be fetched from here.
        diters = [iter(s) for s in self.rnd_samplers]

        num_samples_active_dset = 0
        while True:
            try:
                yield next(diters[active_dsets[active_dset_idx]]) + self.cumulative_sizes[active_dsets[active_dset_idx]]
                num_samples_active_dset += 1
                if num_samples_active_dset == self.num_samples_per_dataset:
                    # Reset the active dataset
                    active_dset_idx = (active_dset_idx + 1) % len(active_dsets)
                    num_samples_active_dset = 0
            except StopIteration:
                active_dsets.pop(active_dset_idx)
                if len(active_dsets) == 0:  # Exit while(True) loop
                    break
                # Reset the active dataset
                active_dset_idx = (active_dset_idx + 1) % len(active_dsets)
                num_samples_active_dset = 0

    def __len__(self):
        return self.cumulative_sizes[-1]


class MultiSceneBatchSampler(Sampler[int]):
    @staticmethod
    def cumsum(sequence):
        r, s = [0], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets, batch_size, num_batches_per_dataset, drop_last: bool, generator=None):
        super().__init__(datasets)
        self.datasets = list(datasets)
        self.batch_size = batch_size
        self.num_batches_per_dataset = num_batches_per_dataset
        self.drop_last = drop_last

        self.cumulative_sizes = self.cumsum(self.datasets)
        self.rnd_samplers = [RandomSampler(d, generator=generator) for d in self.datasets]

    def __iter__(self):
        # The list of datasets which still have data in them. They are visited sequentially
        active_dsets = list(range(len(self.datasets)))
        active_dset_idx = 0
        # One iterator into the random sampler for each dataset. Data will be fetched from here.
        diters = [iter(s) for s in self.rnd_samplers]

        batch = [0] * self.batch_size
        idx_in_batch = 0
        num_batches_active_dset = 0
        while True:
            try:
                batch[idx_in_batch] = next(diters[active_dsets[active_dset_idx]]) + self.cumulative_sizes[active_dsets[active_dset_idx]]
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    # Reset batch for new scene
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
                    num_batches_active_dset += 1
                if num_batches_active_dset == self.num_batches_per_dataset:
                    # We have just yielded so batch is empty (don't need to reset)
                    # Reset the active dataset
                    active_dset_idx = (active_dset_idx + 1) % len(active_dsets)
                    num_batches_active_dset = 0
            except StopIteration:
                if idx_in_batch > 0 and not self.drop_last:
                    yield batch[:idx_in_batch]
                active_dsets.pop(active_dset_idx)
                if len(active_dsets) == 0:  # Exit while(True)
                    break
                # Reset the active dataset
                active_dset_idx = (active_dset_idx + 1) % len(active_dsets)
                num_batches_active_dset = 0
                # Reset batch for new scene
                idx_in_batch = 0
                batch = [0] * self.batch_size

    def __len__(self):
        if self.drop_last:
            return sum(len(d) // self.batch_size for d in self.datasets)
        else:
            return sum((len(d) + self.batch_size - 1) // self.batch_size for d in self.datasets)

