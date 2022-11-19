from collections import Sequence

from torch.utils.data import Sampler, SequentialSampler

from plenoxels.datasets.base_dataset import BaseDataset


class MultiSceneSampler(Sampler[int]):
    @staticmethod
    def cumsum(sequence):
        r, s = [0], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Sequence[BaseDataset], num_samples_per_dataset):
        super().__init__(datasets)
        self.datasets = list(datasets)
        self.num_samples_per_dataset = num_samples_per_dataset

        self.cumulative_sizes = self.cumsum(self.datasets)
        self.rnd_samplers = [SequentialSampler(d) for d in self.datasets]

    def __iter__(self):
        # The list of datasets which still have data in them. They are visited sequentially
        active_dsets = list(range(len(self.datasets)))
        active_dset_idx = 0
        # One iterator into the random sampler for each dataset. Data will be fetched from here.
        diters = [iter(s) for s in self.rnd_samplers]

        num_samples_active_dset = 0
        while True:
            try:
                # dataset sizes constantly change due to dynamic batch size
                self.cumulative_sizes = self.cumsum(self.datasets)
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
