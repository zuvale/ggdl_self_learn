from torch_geometric.data import Dataset, InMemoryDataset
from torch_geometric.transforms import Compose as g_Compose


class ProcessedInMemoryDataset(InMemoryDataset):
    def __init__(
        self, base_dataset: Dataset, processors=None, clone: bool=True
    ) -> None:
        self.base_dataset = base_dataset
        self.processors = g_Compose(processors or [])
        self.clone = clone

        super().__init__(root=None, transform=None)

        data_list = []
        for data in self.base_dataset:
            if self.clone:
                data = data.clone()

            data = self.processors(data)
            data_list.append(data)

        self._data, self.slices = self.collate(data_list)
    
    @property
    def data(self):
        return self._data