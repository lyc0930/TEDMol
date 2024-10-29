from pytorch_lightning import LightningDataModule
import torch_geometric
from data_provider.pretrain_dataset import GINPretrainDataset


class GINPretrainDataModule(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'dataset/MoLa-D',
        text_max_len: int = 112,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = GINPretrainDataset(root+'/pretrain/', text_max_len)

    def train_dataloader(self):
        loader = torch_geometric.loader.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True
        )
        return loader