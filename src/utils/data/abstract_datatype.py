try:
    from torch_geometric.data import LightningDataset
except ImportError:
    from torch_geometric.data.lightning import LightningDataset
from torch.utils.data import DataLoader

# pytorch_lightning explicitly imported for isinstance registration.
# PyG ≥ 2.5 switched its internal Lightning import to the ``lightning``
# package, but the rest of scgg / LUNA uses ``pytorch_lightning`` —
# they're functionally identical but different Python objects, so
# ``isinstance(dm, pytorch_lightning.LightningDataModule)`` returns
# False for an instance whose only Lightning parent is
# ``lightning.pytorch.LightningDataModule``. This bites Lightning's
# ``is_overridden`` ("Expected a parent" ValueError) during trainer
# setup. The fix is to add pytorch_lightning's class as an explicit
# secondary parent below. The two LightningDataModule implementations
# don't conflict in MRO because LightningDataset already provides
# all the dataloader hooks; pytorch_lightning's contribution here is
# purely the isinstance-registration ABC + a few no-op default
# methods.
import pytorch_lightning as _pl_for_isinstance


class AbstractDataModule(LightningDataset, _pl_for_isinstance.LightningDataModule):
    def __init__(self, cfg, train_dataset, val_dataset, test_dataset):
        super().__init__(
            train_dataset,
            val_dataset,
            test_dataset,
            batch_size=cfg.train.batch_size,
            num_workers=32,
            pin_memory=getattr(cfg.dataset, "pin_memory", False),
        )
        self.cfg = cfg

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, self.cfg.train.batch_size)

    def validation_dataloader(self):
        return self._create_dataloader(
            self.validation_dataset, self.cfg.validation.batch_size
        )

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset, self.cfg.test.batch_size)

    def _create_dataloader(self, dataset, batch_size):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=32,
            pin_memory=True,
            collate_fn=self.collate,
            multiprocessing_context="fork",
        )


class AbstractDatasetInfos:
    def __init__(self, cfg):
        self.cfg = cfg
        self.statistics = None
        self.num_cell_class = None
        self.cell_class_decoder = None
        self.num_genes = None


class Statistics:
    def __init__(
        self,
        num_cell_class,
        num_genes,
        num_cell_to_region_mapping_dict,
        cell_class_decoder,
    ):
        self.num_cell_class = num_cell_class
        self.cell_class_decoder = cell_class_decoder
        self.num_cell_to_region_mapping_dict = num_cell_to_region_mapping_dict
        self.num_genes = num_genes
