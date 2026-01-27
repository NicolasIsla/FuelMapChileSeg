###
# FuelMap dataset in the same spirit as the PASTIS dataset implementation
# (OmniSat / PASTIS-HD style): build an intermediate `output` dict, then
# rearrange temporal modalities to (C,T,H,W), temporal subsampling, and
# return a structured dict with {"image":..., "target":..., "metadata":...}.
###

import json
import os
from datetime import datetime
from typing import Dict, Union, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from einops import rearrange

from shapeft.datasets.base import RawGeoFMDataset, temporal_subsampling


def prepare_dates(date_dict, reference_date):
    """Date formatting: dict/json -> tensor(days since ref)."""
    if isinstance(date_dict, str):
        date_dict = json.loads(date_dict)
    d = pd.DataFrame().from_dict(date_dict, orient="index")
    d = d[0].apply(
        lambda x: (
            datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
            - reference_date
        ).days
    )
    return torch.tensor(d.values)


class FuelMap(RawGeoFMDataset):
    """
    Multi-modal, multi-temporal dataset.

    Return format (PASTIS-like):
    {
      "image": {
        "optical":  (C,T,H,W) float32,
        "sar_asc":  (C,T,H,W) float32,
        "sar_desc": (C,T,H,W) float32,
        "elevation":(C,1,H,W) float32,   # static with T=1
        "mTPI":     (C,1,H,W) float32,   # static with T=1
        "landforms":(C,1,H,W) float32,   # static with T=1
      },
      "target":  (4,H,W) float32  (or torch tensor),
      "metadata": (T,) long  (S2 dates offsets aligned with optical selection),
      "mask": (H,W) or (1,H,W) bool/int (as stored; recommended bool)
    }
    """

    def __init__(
        self,
        split: str,
        dataset_name: str,
        multi_modal: bool,
        multi_temporal: int,
        root_path: str,
        classes: list,
        num_classes: int,
        ignore_index: int,
        img_size: int,
        bands: dict[str, list[str]],
        distribution: list[int],
        data_mean: dict[str, list[str]],
        data_std: dict[str, list[str]],
        data_min: dict[str, list[str]],
        data_max: dict[str, list[str]],
        download_url: str,
        auto_download: bool,
        reference_date: str = "2020-09-01",
        cover: int = 0,
        obj: str = "regresion",
        map_reg: str = "combustible_disponible",
    ):
        super().__init__(
            split=split,
            dataset_name=dataset_name,
            multi_modal=multi_modal,
            multi_temporal=multi_temporal,
            root_path=root_path,
            classes=classes,
            num_classes=num_classes,
            ignore_index=ignore_index,
            img_size=img_size,
            bands=bands,
            distribution=distribution,
            data_mean=data_mean,
            data_std=data_std,
            data_min=data_min,
            data_max=data_max,
            download_url=download_url,
            auto_download=auto_download,
        )

        assert split in ["train", "val", "test"], "Split must be train, val or test"
        if split == "train":
            folds = [1, 2, 3]
        elif split == "val":
            folds = [4]
        else:
            folds = [5]

        self.obj = obj
        self.map_reg = map_reg
        self.modalities = ["S2", "S1_asc", "S1_des", "elevation", "mTPI", "landforms"]

        self.reference_date = datetime(*map(int, reference_date.split("-")))

        # metadata.geojson
        self.meta_patch = gpd.read_file(os.path.join(self.root_path, "metadata_with_covers.geojson"))
        if cover >= 0:
            self.meta_patch = self.meta_patch[self.meta_patch[f"cover_{obj}"] > cover].copy()

        # folds if missing
        if "Fold" not in self.meta_patch.columns:
            n = len(self.meta_patch)
            folds_auto = np.tile(np.arange(1, 6), n // 5 + 1)[:n]
            self.meta_patch["Fold"] = folds_auto

        # filter by

        # apply folds
        self.meta_patch = pd.concat([self.meta_patch[self.meta_patch["Fold"] == f] for f in folds])
        self.meta_patch = self.meta_patch.sort_values("id").reset_index(drop=True)

        # Build date lookups (same idea, but we will select by indices like Pastis)
        # We keep raw lists in columns dates_S2 / dates_S1_asc / dates_S1_des.
        # get_dates() parses them per patch.
        # Nothing else needed here.

    def __len__(self):
        return len(self.meta_patch)

    def _load_temporal(self, root_path: str, modality: str, name: Union[int, str]) -> torch.Tensor:
        """
        Loads temporal array expected as (T,C,H,W) and returns torch float32.
        """
        path = os.path.join(root_path, f"DATA_{modality}", f"{name}.npy")
        arr = np.load(path)
        if arr.ndim != 4:
            raise ValueError(f"{modality} expected (T,C,H,W), got {arr.shape} at {path}")
        return torch.from_numpy(arr).to(torch.float32)  # (T,C,H,W)

    def _load_static(self, root_path: str, modality: str, name: Union[int, str]) -> torch.Tensor:
        """
        Loads static array expected as (H,W) or (C,H,W). Returns torch float32.
        Output shape: (C, T, H, W) where T = self.multi_temporal.
        """
        path = os.path.join(root_path, f"DATA_{modality}", f"{name}.npy")
        arr = np.load(path)
        ten = torch.from_numpy(arr).to(torch.float32)

        if ten.ndim == 2:
            # (H,W) -> (1,H,W)
            ten = ten.unsqueeze(0)
        elif ten.ndim == 3:
            # already (C,H,W)
            pass
        else:
            raise ValueError(f"{modality} expected (H,W) or (C,H,W), got {arr.shape} at {path}")

        # (C,H,W) -> (C,1,H,W) -> (C,T,H,W)
        ten = ten.unsqueeze(1).repeat(1, self.multi_temporal, 1, 1)
        return ten

    def _get_dates_from_row(self, row: pd.Series, sat: str) -> torch.Tensor:
        """
        Returns a tensor of day offsets for a given satellite (S2, S1_asc, S1_des),
        aligned with the temporal stack length for that modality.
        """
        col = f"dates_{sat}"
        date_str = row.get(col, None)

        if date_str is None or (isinstance(date_str, float) and np.isnan(date_str)):
            return torch.empty((0,), dtype=torch.int32)

        if isinstance(date_str, str):
            date_list = [d for d in date_str.split(",") if d]
            offsets = [
                (datetime.strptime(d, "%Y-%m-%d") - self.reference_date).days
                for d in date_list
            ]
            return torch.tensor(offsets, dtype=torch.int32)

        # fallback if stored as something else
        raise ValueError(f"Unsupported date format for {col}: {type(date_str)}")

    def __getitem__(self, i: int) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:
        row = self.meta_patch.iloc[i]
        name = row["id"]

        # ---- TARGET + MASK ----
        target_np = np.load(os.path.join(self.root_path, f"ANNOTATIONS_class_hier2_{self.obj}", f"{name}.npy"))

        # Convert to torch (recommended; if your pipeline expects numpy, remove these conversions)
        target = torch.from_numpy(target_np).to(torch.int64)
        mask_path = os.path.join(self.root_path, "ANNOTATIONS_masked", f"{name}_masked.npy")
        if os.path.exists(mask_path):
            mask_np = np.load(mask_path)
            mask = torch.from_numpy(mask_np)
        else:
            mask = None  # o máscara todo válida: torch.ones((H,W), dtype=torch.bool)

        # ---- Build output dict (PASTIS style) ----
        output: Dict[str, Union[torch.Tensor, int]] = {"name": int(name)}

        # Temporal modalities loaded as (T,C,H,W)
        output["S2"] = self._load_temporal(self.root_path, "S2", name)
        output["S1_asc"] = self._load_temporal(self.root_path, "S1_asc", name)
        output["S1_des"] = self._load_temporal(self.root_path, "S1_des", name)

        # Static modalities loaded as (C,1,H,W)
        output["elevation"] = self._load_static(self.root_path, "elevation", name)
        output["mTPI"] = self._load_static(self.root_path, "mTPI", name)
        output["landforms"] = self._load_static(self.root_path, "landforms", name)

        # Dates (offsets) for temporal modalities
        output["S2_dates"] = self._get_dates_from_row(row, "S2")
        output["S1_asc_dates"] = self._get_dates_from_row(row, "S1_asc")
        output["S1_des_dates"] = self._get_dates_from_row(row, "S1_des")

        # ---- Rearrange temporal to (C,T,H,W) ----
        optical_ts = rearrange(output["S2"], "t c h w -> c t h w")
        sar_asc_ts = rearrange(output["S1_asc"], "t c h w -> c t h w")
        sar_desc_ts = rearrange(output["S1_des"], "t c h w -> c t h w")

        # ---- Temporal subsampling (PASTIS logic) ----
        if self.multi_temporal == 1:
            # take last frame
            idx = torch.tensor([-1], dtype=torch.long)

            optical_ts = optical_ts[:, idx]
            sar_asc_ts = sar_asc_ts[:, idx]
            sar_desc_ts = sar_desc_ts[:, idx]

            # metadata aligned to S2 selection
            s2_dates = output["S2_dates"]
            metadata = s2_dates[idx].to(torch.long) if len(s2_dates) > 0 else torch.zeros((1,), dtype=torch.long)

        else:
            # Choose up to 35 evenly spaced indices, then subsample to multi_temporal
            def _select_indexes(T: int) -> torch.Tensor:
                max_steps = min(35, T)
                whole = torch.linspace(0, T - 1, steps=max_steps, dtype=torch.long)
                return temporal_subsampling(self.multi_temporal, whole)

            optical_idx = _select_indexes(optical_ts.shape[1])
            asc_idx = _select_indexes(sar_asc_ts.shape[1])
            des_idx = _select_indexes(sar_desc_ts.shape[1])

            optical_ts = optical_ts[:, optical_idx]
            sar_asc_ts = sar_asc_ts[:, asc_idx]
            sar_desc_ts = sar_desc_ts[:, des_idx]

            # metadata from S2 dates with SAME indices as optical selection
            s2_dates = output["S2_dates"]
            if len(s2_dates) > 0:
                # If date count differs from frames, align defensively by min length
                max_valid = min(len(s2_dates), optical_ts.shape[1])
                metadata = s2_dates[:max_valid].to(torch.long)
                if metadata.numel() != optical_ts.shape[1]:
                    # pad/truncate to match T
                    if metadata.numel() < optical_ts.shape[1]:
                        pad = optical_ts.shape[1] - metadata.numel()
                        metadata = torch.cat([metadata, metadata.new_full((pad,), int(metadata[-1]))])
                    else:
                        metadata = metadata[: optical_ts.shape[1]]
            else:
                metadata = torch.zeros((optical_ts.shape[1],), dtype=torch.long)

        # ---- Static tensors already (C,1,H,W) ----
        elev = output["elevation"]
        mtpi = output["mTPI"]
        land = output["landforms"]

        return {
            "image": {
                "optical": optical_ts.to(torch.float32),
                "sar_asc": sar_asc_ts.to(torch.float32),
                "sar_desc": sar_desc_ts.to(torch.float32),
                "elevation": elev.to(torch.float32),
                "mTPI": mtpi.to(torch.float32),
                "landforms": land.to(torch.float32),
            },
            "target": target,
            "metadata": metadata,
        }

    @staticmethod
    def download():
        pass