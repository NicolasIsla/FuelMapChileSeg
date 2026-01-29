# FuelMap dataset (PASTIS-like) adapted for REGRESSION with 3 targets per pixel:
# target = stack([r_norm, H_norm, w_norm]) -> (3, H, W)
# plus mask (0/1) loaded from ANNOTATIONS_mask/{id}.npy

import json
import os
from datetime import datetime
from typing import Dict, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from einops import rearrange

from shapeft.datasets.base import RawGeoFMDataset, temporal_subsampling


def prepare_dates(date_dict, reference_date):
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
    Return format (PASTIS-like):
    {
      "image": { ... },
      "target":  (1,H,W) float32   # [r_norm, H_norm, w_norm]
      "mask":    (H,W) uint8/bool  # 0 invalid, 1 valid
      "metadata": (T,) long
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
        obj: str = "regression",
        # --- regression targets dirs (normalized) ---
        ann_dir: str = "r",
        mask_dir: str = "ANNOTATIONS_mask",
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
        self.modalities = ["S2", "S1_asc", "S1_des", "elevation", "mTPI", "landforms"]
        self.reference_date = datetime(*map(int, reference_date.split("-")))

        # target/mask folders (inside root_path)
        self.ann= f"ANNOTATIONS_{ann_dir}_norm"
        self.mask_dir = mask_dir

        # metadata.geojson
        self.meta_patch = gpd.read_file(os.path.join(self.root_path, "metadata.geojson"))
        if cover > 0:
            self.meta_patch = self.meta_patch[self.meta_patch["cover"] > cover].copy()

        if "Fold" not in self.meta_patch.columns:
            n = len(self.meta_patch)
            folds_auto = np.tile(np.arange(1, 6), n // 5 + 1)[:n]
            self.meta_patch["Fold"] = folds_auto

        self.meta_patch = pd.concat([self.meta_patch[self.meta_patch["Fold"] == f] for f in folds])
        self.meta_patch = self.meta_patch.sort_values("id").reset_index(drop=True)

    def __len__(self):
        return len(self.meta_patch)

    def _load_temporal(self, root_path: str, modality: str, name: Union[int, str]) -> torch.Tensor:
        path = os.path.join(root_path, f"DATA_{modality}", f"{name}.npy")
        arr = np.load(path)
        if arr.ndim != 4:
            raise ValueError(f"{modality} expected (T,C,H,W), got {arr.shape} at {path}")
        return torch.from_numpy(arr).to(torch.float32)  # (T,C,H,W)

    def _load_static(self, root_path: str, modality: str, name: Union[int, str]) -> torch.Tensor:
        path = os.path.join(root_path, f"DATA_{modality}", f"{name}.npy")
        arr = np.load(path)
        ten = torch.from_numpy(arr).to(torch.float32)

        if ten.ndim == 2:
            ten = ten.unsqueeze(0)  # (1,H,W)
        elif ten.ndim == 3:
            pass  # (C,H,W)
        else:
            raise ValueError(f"{modality} expected (H,W) or (C,H,W), got {arr.shape} at {path}")

        ten = ten.unsqueeze(1).repeat(1, self.multi_temporal, 1, 1)  # (C,T,H,W)
        return ten

    def _get_dates_from_row(self, row: pd.Series, sat: str) -> torch.Tensor:
        col = f"dates_{sat}"
        date_str = row.get(col, None)

        if date_str is None or (isinstance(date_str, float) and np.isnan(date_str)):
            return torch.empty((0,), dtype=torch.int32)

        if isinstance(date_str, str):
            date_list = [d for d in date_str.split(",") if d]
            offsets = [(datetime.strptime(d, "%Y-%m-%d") - self.reference_date).days for d in date_list]
            return torch.tensor(offsets, dtype=torch.int32)

        raise ValueError(f"Unsupported date format for {col}: {type(date_str)}")

    def _load_reg_targets_and_mask(self, name: Union[int, str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Loads:
          r_norm, H_norm, w_norm as (H,W) each, stacks -> (3,H,W)
          mask as (H,W) 0/1
        """
        reg_path = os.path.join(self.root_path, self.ann, f"{name}.npy")
        m_path = os.path.join(self.root_path, self.mask_dir,  f"{name}.npy")

        reg = np.load(reg_path).astype(np.float32)
        mask = np.load(m_path)

        if mask.ndim != 2:
            raise ValueError(f"mask expected (H,W), got {mask.shape} at {m_path}")

        # enforce 0/1
        mask = (mask > 0).astype(np.uint8)

        if reg.shape != mask.shape:
            raise ValueError(
                f"Target/mask shape mismatch for id={name}: "
                f"r{reg.shape} mask{mask.shape}"
            )

        target = np.stack([reg], axis=0)  # (1,H,W)

        target_t = torch.from_numpy(target).to(torch.float32)
        mask_t = torch.from_numpy(mask)  # uint8

        return target_t, mask_t

    def __getitem__(self, i: int) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:
        row = self.meta_patch.iloc[i]
        name = row["id"]

        # ---- TARGET (3,H,W) + MASK (H,W) ----
        target, mask = self._load_reg_targets_and_mask(name)

        # ---- Build output dict ----
        output: Dict[str, Union[torch.Tensor, int]] = {"name": int(name)}

        output["S2"] = self._load_temporal(self.root_path, "S2", name)
        output["S1_asc"] = self._load_temporal(self.root_path, "S1_asc", name)
        output["S1_des"] = self._load_temporal(self.root_path, "S1_des", name)

        output["elevation"] = self._load_static(self.root_path, "elevation", name)
        output["mTPI"] = self._load_static(self.root_path, "mTPI", name)
        output["landforms"] = self._load_static(self.root_path, "landforms", name)

        output["S2_dates"] = self._get_dates_from_row(row, "S2")
        output["S1_asc_dates"] = self._get_dates_from_row(row, "S1_asc")
        output["S1_des_dates"] = self._get_dates_from_row(row, "S1_des")

        # ---- Rearrange to (C,T,H,W) ----
        optical_ts = rearrange(output["S2"], "t c h w -> c t h w")
        sar_asc_ts = rearrange(output["S1_asc"], "t c h w -> c t h w")
        sar_desc_ts = rearrange(output["S1_des"], "t c h w -> c t h w")

        # ---- Temporal subsampling ----
        if self.multi_temporal == 1:
            idx = torch.tensor([-1], dtype=torch.long)
            optical_ts = optical_ts[:, idx]
            sar_asc_ts = sar_asc_ts[:, idx]
            sar_desc_ts = sar_desc_ts[:, idx]

            s2_dates = output["S2_dates"]
            metadata = s2_dates[idx].to(torch.long) if len(s2_dates) > 0 else torch.zeros((1,), dtype=torch.long)

        else:
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

            s2_dates = output["S2_dates"]
            if len(s2_dates) > 0:
                # align by the SAME indices as optical selection (defensive)
                max_valid = min(len(s2_dates), optical_ts.shape[1])
                metadata = s2_dates[:max_valid].to(torch.long)
                if metadata.numel() != optical_ts.shape[1]:
                    if metadata.numel() < optical_ts.shape[1]:
                        pad = optical_ts.shape[1] - metadata.numel()
                        metadata = torch.cat([metadata, metadata.new_full((pad,), int(metadata[-1]))])
                    else:
                        metadata = metadata[: optical_ts.shape[1]]
            else:
                metadata = torch.zeros((optical_ts.shape[1],), dtype=torch.long)

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
            "target": target,   # (1,H,W) float32
            "mask": mask,       # (H,W) uint8 (0/1)
            "metadata": metadata,
        }

    @staticmethod
    def download():
        pass