from typing import Callable
import torch
import torch.nn.functional as F

def get_collate_fn(modalities: list[str]) -> Callable:
    def collate_fn(batch):
        # 1) T_max
        T_max = 0
        for modality in modalities:
            for x in batch:
                if len(x["image"][modality].shape) == 4:
                    T_max = max(T_max, x["image"][modality].shape[1])

        # 2) pad temporal
        for modality in modalities:
            for i, x in enumerate(batch):
                if len(x["image"][modality].shape) == 4:
                    T = x["image"][modality].shape[1]
                    if T < T_max:
                        padding = (0, 0, 0, 0, 0, T_max - T)
                        batch[i]["image"][modality] = F.pad(
                            x["image"][modality], padding, "constant", 0
                        )

        # 3) build out
        batch_out = {
            "image": {
                modality: torch.stack([x["image"][modality] for x in batch])
                for modality in modalities
            },
            "target": torch.stack([x["target"] for x in batch]),
            "metadata": torch.stack([x["metadata"] for x in batch]),
        }

        # 4) add mask if present
        if "mask" in batch[0]:
            # acepta (H,W) o (1,H,W); apila a (B,...) tal cual
            batch_out["mask"] = torch.stack([x["mask"] for x in batch])

        return batch_out

    return collate_fn