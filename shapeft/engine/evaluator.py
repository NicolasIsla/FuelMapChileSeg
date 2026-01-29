import logging
import os
import time
from pathlib import Path
import math
import wandb

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import math


class Evaluator:
    """
    Evaluator class for evaluating the models.
    Attributes:
        val_loader (DataLoader): DataLoader for the validation dataset.
        exp_dir (str | Path): Directory for experiment outputs.
        device (torch.device): Device to run the evaluation on (e.g., CPU or GPU).
        use_wandb (bool): Flag to indicate if Weights and Biases (wandb) is used for logging.
        logger (logging.Logger): Logger for logging information.
        classes (list): List of class names in the dataset.
        split (str): Dataset split (e.g., 'train', 'val', 'test').
        ignore_index (int): Index to ignore in the dataset.
        num_classes (int): Number of classes in the dataset.
        max_name_len (int): Maximum length of class names.
        wandb (module): Weights and Biases module for logging (if use_wandb is True).
    Methods:
        __init__(val_loader: DataLoader, exp_dir: str | Path, device: torch.device, use_wandb: bool) -> None:
            Initializes the Evaluator with the given parameters.
        evaluate(model: torch.nn.Module, model_name: str, model_ckpt_path: str | Path | None = None) -> None:
            Evaluates the given model. This method should be implemented by subclasses.
        __call__(model: torch.nn.Module) -> None:
            Calls the evaluator on the given model.
        compute_metrics() -> None:
            Computes evaluation metrics. This method should be implemented by subclasses.
        log_metrics(metrics: dict) -> None:
            Logs the computed metrics. This method should be implemented by subclasses.
    """

    def __init__(
            self,
            val_loader: DataLoader,
            exp_dir: str | Path,
            device: torch.device,
            inference_mode: str = 'sliding',
            sliding_inference_batch: int = None,
            use_wandb: bool = False,
            dataset_name: str = 'sen1floods11'
    ) -> None:
        self.rank = int(os.environ["RANK"])
        self.val_loader = val_loader
        self.logger = logging.getLogger()
        self.exp_dir = exp_dir
        self.device = device
        self.inference_mode = inference_mode
        self.sliding_inference_batch = sliding_inference_batch
        self.classes = self.val_loader.dataset.classes
        self.split = self.val_loader.dataset.split
        self.ignore_index = self.val_loader.dataset.ignore_index
        self.num_classes = len(self.classes)
        self.max_name_len = max([len(name) for name in self.classes])
        self.dataset_name = dataset_name

        self.use_wandb = use_wandb

    def evaluate(
            self,
            model: torch.nn.Module,
            model_name: str,
            model_ckpt_path: str | Path | None = None,
    ) -> None:
        raise NotImplementedError

    def __call__(self, model):
        pass

    def compute_metrics(self):
        pass

    def log_metrics(self, metrics):
        pass

    @staticmethod
    def sliding_inference(model, img, input_size, output_shape=None, stride=None, max_batch=None):
        b, c, t, height, width = img[list(img.keys())[0]].shape

        if stride is None:
            h = int(math.ceil(height / input_size))
            w = int(math.ceil(width / input_size))
        else:
            h = math.ceil((height - input_size) / stride) + 1
            w = math.ceil((width - input_size) / stride) + 1

        h_grid = torch.linspace(0, height - input_size, h).round().long()
        w_grid = torch.linspace(0, width - input_size, w).round().long()
        num_crops_per_img = h * w

        for k, v in img.items():
            img_crops = []
            for i in range(h):
                for j in range(w):
                    img_crops.append(v[:, :, :, h_grid[i]:h_grid[i] + input_size, w_grid[j]:w_grid[j] + input_size])
            img[k] = torch.cat(img_crops, dim=0)

        pred = []
        max_batch = max_batch if max_batch is not None else b * num_crops_per_img
        batch_num = int(math.ceil(b * num_crops_per_img / max_batch))
        for i in range(batch_num):
            img_ = {k: v[max_batch * i: min(max_batch * i + max_batch, b * num_crops_per_img)] for k, v in img.items()}
            pred_ = model.forward(img_, output_shape=(input_size, input_size))
            pred.append(pred_)
        pred = torch.cat(pred, dim=0)
        pred = pred.view(num_crops_per_img, b, -1, input_size, input_size).transpose(0, 1)

        merged_pred = torch.zeros((b, pred.shape[2], height, width), device=pred.device)
        pred_count = torch.zeros((b, height, width), dtype=torch.long, device=pred.device)
        for i in range(h):
            for j in range(w):
                merged_pred[:, :, h_grid[i]:h_grid[i] + input_size,
                w_grid[j]:w_grid[j] + input_size] += pred[:, h * i + j]
                pred_count[:, h_grid[i]:h_grid[i] + input_size,
                w_grid[j]:w_grid[j] + input_size] += 1

        merged_pred = merged_pred / pred_count.unsqueeze(1)
        if output_shape is not None:
            merged_pred = F.interpolate(merged_pred, size=output_shape, mode="bilinear")

        return merged_pred

class SegEvaluator(Evaluator):
    """
    SegEvaluator is a class for evaluating segmentation models. It extends the Evaluator class and provides methods
    to evaluate a model, compute metrics, and log the results.
    Attributes:
        val_loader (DataLoader): DataLoader for the validation dataset.
        exp_dir (str | Path): Directory for saving experiment results.
        device (torch.device): Device to run the evaluation on.
        use_wandb (bool): Flag to indicate whether to use Weights and Biases for logging.
    Methods:
        evaluate(model, model_name='model', model_ckpt_path=None):
            Evaluates the given model on the validation dataset and computes metrics.
        __call__(model, model_name, model_ckpt_path=None):
            Calls the evaluate method. This allows the object to be used as a function.
        compute_metrics(confusion_matrix):
            Computes various metrics such as IoU, precision, recall, F1-score, mean IoU, mean F1-score, and mean accuracy
            from the given confusion matrix.
        log_metrics(metrics):
            Logs the computed metrics. If use_wandb is True, logs the metrics to Weights and Biases.
    """

    def __init__(
            self,
            val_loader: DataLoader,
            exp_dir: str | Path,
            device: torch.device,
            inference_mode: str = 'sliding',
            sliding_inference_batch: int = None,
            use_wandb: bool = False,
            dataset_name: str = 'PASTIS-HD',
            save_predictions: bool = False,
            save_logits: bool = False,          # OJO: puede ser ENORME
            save_targets: bool = True,
            preds_subdir: str = "predictions",
    ):
        super().__init__(val_loader, exp_dir, device, inference_mode, sliding_inference_batch, use_wandb, dataset_name)

        self.save_predictions = save_predictions
        self.save_logits = save_logits
        self.save_targets = save_targets

        self.preds_dir = Path(self.exp_dir) / preds_subdir / self.split / f"rank{self.rank}"
        if self.rank == 0:
            (Path(self.exp_dir) / preds_subdir / self.split).mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()
        self.preds_dir.mkdir(parents=True, exist_ok=True)

    def reshape_transform(self, tensor, height=15, width=15):
        # Reshape (batch, seq_len, embed_dim) -> (batch, embed_dim, height, width)
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    @torch.no_grad()
    def evaluate(self, model, model_name='model', model_ckpt_path=None):
        t = time.time()

        if model_ckpt_path is not None:
            model_dict = torch.load(model_ckpt_path, map_location=self.device, weights_only=False)
            model_name = os.path.basename(model_ckpt_path).split(".")[0]
            if "model" in model_dict:
                model.module.load_state_dict(model_dict["model"])
            else:
                model.module.load_state_dict(model_dict)

            self.logger.info(f"Loaded {model_name} for evaluation")
        model.eval()

        tag = f"Evaluating {model_name} on {self.split} set"
        confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), device=self.device
        )
        sample_counter = 0
        for batch_idx, data in enumerate(tqdm(self.val_loader, desc=tag)):
            image, target = data["image"], data["target"]
            image = {k: v.to(self.device) for k, v in image.items()}
            target = target.to(self.device)

            if self.inference_mode == "sliding":
                input_size = model.module.encoder.input_size
                if model.module.encoder.model_name != "utae_encoder":
                    logits = self.sliding_inference(model, image, input_size, output_shape=target.shape[-2:],
                                                    max_batch=self.sliding_inference_batch)
                else: 
                    logits = model(image, batch_positions=data["metadata"])
            elif self.inference_mode == "whole":
                if model.module.encoder.model_name != "utae_encoder":
                    logits = model(image, output_shape=target.shape[-2:])
                else: 
                    logits = model(image, batch_positions=data["metadata"])                    
            else:
                raise NotImplementedError((f"Inference mode {self.inference_mode} is not implemented."))
            if logits.shape[1] == 1:
                pred = (torch.sigmoid(logits) > 0.5).type(torch.int64).squeeze(dim=1)
            else:
                pred = torch.argmax(logits, dim=1)

            if self.save_predictions:
                B = pred.shape[0]

                for i in range(B):
                    sid = f"{batch_idx:06d}_{i:02d}_{sample_counter:08d}"
                    out_path = self.preds_dir / f"{sid}.npz"

                    out = {
                        "pred": pred[i].detach().cpu().to(torch.int16).numpy(),  # (H,W)
                    }

                    if self.save_targets:
                        out["target"] = target[i].detach().cpu().to(torch.int16).numpy()

                    if self.save_logits:
                        # logits: (C,H,W) - puede ser grande
                        out["logits"] = logits[i].detach().cpu().to(torch.float16).numpy()

                    np.savez_compressed(out_path, **out)

                sample_counter += B
            valid_mask = target != self.ignore_index
            pred, target = pred[valid_mask], target[valid_mask]

            count = torch.bincount(
                (pred * self.num_classes + target), minlength=self.num_classes ** 2
            )
            confusion_matrix += count.view(self.num_classes, self.num_classes)

        torch.distributed.all_reduce(
            confusion_matrix, op=torch.distributed.ReduceOp.SUM
        )
        print(confusion_matrix.cpu())
        metrics = self.compute_metrics(confusion_matrix.cpu())
        self.log_metrics(metrics)

        used_time = time.time() - t

        return metrics, used_time

    @torch.no_grad()
    def __call__(self, model, model_name, model_ckpt_path=None):
        return self.evaluate(model, model_name, model_ckpt_path)

    def compute_metrics(self, confusion_matrix):
        # Calculate IoU for each class
        intersection = torch.diag(confusion_matrix)
        union = confusion_matrix.sum(dim=1) + confusion_matrix.sum(dim=0) - intersection
        iou = (intersection / (union + 1e-6)) * 100

        # Calculate precision and recall for each class
        precision = intersection / (confusion_matrix.sum(dim=0) + 1e-6) * 100
        recall = intersection / (confusion_matrix.sum(dim=1) + 1e-6) * 100

        # Calculate F1-score for each class
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        # Calculate mean IoU, mean F1-score, and mean Accuracy
        miou = iou.mean().item()
        mf1 = f1.mean().item()
        macc = (intersection.sum() / (confusion_matrix.sum() + 1e-6)).item() * 100

        # Convert metrics to CPU and to Python scalars
        iou = iou.cpu()
        f1 = f1.cpu()
        precision = precision.cpu()
        recall = recall.cpu()

        # Prepare the metrics dictionary
        metrics = {
            "IoU": [iou[i].item() for i in range(self.num_classes)],
            "mIoU": miou,
            "F1": [f1[i].item() for i in range(self.num_classes)],
            "mF1": mf1,
            "mAcc": macc,
            "Precision": [precision[i].item() for i in range(self.num_classes)],
            "Recall": [recall[i].item() for i in range(self.num_classes)],
        }

        return metrics

    def log_metrics(self, metrics):
        def format_metric(name, values, mean_value):
            header = f"------- {name} --------\n"
            metric_str = (
                    "\n".join(
                        c.ljust(self.max_name_len, " ") + "\t{:>7}".format("%.3f" % num)
                        for c, num in zip(self.classes, values)
                    )
                    + "\n"
            )
            mean_str = (
                    "-------------------\n"
                    + "Mean".ljust(self.max_name_len, " ")
                    + "\t{:>7}".format("%.3f" % mean_value)
            )
            return header + metric_str + mean_str

        iou_str = format_metric("IoU", metrics["IoU"], metrics["mIoU"])
        f1_str = format_metric("F1-score", metrics["F1"], metrics["mF1"])

        precision_mean = torch.tensor(metrics["Precision"]).mean().item()
        recall_mean = torch.tensor(metrics["Recall"]).mean().item()

        precision_str = format_metric("Precision", metrics["Precision"], precision_mean)
        recall_str = format_metric("Recall", metrics["Recall"], recall_mean)

        macc_str = f"Mean Accuracy: {metrics['mAcc']:.3f} \n"

        self.logger.info(iou_str)
        self.logger.info(f1_str)
        self.logger.info(precision_str)
        self.logger.info(recall_str)
        self.logger.info(macc_str)

        if self.use_wandb and self.rank == 0:
            wandb.log(
                {
                    f"{self.split}_mIoU": metrics["mIoU"],
                    f"{self.split}_mF1": metrics["mF1"],
                    f"{self.split}_mAcc": metrics["mAcc"],
                    **{
                        f"{self.split}_IoU_{c}": v
                        for c, v in zip(self.classes, metrics["IoU"])
                    },
                    **{
                        f"{self.split}_F1_{c}": v
                        for c, v in zip(self.classes, metrics["F1"])
                    },
                    **{
                        f"{self.split}_Precision_{c}": v
                        for c, v in zip(self.classes, metrics["Precision"])
                    },
                    **{
                        f"{self.split}_Recall_{c}": v
                        for c, v in zip(self.classes, metrics["Recall"])
                    },
                }
            )




class RegEvaluator(Evaluator):
    """
    Regression evaluator with per-channel MAE / RMSE / R2.
    Expects dataloader batches with:
      data["image"]   : dict of tensors
      data["target"]  : (B,C,H,W) float
      data["mask"]    : (B,H,W) or (B,1,H,W), 1=valid
      data["metadata"]: optional (for utae_encoder)
    """

    def __init__(
        self,
        val_loader: DataLoader,
        exp_dir: str | Path,
        device: torch.device,
        inference_mode: str = "whole",           # sliding is supported but usually unnecessary if same size
        sliding_inference_batch: int | None = None,
        use_wandb: bool = False,
        dataset_name: str = "FuelMap",
        channel_names: list[str] | None = None,  # e.g. ["r","H","w"]
        save_predictions: bool = False,
        save_logits: bool = False,          # OJO: puede ser ENORME
        save_targets: bool = True,
        preds_subdir: str = "predictions",
    ):
        super().__init__(val_loader, exp_dir, device, inference_mode, sliding_inference_batch, use_wandb, dataset_name)
        self.channel_names = channel_names

        self.save_predictions = save_predictions
        self.save_logits = save_logits
        self.save_targets = save_targets

        self.preds_dir = Path(self.exp_dir) / preds_subdir / self.split / f"rank{self.rank}"
        if self.rank == 0:
            (Path(self.exp_dir) / preds_subdir / self.split).mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()
        self.preds_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_mask(self, mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        elif mask.ndim == 4:
            if mask.shape[1] != 1:
                raise ValueError(f"mask expected (B,1,H,W); got {tuple(mask.shape)}")
        else:
            raise ValueError(f"mask ndim unsupported: {mask.ndim}, shape={tuple(mask.shape)}")

        mask = mask.to(device=target.device)
        mask = (mask > 0.5)
        return mask  # bool, (B,1,H,W)

    def _infer(self, model, image, target, metadata=None):
        # mirrors SegEvaluator logic but returns regression preds
        if self.inference_mode == "sliding":
            input_size = model.module.encoder.input_size
            if model.module.encoder.model_name != "utae_encoder":
                pred = self.sliding_inference(
                    model, image, input_size,
                    output_shape=target.shape[-2:],
                    max_batch=self.sliding_inference_batch
                )
            else:
                pred = model(image, batch_positions=metadata)
        elif self.inference_mode == "whole":
            if model.module.encoder.model_name != "utae_encoder":
                pred = model(image, output_shape=target.shape[-2:])
            else:
                pred = model(image, batch_positions=metadata)
        else:
            raise NotImplementedError(f"Inference mode {self.inference_mode} is not implemented.")
        return pred

    @torch.no_grad()
    def evaluate(self, model, model_name="model", model_ckpt_path=None):
        t0 = time.time()

        if model_ckpt_path is not None:
            model_dict = torch.load(model_ckpt_path, map_location=self.device, weights_only=False)
            model_name = os.path.basename(model_ckpt_path).split(".")[0]
            if "model" in model_dict:
                model.module.load_state_dict(model_dict["model"])
            else:
                model.module.load_state_dict(model_dict)
            self.logger.info(f"Loaded {model_name} for evaluation")

        model.eval()

        # Accumulators (sum over valid pixels, globally reduced)
        sum_abs = None
        sum_sq = None
        sum_y = None
        sum_y2 = None
        sum_res2 = None
        n_valid = None

        tag = f"Evaluating {model_name} on {self.split} set"
        sample_counter = 0
        for _, data in enumerate(tqdm(self.val_loader, desc=tag)):
            image = {k: v.to(self.device) for k, v in data["image"].items()}
            target = data["target"].to(self.device).float()
            mask = data.get("mask", None)
            if mask is None:
                raise KeyError("Regression evaluation expects data['mask'] with 1=valid, 0=invalid.")
            mask = self._ensure_mask(mask.to(self.device), target)  # bool (B,1,H,W)

            metadata = data.get("metadata", None)
            if metadata is not None:
                metadata = metadata.to(self.device)

            pred = self._infer(model, image, target, metadata=metadata).float()  # (B,C,H,W)
            if self.save_predictions:
                B, C, H, W = pred.shape
                for i in range(B):
                    sid = f"{_:06d}_{i:02d}_{sample_counter:08d}"
                    out_path = self.preds_dir / f"{sid}.npz"

                    out = {
                        "pred": pred[i].detach().cpu().to(torch.float16).numpy(),     # (C,H,W)
                        "mask": mask[i].detach().cpu().to(torch.uint8).numpy(),       # (1,H,W) bool-ish
                    }
                    if self.save_targets:
                        out["target"] = target[i].detach().cpu().to(torch.float16).numpy()

                    if self.save_logits:
                        # en regresión "logits" == pred normalmente; solo guarda si realmente es distinto
                        out["logits"] = pred[i].detach().cpu().to(torch.float16).numpy()

                    np.savez_compressed(out_path, **out)

                sample_counter += B

            if pred.shape != target.shape:
                raise ValueError(f"pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")

            B, C, H, W = target.shape
            if self.channel_names is None:
                ch_names = [f"ch{c}" for c in range(C)]
            else:
                if len(self.channel_names) != C:
                    raise ValueError(f"channel_names length {len(self.channel_names)} != C {C}")
                ch_names = self.channel_names

            # broadcast mask to channels
            valid = mask.expand(-1, C, -1, -1)  # (B,C,H,W)
            eps = 1e-6

            diff = pred - target
            abs_diff = diff.abs()
            sq_diff = diff.pow(2)

            # initialize accumulators once we know C
            if sum_abs is None:
                sum_abs = torch.zeros((C,), device=self.device)
                sum_sq = torch.zeros((C,), device=self.device)
                sum_y = torch.zeros((C,), device=self.device)
                sum_y2 = torch.zeros((C,), device=self.device)
                sum_res2 = torch.zeros((C,), device=self.device)
                n_valid = torch.zeros((C,), dtype=torch.long, device=self.device)

            for c in range(C):
                v = valid[:, c]
                nv = v.sum().to(torch.long)
                if nv.item() == 0:
                    continue
                y = target[:, c][v]
                e_abs = abs_diff[:, c][v]
                e_sq = sq_diff[:, c][v]
                r2 = e_sq.sum()  # SSE (residual sum of squares)

                sum_abs[c] += e_abs.sum()
                sum_sq[c] += e_sq.sum()
                sum_y[c] += y.sum()
                sum_y2[c] += (y * y).sum()
                sum_res2[c] += r2
                n_valid[c] += nv

        # reduce across ranks
        for ten in [sum_abs, sum_sq, sum_y, sum_y2, sum_res2]:
            torch.distributed.all_reduce(ten, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(n_valid, op=torch.distributed.ReduceOp.SUM)

        # compute metrics
        eps = 1e-6
        C = int(n_valid.numel())
        mae = torch.zeros((C,), device="cpu")
        rmse = torch.zeros((C,), device="cpu")
        r2 = torch.zeros((C,), device="cpu")

        for c in range(C):
            n = float(n_valid[c].item())
            if n <= 0:
                mae[c] = float("nan")
                rmse[c] = float("nan")
                r2[c] = float("nan")
                continue

            mae_c = (sum_abs[c] / (n + eps)).detach().cpu()
            rmse_c = torch.sqrt(sum_sq[c] / (n + eps)).detach().cpu()

            # SST = sum(y^2) - (sum(y)^2)/n
            sst = (sum_y2[c] - (sum_y[c] * sum_y[c]) / (n + eps)).clamp_min(0.0)
            sse = sum_res2[c]
            r2_c = (1.0 - (sse / (sst + eps))).detach().cpu()

            mae[c] = mae_c
            rmse[c] = rmse_c
            r2[c] = r2_c

        # averages across channels (nan-safe)
        def _nanmean(x: torch.Tensor) -> float:
            x = x[torch.isfinite(x)]
            return x.mean().item() if x.numel() else float("nan")

        metrics = {
            "MAE": [mae[c].item() for c in range(C)],
            "RMSE": [rmse[c].item() for c in range(C)],
            "R2": [r2[c].item() for c in range(C)],
            "mMAE": _nanmean(mae),
            "mRMSE": _nanmean(rmse),
            "mR2": _nanmean(r2),
        }

        self.log_metrics(metrics)
        used_time = time.time() - t0
        return metrics, used_time

    @torch.no_grad()
    def __call__(self, model, model_name, model_ckpt_path=None):
        return self.evaluate(model, model_name, model_ckpt_path)

    def log_metrics(self, metrics: dict):
        # channel labels
        C = len(metrics["MAE"])
        if self.channel_names is None:
            ch_names = [f"ch{c}" for c in range(C)]
        else:
            ch_names = self.channel_names

        def fmt_block(title, values, mean_value):
            header = f"------- {title} --------\n"
            body = "\n".join(
                f"{name}\t{val:>10.6f}" if (val == val) else f"{name}\t{'nan':>10}"
                for name, val in zip(ch_names, values)
            )
            mean = f"\n-------------------\nMean\t{mean_value:>10.6f}" if (mean_value == mean_value) else "\n-------------------\nMean\tnan"
            return header + body + mean

        self.logger.info(fmt_block("MAE", metrics["MAE"], metrics["mMAE"]))
        self.logger.info(fmt_block("RMSE", metrics["RMSE"], metrics["mRMSE"]))
        self.logger.info(fmt_block("R2", metrics["R2"], metrics["mR2"]))

        if self.use_wandb and self.rank == 0 and wandb is not None:
            log_dict = {
                f"{self.split}_mMAE": metrics["mMAE"],
                f"{self.split}_mRMSE": metrics["mRMSE"],
                f"{self.split}_mR2": metrics["mR2"],
            }
            for name, v in zip(ch_names, metrics["MAE"]):
                log_dict[f"{self.split}_MAE_{name}"] = v
            for name, v in zip(ch_names, metrics["RMSE"]):
                log_dict[f"{self.split}_RMSE_{name}"] = v
            for name, v in zip(ch_names, metrics["R2"]):
                log_dict[f"{self.split}_R2_{name}"] = v

            wandb.log(log_dict)