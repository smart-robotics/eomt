import json
from pathlib import Path
from typing import Union, Callable, Optional
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import tv_tensors
from pycocotools import mask as coco_mask
import torch
from PIL import Image
import logging
from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms


# We map all custom categories to class 0 since classification is irrelevant.
# The base model requires background as the last class (num_classes).
CLASS_MAPPING = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
}

logger = logging.getLogger(__name__)
MIN_POLYGON_COORDS = 6


def _validate_rle(rle: dict) -> bool:
    """Check that an RLE dict has the required keys and non-empty counts."""
    return (
        isinstance(rle, dict)
        and "counts" in rle
        and "size" in rle
        and rle.get("counts") not in (None, "", b"", [])
    )


def _validate_polygons(segmentation: list, width: int, height: int) -> tuple[bool, str]:
    """Validate a list-of-polygons segmentation. Returns (is_valid, reason_if_invalid)."""
    if not segmentation:
        return False, "segmentation list is empty"

    for poly in segmentation:
        if not isinstance(poly, (list, tuple)) or len(poly) == 0:
            return False, f"polygon is empty or not a list: {poly!r}"
        if len(poly) < MIN_POLYGON_COORDS:
            return False, (
                f"polygon has only {len(poly)} coordinate values "
                f"(need at least {MIN_POLYGON_COORDS})"
            )
        if len(poly) % 2 != 0:
            return False, f"polygon has an odd number of coordinates: {len(poly)}"

    return True, ""

class SmartRoboticsDataset(TorchDataset):
    def __init__(
        self,
        root_dir: Path,
        split: str,
        transforms: Optional[Callable] = None,
        target_parser: Optional[Callable] = None,
        sub_dirs: Optional[list] = None,
        debug: bool = False,
        debug_per_dataset: int = 6,
    ):
        """
        Args:
            root_dir:   Root folder containing product sub-folders.
            split:      Dataset split, e.g. "train" or "val".
            transforms: Optional image/target transforms.
            target_parser: Callable that converts raw annotations to tensors.
            sub_dirs:   If given, only these named sub-folders are read from
                        ``root_dir``.  Useful when the root contains many
                        product folders and only a subset is wanted.
                        Example: ["coolblue_2024_12", "coolblue_2025_01"]
                        A ``FileNotFoundError`` is raised for any name that
                        does not exist under ``root_dir``.
                        When ``None`` (default) every sub-folder is scanned.
        """
        self.transforms = transforms
        self.target_parser = target_parser

        # These aggregate data across ALL selected sub-folders
        self.images = []
        self.labels_by_id = {}
        self.polygons_by_id = {}
        self.is_crowd_by_id = {}
        self.img_dir_by_filename = {}

        self.debug = debug
        self.debug_per_dataset = debug_per_dataset
        # Track how many debug images saved per dataset_name
        self.debug_counts: dict[str, int] = {}

        root = Path(root_dir)
        if sub_dirs is not None:
            # Only use the explicitly whitelisted folder names
            candidates = []
            for name in sub_dirs:
                sub = root / name
                if not sub.is_dir():
                    raise FileNotFoundError(
                        f"Whitelisted sub-folder '{name}' not found under '{root}'. "
                        f"Available folders: {[d.name for d in sorted(root.iterdir()) if d.is_dir()]}"
                    )
                candidates.append(sub)
        else:
            # Original behaviour: scan every sub-folder
            candidates = sorted(sub for sub in root.iterdir() if sub.is_dir())

        for sub in candidates:

            img_dir = sub / "rgb"
            splits_dir = sub / "manual_split"

            if not img_dir.exists():
                continue

            json_files = []
            if splits_dir.exists():
                json_files = sorted(splits_dir.glob(f"{split}_*.json"))

            if not json_files:
                # No manual_split folder or no matching split JSONs.
                # Fall back to the known annotations.json at the sub-folder root.
                fallback_json = sub / "annotations.json"
                if fallback_json.exists():
                    logger.warning(
                        "[%s] No manual_split/%s_*.json found. "
                        "Falling back to annotations.json — "
                        "ALL images and annotations will be used for the '%s' split and "
                        "metrics will be computed over every annotated image in this dataset.",
                        sub.name,
                        split,
                        split,
                    )
                    with open(fallback_json) as f:
                        data = json.load(f)

                    id_to_filename = {
                        img["id"]: Path(img["file_name"]).name
                        for img in data["images"]
                    }

                    for img in data["images"]:
                        fname = Path(img["file_name"]).name
                        self.images.append(img)
                        self.img_dir_by_filename[fname] = img_dir
                        self.labels_by_id.setdefault(fname, {})
                        self.polygons_by_id.setdefault(fname, {})
                        self.is_crowd_by_id.setdefault(fname, {})

                    for ann in data.get("annotations", []):
                        fname = id_to_filename[ann["image_id"]]
                        ann_id = ann["id"]
                        self.labels_by_id[fname][ann_id] = ann["category_id"]
                        self.polygons_by_id[fname][ann_id] = ann["segmentation"]
                        self.is_crowd_by_id[fname][ann_id] = bool(ann["iscrowd"])
                else:
                    # No annotations anywhere — load images only (inference-only fallback).
                    logger.warning(
                        "[%s] No manual_split/%s_*.json and no annotations.json found. "
                        "Loading all images from rgb/ with empty annotations for the '%s' split. "
                        "No ground-truth is available, so metrics will not be meaningful.",
                        sub.name,
                        split,
                        split,
                    )
                    for img_path in sorted(img_dir.iterdir()):
                        if img_path.is_file() and img_path.suffix.lower() in (
                            ".png", ".jpg", ".jpeg"
                        ):
                            fname = img_path.name
                            self.images.append({"file_name": fname})
                            self.img_dir_by_filename[fname] = img_dir
                            self.labels_by_id.setdefault(fname, {})
                            self.polygons_by_id.setdefault(fname, {})
                            self.is_crowd_by_id.setdefault(fname, {})
            else:
                for json_file in json_files:
                    with open(json_file) as f:
                        data = json.load(f)

                    id_to_filename = {img["id"]: Path(img["file_name"]).name for img in data["images"]}

                    for img in data["images"]:
                        fname = Path(img["file_name"]).name
                        self.images.append(img)
                        self.img_dir_by_filename[fname] = img_dir
                        self.labels_by_id.setdefault(fname, {})
                        self.polygons_by_id.setdefault(fname, {})
                        self.is_crowd_by_id.setdefault(fname, {})

                    for ann in data.get("annotations", []):
                        fname = id_to_filename[ann["image_id"]]
                        ann_id = ann["id"]
                        self.labels_by_id[fname][ann_id] = ann["category_id"]
                        self.polygons_by_id[fname][ann_id] = ann["segmentation"]
                        self.is_crowd_by_id[fname][ann_id] = bool(ann["iscrowd"])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        img_info = self.images[index]
        img_filename = Path(img_info["file_name"]).name
        img_dir = self.img_dir_by_filename[img_filename]
        dataset_name = img_dir.parent.name

        img = tv_tensors.Image(Image.open(img_dir / img_filename).convert("RGB"))

        masks, labels, is_crowd = self.target_parser(
            polygons_by_id=self.polygons_by_id.get(img_filename, {}),
            labels_by_id=self.labels_by_id.get(img_filename, {}),
            is_crowd_by_id=self.is_crowd_by_id.get(img_filename, {}),
            width=img.shape[-1],
            height=img.shape[-2],
        )

        if not masks:
            target = {
                "masks": tv_tensors.Mask(torch.empty((0, img.shape[-2], img.shape[-1]), dtype=torch.bool)),
                "labels": torch.empty((0,), dtype=torch.long),
                "is_crowd": torch.empty((0,), dtype=torch.bool),
                "img_path": str(img_dir / img_filename),
                "dataset_name": dataset_name, # Added metadata
            }
        else:
            target = {
                "masks": tv_tensors.Mask(torch.stack(masks)),
                "labels": torch.tensor(labels, dtype=torch.long),
                "is_crowd": torch.tensor(is_crowd, dtype=torch.bool),
                "img_path": str(img_dir / img_filename),
                "dataset_name": dataset_name, # Added metadata
            }

        if self.transforms is not None:
        # Decide whether to save a debug comparison for this image
            should_debug = False
            if self.debug:
                count = self.debug_counts.get(dataset_name, 0)
                if count < self.debug_per_dataset:
                    should_debug = True
                    self.debug_counts[dataset_name] = count + 1

            img, target = self.transforms(
                img, target,
                img_idx=index,
                debug=should_debug,
            )

        return img, target


class SmartRoboticsInstance(LightningDataModule):
    # ---------------------------------------------------------------------------
    # Hardcoded whitelist of sub-folder names to load from `path`.
    # Set to None to load ALL sub-folders (original behaviour).
    # Override this list to restrict which product datasets are used.
    # ---------------------------------------------------------------------------
    DEFAULT_SUB_DIRS: Optional[list] = [
        #"coolblue_2024_12"
    ]

    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 4,
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = 80,
        color_jitter_enabled=False,
        brightness: Optional[float] = None,
        contrast: Optional[float] = None,
        saturation: Optional[float] = None,
        hue: Optional[float] = None,
        scale_range=(0.1, 2.0),
        check_empty_targets=True,
        sub_dirs: Optional[list] = None,
        debug_augmentations: bool = False,
        debug_augmentations_per_dataset: int = 6,
    ) -> None:
        """
        Args:
            path:       Root folder that contains the product sub-folders
                        (e.g. /var/data/datasets).
            sub_dirs:   Explicit list of sub-folder names to use.  When
                        ``None`` (default) ``DEFAULT_SUB_DIRS`` is used.
                        Pass an empty list ``[]`` to scan every sub-folder.
        """
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        # Resolve which sub-folders to use:
        #   - explicit argument wins
        #   - fall back to class-level DEFAULT_SUB_DIRS
        #   - passing [] disables filtering (scan everything)
        self.sub_dirs = sub_dirs if sub_dirs is not None else self.DEFAULT_SUB_DIRS
        self.save_hyperparameters(ignore=["_class_path"])

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            max_brightness_delta=brightness * 255 if brightness is not None else 32,
            max_contrast_factor=contrast if contrast is not None else 0.5,
            saturation_factor=saturation if saturation is not None else 0.5,
            max_hue_delta=hue * 360 if hue is not None else 18,
            scale_range=scale_range,
        )

        self.debug_augmentations = debug_augmentations
        self.debug_augmentations_per_dataset = debug_augmentations_per_dataset

    @staticmethod
    def target_parser(
        polygons_by_id: dict[int, list[list[float]]],
        labels_by_id: dict[int, int],
        is_crowd_by_id: dict[int, bool],
        width: int,
        height: int,
        **kwargs,
    ):
        masks, labels, is_crowd = [], [], []

        for label_id, cls_id in labels_by_id.items():

            # 1. Unknown category
            if cls_id not in CLASS_MAPPING:
                logger.debug(
                    "Skipping annotation id=%s: category_id=%s not in CLASS_MAPPING.",
                    label_id, cls_id,
                )
                continue

            segmentation = polygons_by_id[label_id]

            # 2. None segmentation
            if segmentation is None:
                logger.warning("Skipping annotation id=%s: segmentation is None.", label_id)
                continue

            try:
                if isinstance(segmentation, dict):
                    # 3. RLE path
                    if not _validate_rle(segmentation):
                        logger.warning(
                            "Skipping annotation id=%s: invalid RLE dict: %s",
                            label_id, segmentation,
                        )
                        continue
                    rle = segmentation

                else:
                    # 4. Polygon path
                    valid, reason = _validate_polygons(segmentation, width, height)
                    if not valid:
                        logger.warning("Skipping annotation id=%s: %s", label_id, reason)
                        continue

                    rles = coco_mask.frPyObjects(segmentation, height, width)
                    rle = coco_mask.merge(rles) if isinstance(rles, list) else rles

                # 5. Zero-area mask
                decoded = coco_mask.decode(rle)
                if decoded.max() == 0:
                    logger.warning(
                        "Skipping annotation id=%s: decoded mask is entirely empty (zero area).",
                        label_id,
                    )
                    continue

                masks.append(tv_tensors.Mask(torch.as_tensor(decoded, dtype=torch.bool)))
                labels.append(CLASS_MAPPING[cls_id])
                is_crowd.append(is_crowd_by_id[label_id])

            except Exception as exc:
                # 6. Catch-all for unexpected errors
                logger.error(
                    "Skipping annotation id=%s due to unexpected error: %s",
                    label_id, exc,
                    exc_info=True,
                )
                continue

        return masks, labels, is_crowd

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        sub_dirs = self.sub_dirs if self.sub_dirs else None   # [] → None → scan all

        self.train_dataset = SmartRoboticsDataset(
            root_dir=Path(self.path),
            split="train",
            transforms=self.transforms,
            # transforms=None,
            target_parser=self.target_parser,
            sub_dirs=sub_dirs,
            debug=self.debug_augmentations,
            debug_per_dataset=self.debug_augmentations_per_dataset,
        )

        self.val_dataset = SmartRoboticsDataset(
            root_dir=Path(self.path),
            split="val",
            transforms=None,
            target_parser=self.target_parser,
            sub_dirs=sub_dirs,
        )

        self.test_dataset = SmartRoboticsDataset(
            root_dir=Path(self.path),
            split="test",
            transforms=None,
            target_parser=self.target_parser,
            sub_dirs=sub_dirs,
        )

        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
        test_loader = DataLoader(
            self.test_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
        return [val_loader, test_loader]