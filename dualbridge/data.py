"""Paired datasets. Every item is (x0, y) in [-1, 1], x0 = target, y = source.

aligned_ab   pix2pix layout: one image per pair with A | B side by side
             (cityscapes, edges2handbags from the pix2pix download).
folders      official-repo layout: <root>/<split>/A (source) and /B (target).
sr           super-resolution: HR crops; y = bicubic down-x{scale} then up again.
sr_sample    super-resolution on the two photos bundled with scikit-learn.
synthetic    procedurally generated pairs (optionally with 4 valid outputs each).
"""
import os
import random
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _list_images(d):
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXT))


def to_tensor(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1).contiguous()


class AlignedAB(Dataset):
    def __init__(self, root, split="train", image_size=256, source="B", flip=False):
        if split == "test" and not os.path.isdir(os.path.join(root, "test")):
            split_dir = "val"  # pix2pix releases ship train/ and val/ only
        else:
            split_dir = split
        self.files = _list_images(os.path.join(root, split_dir))
        self.size, self.source, self.flip = image_size, source, flip and split == "train"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        w, h = img.size
        a, b = img.crop((0, 0, w // 2, h)), img.crop((w // 2, 0, w, h))
        src, tgt = (b, a) if self.source == "B" else (a, b)
        src = src.resize((self.size, self.size), Image.NEAREST)  # labels/edges: keep hard edges
        tgt = tgt.resize((self.size, self.size), Image.BICUBIC)
        x0, y = to_tensor(tgt), to_tensor(src)
        if self.flip and random.random() < 0.5:
            x0, y = x0.flip(-1), y.flip(-1)
        return x0, y


class Folders(Dataset):
    def __init__(self, root, split="train", image_size=256, flip=False):
        self.a = _list_images(os.path.join(root, split, "A"))
        self.b = _list_images(os.path.join(root, split, "B"))
        assert len(self.a) == len(self.b), "A and B must contain the same number of images"
        self.size, self.flip = image_size, flip and split == "train"

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        y = to_tensor(Image.open(self.a[i]).resize((self.size, self.size), Image.BICUBIC))
        x0 = to_tensor(Image.open(self.b[i]).resize((self.size, self.size), Image.BICUBIC))
        if self.flip and random.random() < 0.5:
            x0, y = x0.flip(-1), y.flip(-1)
        return x0, y


class SuperRes(Dataset):
    """Random HR crops for training, centre crops for testing."""

    def __init__(self, root, split="train", image_size=256, scale=4, crops_per_image=1):
        self.files = _list_images(root)
        self.size, self.scale, self.train = image_size, scale, split == "train"
        self.repeat = crops_per_image if self.train else 1

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, i):
        hr = Image.open(self.files[i % len(self.files)]).convert("RGB")
        w, h = hr.size
        if min(w, h) < self.size:
            f = self.size / min(w, h)
            hr = hr.resize((max(self.size, round(w * f)), max(self.size, round(h * f))), Image.BICUBIC)
            w, h = hr.size
        if self.train:
            l, t = random.randint(0, w - self.size), random.randint(0, h - self.size)
        else:
            l, t = (w - self.size) // 2, (h - self.size) // 2
        hr = hr.crop((l, t, l + self.size, t + self.size))
        lr = hr.resize((self.size // self.scale,) * 2, Image.BICUBIC).resize((self.size,) * 2, Image.BICUBIC)
        x0, y = to_tensor(hr), to_tensor(lr)
        if self.train and random.random() < 0.5:
            x0, y = x0.flip(-1), y.flip(-1)
        return x0, y


class Synthetic(Dataset):
    """'Label map -> textured image' toy task: flat-coloured shapes become shaded,
    striped shapes. Deterministic per index, so train/test splits are stable.

    ambiguous=False  one correct output per label map.
    ambiguous=True   the stripes take one of 4 patterns (2 directions x 2 phases)
                     that the label map does not reveal, so there are 4 equally
                     valid outputs. Their pixel average has no stripes at all,
                     which is what a pure regression (MSE/L1-optimal) model
                     drifts toward. Training draws a random pattern each time;
                     test item i uses pattern i % 4 as its ground truth.
    """

    N_MODES = 4

    def __init__(self, split="train", image_size=64, n=2000, ambiguous=False):
        self.size, self.n, self.offset = image_size, n, 0 if split == "train" else 10 ** 6
        self.ambiguous, self.train = ambiguous, split == "train"

    def __len__(self):
        return self.n

    def _label(self, i):
        rng = random.Random(i + self.offset)
        s = self.size
        lab = Image.new("RGB", (s, s), (0, 0, 0))
        d = ImageDraw.Draw(lab)
        palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        for _ in range(rng.randint(2, 4)):
            x1, y1 = rng.randint(0, s - 16), rng.randint(0, s - 16)
            x2, y2 = x1 + rng.randint(10, s // 2), y1 + rng.randint(10, s // 2)
            col = rng.choice(palette)
            (d.ellipse if rng.random() < 0.5 else d.rectangle)((x1, y1, x2, y2), fill=col)
        return to_tensor(lab)

    def _render(self, y, mode):
        s = self.size
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s), indexing="ij")
        shade = 0.6 + 0.4 * torch.cos(3 * xx + 2 * yy)
        if self.ambiguous:
            direction = xx + yy if mode % 2 == 0 else xx - yy
            stripes = 0.35 * torch.sin(10 * direction + (mode // 2) * torch.pi)
        else:
            stripes = 0.15 * torch.sin(12 * (xx + yy))
        x0 = (y * shade + stripes * (y > 0).float()).clamp(-1, 1)
        background = (-0.6 + 0.2 * yy).expand(3, -1, -1)
        return torch.where(y.sum(0, keepdim=True) == -3, background, x0)

    def modes(self, i):
        """All valid outputs for test item i, shape (N_MODES, 3, H, W)."""
        y = self._label(i)
        return torch.stack([self._render(y, m) for m in range(self.N_MODES)])

    def __getitem__(self, i):
        y = self._label(i)
        if not self.ambiguous:
            mode = 0
        elif self.train:
            mode = random.randrange(self.N_MODES)
        else:
            mode = i % self.N_MODES
        return self._render(y, mode), y


class SampleImagesSR(Dataset):
    """4x super-resolution on the two photos bundled with scikit-learn
    (china.jpg, flower.jpg, 427x640 each), so no download is needed.

    The right-most `test_frac` of each photo is held out. Training takes random
    crops (with flips) from the left part; testing uses a fixed grid of half-
    overlapping crops from the held-out strip.
    """

    def __init__(self, split="train", image_size=64, scale=4, test_frac=0.25, n_train=4000):
        from sklearn.datasets import load_sample_images
        imgs = load_sample_images().images
        self.size, self.scale, self.train, self.n_train = image_size, scale, split == "train", n_train
        self.regions, self.crops = [], []
        for img in imgs:
            h, w, _ = img.shape
            cut = int(w * (1 - test_frac))
            region = img[:, :cut] if self.train else img[:, cut:]
            self.regions.append(Image.fromarray(region))
            if not self.train:
                rw, rh = region.shape[1], region.shape[0]
                for top in range(0, rh - image_size + 1, image_size // 2):
                    for left in range(0, rw - image_size + 1, image_size // 2):
                        self.crops.append((len(self.regions) - 1, left, top))

    def __len__(self):
        return self.n_train if self.train else len(self.crops)

    def __getitem__(self, i):
        s = self.size
        if self.train:
            region = random.choice(self.regions)
            w, h = region.size
            left, top = random.randint(0, w - s), random.randint(0, h - s)
        else:
            r, left, top = self.crops[i]
            region = self.regions[r]
        hr = region.crop((left, top, left + s, top + s))
        lr = hr.resize((s // self.scale,) * 2, Image.BICUBIC).resize((s, s), Image.BICUBIC)
        x0, y = to_tensor(hr), to_tensor(lr)
        if self.train and random.random() < 0.5:
            x0, y = x0.flip(-1), y.flip(-1)
        return x0, y


def build_dataset(cfg, split):
    d = cfg["data"]
    kind = d["type"]
    if kind == "aligned_ab":
        return AlignedAB(d["root"], split, d["image_size"], d.get("source", "B"), d.get("flip", False))
    if kind == "folders":
        return Folders(d["root"], split, d["image_size"], d.get("flip", False))
    if kind == "sr":
        root = d["train_root"] if split == "train" else d["test_roots"][d.get("test_set", 0)]
        return SuperRes(root, split, d["image_size"], d.get("scale", 4), d.get("crops_per_image", 1))
    if kind == "synthetic":
        n = d.get("n_train", 2000) if split == "train" else d.get("n_test", 64)
        return Synthetic(split, d["image_size"], n, d.get("ambiguous", False))
    if kind == "sr_sample":
        return SampleImagesSR(split, d["image_size"], d.get("scale", 4), d.get("test_frac", 0.25),
                              d.get("n_train", 4000))
    raise ValueError(kind)
