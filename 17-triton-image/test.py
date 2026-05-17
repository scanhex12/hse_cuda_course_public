#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from kernel import rgb2grey

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src.jpg"
DST = ROOT / "dst.jpg"


def load_rgb_tensor(path: Path) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).cuda()


def load_rgb_numpy(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def main() -> None:
    if not SRC.exists() or not DST.exists():
        raise SystemExit(
            f"Missing {SRC.name} or {DST.name}. Run: python3 make_test_images.py"
        )

    img = load_rgb_tensor(SRC)
    ref = load_rgb_numpy(DST)

    out = rgb2grey(img).detach().cpu().numpy()

    max_err = float(np.max(np.abs(out - ref)))
    mean_err = float(np.mean(np.abs(out - ref)))
    print(f"max abs error: {max_err:.6f}")
    print(f"mean abs error: {mean_err:.6f}")

    if max_err > 0.035:
        raise SystemExit(f"FAIL: max error {max_err} > 0.035")
    print("OK")


if __name__ == "__main__":
    main()
