"""
Quick visual preview: apply the available DIP methods to one or more images
and save a labelled comparison grid you can open and look at.

Usage:
    python preview.py path/to/image.jpg      # one specific image
    python preview.py --n 5                  # 5 random images from splits/tune.txt
"""
import argparse                            # reads command-line arguments
import random                              # picks random images for --n
from pathlib import Path                   # easy path handling

import cv2                                 # image loading, resizing, drawing, saving
import numpy as np                         # array stacking for the grid

from dip.color_correction import correct          # color correction methods (gray_world, white_patch, red_comp)
from dip.contrast import enhance_contrast         # contrast methods (hist_eq, clahe, stretch, gamma)
from dip.dehazing import dehaze                   # dehazing methods (dcp, udcp)
from dip.fusion import adaptive_enhance, fuse     # adaptive enhancement and fusion

ROOT = Path(__file__).parent               # folder where this script lives
RAW_DIR = ROOT / "data" / "raw-890"        # degraded images
REF_DIR = ROOT / "data" / "reference-890"  # reference images (shown if a match exists)
TUNE_FILE = ROOT / "splits" / "tune.txt"   # images allowed for parameter tuning
OUT_DIR = ROOT / "results" / "previews"    # where grids are saved
PANEL = 256                                # each panel fits inside a 256 x 256 box


def fit(img):                              # resize keeping aspect ratio, pad to a square panel
    h, w = img.shape[:2]                   # current height and width
    s = PANEL / max(h, w)                  # scale so the longer side becomes 256
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)  # shrink/enlarge
    canvas = np.zeros((PANEL, PANEL, 3), np.uint8)       # black square canvas
    y, x = (PANEL - img.shape[0]) // 2, (PANEL - img.shape[1]) // 2  # offsets to center the image
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img  # paste the image in the middle
    return canvas                          # return the padded panel


def label(img, text):                      # draw a title strip on top of a panel
    img = img.copy()                       # don't modify the original panel
    cv2.rectangle(img, (0, 0), (PANEL, 22), (0, 0, 0), -1)  # black bar behind the text
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  # white label
    return img                             # return the labelled panel


def make_panels(img, ref=None, contrast_strength=0.0):  # run every method and collect panels
    wb = correct(img, "gray_world")        # gray world result, reused in the combos below
    rc = correct(img, "red_comp")          # red channel compensation result
    combo1 = correct(rc, "gray_world")     # combo: red compensation, then gray world
    combo2 = enhance_contrast(combo1, "clahe")  # combo: ... then CLAHE (full mini-pipeline)
    panels = [
        ("original", img),
        ("gray_world", wb),
        ("white_patch", correct(img, "white_patch")),
        ("red_comp", rc),
        ("dcp dehaze", dehaze(img, "dcp")),
        ("udcp dehaze", dehaze(img, "udcp")),
        ("clahe only", enhance_contrast(img, "clahe")),
        ("stretch only", enhance_contrast(img, "stretch")),
        ("red_comp + gray_world", combo1),
        ("rc + gw + clahe", combo2),
        ("fusion", fuse(img, contrast_strength=contrast_strength, pipeline="fusion")),
        ("udcp + fusion", fuse(
            dehaze(img, "udcp"),
            dehaze_method=None,
            pipeline="fusion",
        )),
    ]
    if ref is not None:                    # add the reference as a final panel when available
        panels.append(("reference", ref))
    return panels


def make_fusion_dcp_panels(img, ref=None, contrast_strength=0.0):
    """Build the focused DCP, color, contrast, and soft-fusion comparison."""
    dcp = dehaze(img, "dcp")
    dcp_red = correct(dcp, "red_comp", alpha=0.8)
    dcp_red_clahe = enhance_contrast(dcp_red, "clahe")
    panels = [
        ("original", img),
        ("dcp dehaze", dcp),
        ("dcp + red compensation", dcp_red),
        ("dcp + red + CLAHE", dcp_red_clahe),
        ("soft dcp + fusion", fuse(
            img,
            dehaze_method="dcp",
            dehaze_strength=0.45,
            output_sharpen=0.3,
            pipeline="fusion",
        )),
        ("adaptive result", adaptive_enhance(img)),
    ]
    if ref is not None:
        panels.append(("reference", ref))
    return panels


def build_grid(panels, cols=4):            # arrange labelled panels into rows of `cols`
    tiles = [label(fit(im), t) for t, im in panels]       # resize and label every panel
    while len(tiles) % cols:               # pad with black tiles so the last row is full
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]  # join tiles into rows
    return np.vstack(rows)                 # stack rows into one big image


def main():                                # parse arguments, process images, save grids
    p = argparse.ArgumentParser()          # set up the argument parser
    p.add_argument("image", nargs="?", help="path to one image")  # optional single image path
    p.add_argument("--n", type=int, default=3, help="how many random tune images to preview")  # count for random mode
    p.add_argument("--fusion-dcp", action="store_true",
                   help="show only original, fusion, DCP, DCP + fusion, and reference")
    p.add_argument("--high-contrast", action="store_true",
                   help="apply moderate CLAHE contrast to fusion results")
    args = p.parse_args()                  # read what the user typed

    if args.image:                         # a specific image was given
        paths = [Path(args.image)]
    elif TUNE_FILE.exists():               # otherwise sample from the tune split
        names = TUNE_FILE.read_text().split("\n")
        paths = [RAW_DIR / n for n in random.sample(names, min(args.n, len(names)))]
    else:                                  # nothing to work with
        raise SystemExit("Give an image path, or run make_split.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)           # create the output folder if needed
    for path in paths:                     # process each chosen image
        img = cv2.imread(str(path))        # load as BGR uint8
        if img is None:                    # unreadable file
            print(f"Could not read {path}, skipping")
            continue
        ref_path = REF_DIR / path.name     # look for a reference with the same filename
        ref = cv2.imread(str(ref_path)) if ref_path.exists() else None  # load it if present
        if args.fusion_dcp:
            panels = make_fusion_dcp_panels(
                img, ref, contrast_strength=0.35 if args.high_contrast else 0.0
            )
            suffix = "_high_contrast" if args.high_contrast else ""
            out = OUT_DIR / f"{path.stem}_fusion_dcp{suffix}.png"
        else:
            strength = 0.35 if args.high_contrast else 0.0
            panels = make_panels(img, ref, contrast_strength=strength)
            suffix = "_high_contrast" if args.high_contrast else ""
            out = OUT_DIR / f"{path.stem}_grid{suffix}.png"
        cv2.imwrite(str(out), build_grid(panels))        # build and save the grid
        print("saved", out)


if __name__ == "__main__":                 # run only when executed directly
    main()