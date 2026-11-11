"""
Split the dataset into two fixed subsets:
  tune.txt : images you may look at while choosing parameters (CLAHE clip, gamma, etc.)
  test.txt : images used ONLY for the final reported numbers
"""
import os                      # os.listdir lists the files inside a folder
import random                  # used to shuffle the file list
import sys                     # sys.exit stops the script with an error message
from pathlib import Path       # Path makes folder paths easy to build and check

# ---- settings ----
ROOT = Path(__file__).parent               # folder where this script lives
RAW_DIR = ROOT / "data" / "raw-890"        # folder with the degraded underwater images
REF_DIR = ROOT / "data" / "reference-890"  # folder with the matching reference images
SPLIT_DIR = ROOT / "splits"                # folder where the split files will be saved
TEST_SIZE = 90                             # how many images go into the test set
SEED = 42                                  # fixed number so the shuffle is the same every run
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp"}  # file types we treat as images (ignores others)
# ------------------


def list_images(folder):                   # define a helper that returns image names in a folder
    return {                               # build a set (fast to compare, no duplicates)
        f for f in os.listdir(folder)      # go through every file name in the folder
        if Path(f).suffix.lower() in VALID_EXT  # keep it only if its extension is an image type
    }


def main():                                # main function holding the whole procedure
    if not RAW_DIR.exists() or not REF_DIR.exists():  # check both dataset folders exist
        sys.exit(f"Missing folder. Expected:\n  {RAW_DIR}\n  {REF_DIR}")  # stop and say what is missing

    raw, ref = list_images(RAW_DIR), list_images(REF_DIR)  # get image names from both folders
    paired = sorted(raw & ref)             # & keeps only names in BOTH folders; sorted gives a stable order

    if raw != ref:                         # if the two folders don't match exactly
        print(f"Warning: {len(raw - ref)} raw-only and {len(ref - raw)} reference-only "
              f"files were skipped.")      # tell the user how many unmatched files were dropped
    if len(paired) <= TEST_SIZE:           # make sure there are enough images to split
        sys.exit(f"Only {len(paired)} paired images found, need more than {TEST_SIZE}.")  # stop if not

    random.Random(SEED).shuffle(paired)    # shuffle the list with a seeded generator (reproducible)
    test = sorted(paired[:TEST_SIZE])      # first 90 shuffled names become the test set (sorted for tidy files)
    tune = sorted(paired[TEST_SIZE:])      # all remaining names become the tuning set

    SPLIT_DIR.mkdir(exist_ok=True)         # create the splits folder if it doesn't exist yet
    (SPLIT_DIR / "test.txt").write_text("\n".join(test))  # save test names, one per line
    (SPLIT_DIR / "tune.txt").write_text("\n".join(tune))  # save tune names, one per line

    print(f"Paired images: {len(paired)}")                       # report total usable pairs
    print(f"tune: {len(tune)}  -> {SPLIT_DIR / 'tune.txt'}")     # report tune count and where it was saved
    print(f"test: {len(test)}  -> {SPLIT_DIR / 'test.txt'}")     # report test count and where it was saved


if __name__ == "__main__":                 # True only when you run this file directly (not when imported)
    main()                                 # run the procedure above