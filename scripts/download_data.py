"""Download a pix2pix paired dataset into data/<name>/ (train/ and val/).

    python scripts/download_data.py cityscapes        # ~100 MB
    python scripts/download_data.py edges2handbags    # ~8.6 GB

DF2K / BSD100 / Urban100 for super-resolution are not downloaded here; get the
HR images from their official pages and point configs/sr_df2k.yaml at them.
"""
import os
import sys
import tarfile
import urllib.request

URL = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{}.tar.gz"


def main(name):
    os.makedirs("data", exist_ok=True)
    archive = os.path.join("data", f"{name}.tar.gz")
    if not os.path.exists(archive):
        print(f"downloading {URL.format(name)}")
        urllib.request.urlretrieve(URL.format(name), archive,
                                   lambda b, bs, tot: print(f"\r{b * bs / 1e6:.0f}/{tot / 1e6:.0f} MB", end=""))
        print()
    with tarfile.open(archive) as tar:
        tar.extractall("data", filter="data")
    print("extracted to", os.path.join("data", name))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cityscapes")
