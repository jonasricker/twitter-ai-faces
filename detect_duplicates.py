import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from twitter_ai_faces.image import compute_hash


def main(args):
    # create output directory
    output_dir = Path("results/detect_duplicates")
    output_dir.mkdir(exist_ok=True, parents=True)

    # compute image hashes
    files = sorted(args.input_dir.iterdir())
    hash_df = compute_hash(files=files, hash_function=args.hash_func)
    hash_arr = hash_df[args.hash_func].apply(int, base=16).values
    binary_arr = np.array([list(np.binary_repr(vec)) for vec in hash_arr]).astype(int)

    # perform clustering
    db = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric="hamming")
    db.fit(binary_arr)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    sizes = [(labels == label).sum() for label in range(n_clusters)]

    print("Estimated number of clusters:", n_clusters)
    print("Estimated number of noise points:", n_noise)
    print("Average cluster size: ", np.mean(sizes))

    # visualize clusters
    if n_clusters != 0:
        canvas = Image.new("RGB", (64 * max(sizes), 64 * n_clusters))
        for label in tqdm(range(n_clusters)):
            cluster_files = hash_df.iloc[labels == label]["file"].values
            for i, file in enumerate(cluster_files):
                canvas.paste(Image.open(file).resize((64, 64)), (64 * i, 64 * label))
        canvas.save(output_dir / "images.jpeg")

    # save cluster labels
    cluster_labels = pd.DataFrame({"cluster": labels}, index=hash_df.index)
    cluster_labels.to_csv(output_dir / "cluster_labels.csv")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, help="Path to image directory")
    parser.add_argument(
        "--hash-func",
        default="phash",
        help="Valid name of hash function from ImageHash library",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=3 / 64,
        help="Maximum bit difference for to images to be considered identical",
    )
    parser.add_argument(
        "--min-samples", type=int, default=2, help="Minimum number of images in cluster"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
