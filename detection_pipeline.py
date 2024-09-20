import argparse
from pathlib import Path
from pprint import pformat

from twitter_ai_faces.detection import compute_predictions
from twitter_ai_faces.image import (
    compute_reconstruction,
    detect_faces,
    detect_faces_post,
    faces_aligned,
    faces_valid,
)


def main(args):
    # create output directory
    output_dir = Path("results/detection_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)

    # save configuration
    with open(output_dir / "config.txt", "w") as f:
        f.write(pformat(vars(args), sort_dicts=False))

    # store all results in dataframe
    df = None

    # ===== PRE-FILTER =====

    # detect faces and landmarks
    print("Detecting faces...")
    faces = detect_faces(
        path=args.input_dir,
        batch_size=args.pre_batch_size,
        num_workers=args.pre_num_workers,
    )

    # post-process landmarks
    faces = detect_faces_post(faces)
    df = faces.sort_values("file")
    print("...done")
    print(f"Total images: {len(faces)}")

    # check if eye distance is sufficient
    valid = faces_valid(df=faces, min_eye_dist=0.1)
    df = df.merge(valid)
    print(f"Valid faces: {df['valid'].sum()}")

    # ===== CLASSIFIER =====

    # compute detection scores
    print("Computing scores...")
    scores = compute_predictions(
        path=df[df["valid"]]["file"].tolist(),
        weights_path=args.det_weights,
        crop_size=224,
        batch_size=args.det_batch_size,
        num_workers=args.det_num_workers,
    )
    df = df.merge(scores, how="outer")
    print("...done")

    # ===== ASSISTANCE TOOLS =====

    # compute alignment reference
    alignment_reference_faces = detect_faces_post(
        detect_faces(
            path=args.alignment_reference_dir,
            batch_size=args.pre_batch_size,
            num_workers=args.pre_num_workers,
        )
    )

    # check if faces are aligned
    aligned = faces_aligned(
        df[df["valid"]],
        reference=alignment_reference_faces,
        num_stds=args.alignment_num_stds,
    )
    df = df.merge(aligned, how="outer")
    df["aligned"] = df["aligned"].fillna(False)
    print(f"Aligned: {df['aligned'].sum()}")

    # compute reconstruction for images that are aligned
    reconstruction_paths = compute_reconstruction(
        path=df[df["aligned"]]["file"].tolist(),
        output_dir=output_dir / "reconstructions",
        seed=args.seed,
        num_steps=args.reconstruction_steps,
    )
    df.loc[df["aligned"], "reconstruction_path"] = reconstruction_paths

    # combine and save all data
    print("Saving results...")
    df.to_csv(output_dir / "dataframe.csv")
    print("...done")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, help="Path to image directory")

    # pre-filtering
    parser.add_argument(
        "--pre-batch-size",
        type=int,
        default=16,
        help="Batch size for pre-filtering stage",
    )
    parser.add_argument(
        "--pre-num-workers",
        type=int,
        default=8,
        help="Number of workers for pre-filtering stage",
    )

    # detection
    parser.add_argument(
        "--det-weights", type=Path, help="Path to classifier checkpoint"
    )
    parser.add_argument(
        "--det-batch-size", type=int, default=16, help="Batch size for detection stage"
    )
    parser.add_argument(
        "--det-num-workers",
        type=int,
        default=8,
        help="Number of workers for detection stage",
    )

    # alignment
    parser.add_argument(
        "--alignment-reference-dir",
        type=Path,
        help="Path to alignment reference directory",
    )
    parser.add_argument(
        "--alignment-num-stds",
        type=int,
        default=7,
        help="Number of standard deviations s.t. a faces is considered aligned",
    )

    # reconstruction
    parser.add_argument(
        "--reconstruction-steps",
        type=int,
        default=1000,
        help="Number of reconstruction steps",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
