# [[RAID2024] AI-Generated Faces in the Real World: A Large-Scale Case Study of Twitter Profile Images](https://arxiv.org/abs/2404.14244)
[Jonas Ricker](https://jonasricker.com), [Dennis Assenmacher](https://dennisassenmacher.de/), [Thorsten Holz](https://cispa.de/en/people/c01thho), [Asja Fischer](https://informatik.rub.de/fischer/), and [Erwin Quiring](https://www.erwinquiring.com/)   
International Symposium on Research in Attacks, Intrusions and Defenses (RAID), 2024

## Setup
We tested our code with Python 3.10. Install all required packages (preferably in a virtual environment) using
```
pip install -r requirements.txt
pip install -e .
```

## Data and Checkpoint
The following can be downloaded from [Zenodo](https://zenodo.org/doi/10.5281/zenodo.13791745):
- raw images from FFHQ and TPDNE
- images from FFHQ and TPDNE uploaded/downloaded to Twitter
- trained checkpoint for our ResNet detector
- IDs of the 7723 accounts we identified as using AI-generated profile images

We cannot provide the actual profile images due to Twitter's terms of service.

## Detection Pipeline
To use our detection pipeline, you first need to download the necessary weights:
- download the [weights](https://github.com/hollance/BlazeFace-PyTorch) (`anchors.npy`, `anchorsback.npy`, `blazeface.pth`, `blazefaceback.pth`) for BlazeFace and put them in `weights/blazeface`
- download the checkpoint for our trained classifier and put them in `weights/resnet`

Then, run
```
python detection_pipeline.py --input-dir path/to/images --det-weights weights/resnet/ffhq+pseudo_vs_tpdne_0.1.ckpt --alignment-reference-dir path/to/alignment/ref
```
where `path/to/alignment/ref` contains faces with the desired alignment that should be used as a reference.
The results will be saved to `results/detection_pipeline`. Use `-h` to learn more about optional arguments.

## Detector
To train your own real/fake detector, run
```
python train_detector.py --run-name my_run --real-dirs path/to/reals path/to/other/reals --fake-dirs path/to/fakes path/to/other/fakes
```
The script expects all image directories to have "train", "val", and "test" subdirectories. The trained checkpoint will be saved to `results/train_detector`. Use `-h` to learn more about optional arguments.

## Duplicate Detection
To identify clusters of duplicate images based on perceptual hashing, run
```
python detect_duplicates.py --input-dir path/to/images
```
The results will be saved to `results/detect_duplicates`. Use `-h` to learn more about optional arguments.

## Content Analysis
In `content_analysis.ipynb` we show how we analyzed the tweet contents, exemplarily for active English accounts. Unfortunately we cannot publish the actual tweets due to Twitter's terms of service.
