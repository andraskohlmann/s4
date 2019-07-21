# Self-Supervised Semantic Segmentation

An experiment to check out an idea, about applying self-supervision with semantic segmentation as a proxy task

## Setup

- Download CityScapes dataset: https://www.cityscapes-dataset.com/downloads/

- Clone Cityscapes scripts: https://github.com/mcordts/cityscapesScripts

- Export CITYSCAPES_DATASET environment variable

- Run createTrainIdLabelImgs.py to create train labels with 19 classes

- Build docker image and run container
