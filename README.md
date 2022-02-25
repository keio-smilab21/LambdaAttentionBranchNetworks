## Visualize Explanations

### Requirements
- matplotlib
- numpy
- opencv
- Pillow
- scikit-image
- scikit-learn
- torch
- torchaudio
- torchinfo
- torchvision
- tqdm

### Usage
1. Create config file
1. Download datasets
1. Run Training
    ```
    $ python train.py -c [path to config json]
    ```
1. Run evaluate
    ```
    $ python evaluate.py -c [path to config json] --pretrained [path to pretrained]
    ```