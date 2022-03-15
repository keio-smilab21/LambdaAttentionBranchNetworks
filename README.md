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
1. Create the config file (the default config files are in "config")
1. Download datasets from [here](https://drive.google.com/drive/folders/1sFabOC5f_iNlclACY_Th4tQWKUwJddXZ?usp=sharing) and extract the files to "datasets"
1. Run Training
    ```
    $ python train.py -c [path to config json]
    ```
    (pretrained models are [here](https://drive.google.com/drive/folders/1rFXBXlT7ieOg351glLjmy3BhT6Yfvffk?usp=sharing))
1. Run evaluate
    ```
    $ python evaluate.py -c [path to config json] --pretrained [path to pretrained]
    ```
1. Visualize the attention map
    ```
    $ python visualize.py -c [path to config json] --block_size [block_size] --insdel_step [step]
    ```