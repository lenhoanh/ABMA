# ABMA: Attention-guided bidirectional memory autoencoder for video anomaly detection

## Prerequisites
  * Linux
  * Python 3.10
  * PyTorch 1.13.1

## Setup
- Clone this repo:
  ```bash
  git clone https://github.com/lenhoanh/ABMA.git
  ``` 
    
- Install the required packages:
  ```bash
  cd ABMA
  ``` 
  ```bash
  conda env create -f abma_env.yml
  ```

## Dataset preparation

- We evaluate `ABMA` on:
  * <a href="http://www.svcl.ucsd.edu/projects/anomaly/dataset.html" target="_blank">UCSD Ped2</a>
  * <a href="http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html" target="_blank">CUHK Avenue</a>
  * <a href="https://svip-lab.github.io/dataset/campus_dataset.html" target="_blank">ShanghaiTech Campus</a>
  * <a href="https://rodrigues-royston.github.io/Multi-timescale_Trajectory_Prediction/" target="_blank">IITB-Corridor</a>

- Download and copy them into `dataset` directory with the following structure: [dataset directory](./docs/dataset_tree.md)
 
## Evaluation

Dataset          |         AUC (%)          |
-----------------|:------------------------:
UCSD Ped2   | 99.02
CUHK Avenue  | 89.11
ShanghaiTech        | 75.51
IITB-Corridor        | 70.90

- First download pre-trained `ABMA` models in link [Google Drive](https://drive.google.com/drive/folders/1PC4QDcTcwRIQev03vJw-uuEwQo6k7PEM?usp=sharing),
then copy them into `output/train` directory with the following structure: [train directory](./docs/output_train.md)

- To evaluate a pretrained `ABMA` on a dataset (ped2, avenue, shanghaitech, or iitb), run the following commands:
    ```bash
    python main.py --model ABMA --method ABMA_SF --phase test --dataset ped2 --device cuda:0 MODEL.params.convAE convAE_abma_k3 TEST.dir_name session_2025_04_17_21_06_15 TEST.file_name epoch_60.pth
    ```
    ```bash
    python main.py --model ABMA --method ABMA_SF --phase test --dataset avenue --device cuda:0 MODEL.params.convAE convAE_abma_k7 TEST.dir_name session_2025_04_18_23_01_55 TEST.file_name epoch_60.pth
    ```
    ```bash
    python main.py --model ABMA --method ABMA_SF --phase test --dataset shanghaitech --device cuda:0 MODEL.params.convAE convAE_abma_k7 TEST.dir_name session_2025_04_11_12_41_40 TEST.file_name epoch_10.pth
    ```
    ```bash
    python main.py --model ABMA --method ABMA_SF --phase test --dataset iitb --device cuda:0 MODEL.params.convAE convAE_abma_k7 TEST.dir_name session_2025_04_28_13_57_33 TEST.file_name epoch_10.pth
    ```

- Then, the results will be printed in terminal and exported to `output/test` directory:  [test directory](./docs/output_test.md)

- To export a video (including anomaly scores and anomaly map), run the following commands:
    ```bash
    python main.py --model ABMA --method ABMA_SF --phase test --dataset ped2 --device cuda:0 MODEL.params.convAE convAE_abma_k3 TEST.dir_name session_2025_04_17_21_06_15 TEST.file_name epoch_60.pth TEST.test_type video TEST.plot_video_ids "[2]" TEST.export_video True
    ```
    ```bash
    python main.py --model ABMA --method ABMA_SF --phase test --dataset avenue --device cuda:0 MODEL.params.convAE convAE_abma_k7 TEST.dir_name session_2025_04_18_23_01_55 TEST.file_name epoch_60.pth TEST.test_type video TEST.plot_video_ids "[3]" TEST.export_video True
    ```
    ```bash
    python main.py --model ABMA --method ABMA_SF --phase test --dataset shanghaitech --device cuda:0 MODEL.params.convAE convAE_abma_k7 TEST.dir_name session_2025_04_11_12_41_40 TEST.file_name epoch_10.pth TEST.test_type video TEST.plot_video_ids "[25]" TEST.export_video True
    ```
    ```bash
    python main.py --model ABMA --method ABMA_SF --phase test --dataset iitb --device cuda:0 MODEL.params.convAE convAE_abma_k7 TEST.dir_name session_2025_04_28_13_57_33 TEST.file_name epoch_10.pth TEST.test_type video TEST.plot_video_ids "[20]" TEST.export_video True
    ```

## Training from scratch

- To train `ABMA` on a dataset_name (ped2, avenue, shanghaitech, or iitb), run the following commands:
    ```bash
     python main.py --model ABMA --method ABMA_SF --dataset dataset_name --phase train --device cuda:0
    ```

- To change other options, modify the config file: `<system/ABMA/ABMA_SF/dataset_name.yaml>`.





