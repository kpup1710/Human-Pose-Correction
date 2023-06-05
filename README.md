
# Human-Pose-Correction
## Clone the repo
```
git clone https://github.com/kpup1710/Human-Pose-Correction
cd Human-Pose-Correction/
```
## If you want to run the conde on your  own machine
```
conda create -n [env_name] python=3.9
conda activate [env_name]
pip install-r requirments.txt
```
## If you want to run it in my folder (dung.nv195861)
```
conda activate hpcenv
```
## Download datasets
First download datasets from this [link](https://drive.google.com/drive/folders/16zYdV5Uk6hzPXuCUJ_TcC6jM4YIiER24?usp=sharing). </br>
After that put the datasets into `./dataset` directory.

## Train the whole model (predictor and corrector)
Run the following command to start the training process

```
python main.py --phase train

```
To evaluate
```
python main.py --phase val
```
