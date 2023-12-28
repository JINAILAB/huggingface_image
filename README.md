# customimage huggingface trainer

## 설명

huggingface image pretrained model들의 finetuning을 지원하는 libarary입니다. 

### 특징

- 간단한 script를 통해서 distribute training, parameter tuning, graph visualize, umap, gradcam 등을 지원합니다.
- wandb를 통하여 parameter 관리와 graph visualize를 수월하게 해줍니다.
- python script에 익숙하지 않은 분들을 위하여 ipynb 파일 또한 지원하여 colab에서도 돌리기 쉽게 지원하고 있습니다.

## 설치 Install

- Clone this repo

```bash
https://github.com/JINAILAB/huggingface_image.git
cd huggingface_image
```

- installation

```bash
pip install huggingface
```

- wandb login

wandb login을 터미널에 치기 전에 가입을 해서 APIkey를 발급 받아야합니다. 

wandb 사이트 가입 후 다음 사이트에서 확인해서 입력해주세요.

[https://app.wandb.ai/authorize](https://app.wandb.ai/authorize)

```bash
wandb login
```

## 데이터셋 준비

데이터는 두 가지 버전을 지원하고 있습니다. 

- 하나의 폴더 → random stratified split
- train, validation 폴더 → train, validation split
- 추후 k-fold 추가 예정

### 하나의 폴더 → random stratified split

하나의 폴더에서 train과 validation으로 random으로 구성하여 나눠줍니다. 데이터 구조는 다음과 같이 구성해주시면 됩니다. 단, 구성 후 반드시 zip파일로 압축해야 실행이 가능합니다.

```bash
data.zip
├── {class1 name}
│   ├── 1.png
│   ├── x.png
├── {class2 name}
│   ├── 1.png 
│   ├── x.png
...
```

이후 다음 코드를 실행하면 dataset을 만들어줍니다.

```bash
python3 build_dataset.py \
--dataset-dir $data경로 \
--test_size 0.2 \
--dataset_name $생성될데이터셋이름
```

### train, validation 폴더 → train, validation split

본인이 원하는대로 train과 validation을 나눈 후 데이터셋을 만들 수 있습니다. 위와 같이  구성 후 반드시 zip파일로 압축해야 실행이 가능합니다.

```bash
data.zip
├── train
│   ├── {class1 name}
│   │   ├── 1.png
│   │   ├── x.png
│   ├── {class2 name}
│   │   ├── 1.png
│   │   ├── x.png
├── val
│   ├── {class1 name}
│   │   ├── 1.png
│   │   ├── x.png
│   ├── {class2 name}
│   │   ├── 1.png
│   │   ├── x.png
```

이후 다음 코드를 실행하면 dataset을 만들어줍니다.

```bash
python3 build_dataset.py \
--dataset-dir $data경로 \
--dataset_name $생성될데이터셋이름 \
--trainval-dataset
```

## training script

여러가지 인자를 통해서 훈련을 제어할 수 있습니다.

### 기본사용법

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py \
-b 32 \
-e 1 \
--use-v2 \
--model facebook/convnext-large-224 \
--project_name test \
--dataset-dir clahe_hiphf \
--umap \
--gradcam
```

- -b, --batch-size
    - batchsize
- -e, --epochs
    - train epoch 수
- --use-v2
    - tensorv2를 사용합니다. transform시 좀 더 빠른 속도를 제공합니다.
- --model
    - 상용할 모델을 입력하면 됩니다. hugginface의 모델과 local 상에 저장된 모델 또한 지원됩니다.
- --project-name
    - wandb에 저장될 프로젝트 이름과 log가 저장될 폴더 이름입니다.
- --dataset-dir
    - finetuning에 사용할 데이터 경로를 적어주면 됩니다.
- --umap
    - umap을 저장합니다.
- --gradcam
    - valid set에 대해서 gradcam을 저장합니다.
- --lr
    - learning rate
- --label-smothing
    - label smoothing
- --early-stopping-epoch
    - 성능이 더 증가하지 않을 때 멈출 epoch 수를 지정해줍니다.
    - 사용하지 않는다면 0으로 지정해주면 됩니다.
- --report
    - 기본값은 wandb입니다.
    - azure_ml, clearml, codecarbon, comet_ml, dagshub, dvclive, flyte, mlflow, neptune, tensorboard 등을 지원합니다.(test 안 해봄)
- --lr
    - learning rate
- --fp16
    - amp 기능입니다.
    - fp32대신 fp16을 사용하여 training 속도가 증가합니다.
- --resize-size
    - default는 None입니다. 지정해주지 않는다면 huggingface에 저장된 parameter값으로 사용됩니다.
    - resize 후 resize-size * crop-pct를 곱하여 random crop에 사용됩니다.
- --crop-pct
    - default는 None입니다.
- --auto-augment
    - default 값으로 ta_wide가 사용됩니다.
    - None, ‘ta_wide’, ‘ra’, ‘augmix’가 지원됩니다.

## **model pretrain available**

## recommend

- google/vit-base-patch16-224
- facebook/convnext-base-224-22k
- facebook/convnext-base-224-22k-1k
- microsoft/resnet-50
- microsoft/swin-base-patch4-window7-224-in22k
- facebook/regnet-y-040

Huggingface pretrained model을 지원하지만 timm 모델은 지원하지 않고 있습니다.

[https://huggingface.co/models?pipeline_tag=image-classification&sort=trending](https://huggingface.co/models?pipeline_tag=image-classification&sort=trending)

또한 umap과 gradcam에 경우 지원하는 모델이 한정적이니 참고해서 훈련시켜주세요.

### umap

- resnet
- convnext
- convnextv2
- cvt
- swin
- swinv2
- vit
- regnet

### gradcam

- resnet
- convnext
- swin
- vit
- regnet