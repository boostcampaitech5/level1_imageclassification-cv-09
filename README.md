# Image Classication Competition 👨‍👨‍👦‍👦
![image](https://user-images.githubusercontent.com/72616557/228166051-e8197cb8-0025-485d-becc-cba4a5c257fd.png)

## About
목표 : 사람의 정면 사진으로 1. 마스크 착용 여부 2. 성별 3. 나이를 아래와 같이 18개 class로 예측 

<img src="https://user-images.githubusercontent.com/33598545/234359070-291d7e20-08c9-4824-ab56-70f4c16acd40.png" width="450" height="400" />

 
<mark>#ViT32</mark> <mark> #ViT16</mark> <mark> #ModelSoups</mark> <mark> #Relabeling</mark> <mark>#Oversampling</mark> <mark>#ContrastiveLearnig</mark> <mark> #WeightedAverageEnsemble</mark> <mark>   #HardVoting</mark> <mark> #SoftVoting</mark> <mark> #Optuna</mark> <mark> #Wandb</mark> 

****





  
## Setting Step
### 1. 가상 환경 설치  
```bash
conda env create -f environment.yml
conda activate model_soups
```
### 2. 추가 패키지 설치
- wandb, albumentations 등 추가 설치  
### 3. pretrained model 다운로드  
- [Model soups](https://github.com/mlfoundations/model-soups/releases/tag/v0.0.2)에서 제공한 ViT-B/32 모델 다운
- 총 72개, 본 프로젝트에서는 최대 40개를 사용했습니다.
```bash
python main.py --download-models --model-location <where models will be stored>  
```
****
## Training Step  
### 1. ViT-B/32, ViT-B/16
#### 1-1. Fine Tuning
```bash
python finetune.py --name {모델명} --i {모델 number} --random-seed {시드 설정}
```
- [Model soups](https://github.com/mlfoundations/model-soups/releases/tag/v0.0.2)에서 제공한 pretrained 모델을 18개의 class vector를 output으로 하는 1개의 linear layer를 추가하여 학습합니다. 
- ViT-B/16 의 경우 Model soups pretrained weight 가 없으므로 clip 라이브러리에서 제공하는 ImageNet pretrained weight 을 사용합니다.
- `--model {ViT-B/32 | ViT-B/16}` : base 모델 설정
- `--name` : 저장할 모델 이름
- `--i` : pretrained model의 index
- `--random-seed` : random seed
- `--lr`, `--batch-size`, `--epochs`, `--data-location`, `--model-location` : learning rate, batch size, epoch, 데이터 경로, 저장할 모델 경로
<!-- - 추가로 learning rate, data-location과 같은 argument들이 있으며, 모든 argument는 default 값을 finetune.py에서 설정할 수 있습니다. -->
- Tip : 쉘 스크립트를 사용하여 학습 자동화하기. training.sh 파일 작성 후 다음 명령어 실행
<!-- - 저장되는 모델 pt 파일명은 "모델명i_epochs10.pt" -->
```bash
bash training.sh
``` 
#### 1-2. Data oversampling
<!-- ```bash  
python finetune.py --old-aug True
```   -->
- Age 속성의 Old class의 적은 train dataset으로 저하된 학습 성능을 개선하기 위해 Old class data만을 추가로 Over sampling하였습니다.
- `--old-aug True` : Old class 1회 추가 over sampling
<!-- - 해당 data에 augmentation을 따로 설정해주기 위해, maskbasedataset.py 에서 get_transform 함수에 추가로 'train2' augmentation을 추가해 주었습니다. -->
#### 1-3. Loss Function 설정
<!-- ```bash
python finetune.py --loss-fn {CrossEntropyLoss | ContrastiveLoss}
```   -->
- Interclass의 거리를 넓히고, Intraclass의 거리를 좁히는 Contrastive Learning을 사용하였습니다.
- `--loss-fn` : ContrastiveLoss or CrossEntropyLoss, default는 CrossEntropyLoss


### 2. Model Soups
- [Model soups](https://github.com/mlfoundations/model-soups/releases/tag/v0.0.2)는 여러 개의 동일한 구조를 가진 pretrained 학습 모델들을 조합하여 하나의 학습 모델을 만드는 앙상블 기법입니다. 
- 수행 과정을 아래와 같습니다.
1. 여러개의 pretrained model을 Test 하여 Accuracy를 얻는다.
2. Accuracy 값으로 내림차순으로 정렬한다.
3. 순차적으로 다음 모델과의 weight값을 average하여 하나의 모델을 생성한다.
4. 생성된 모델의 성능을 측정을 하였을 때 현재까지 가장 좋은 Accuracy보다 성능이 좋으면 저장하고, 3, 4번을 반복한다. 그렇지 않으면 average하지 않고 3, 4번을 반복한다.
5. 가장 Accuracy가 높은 모델을 최종 모델로 선정한다. 
#### 2-1. Fine Tuning
<!-- 모델을 기준으로 하기 때문에, 1-1 의 Fine Tuning에서 "--model ViT-B/32" argument를 이용하여 사용할 수 있습니다. -->
- [Model soups](https://github.com/mlfoundations/model-soups/releases/tag/v0.0.2)에서 제공한 pretrained model은 ViT-B/32 모델입니다.
- 1번과 동일하게 Fine tuning을 진행합니다.
<!-- - 이 외의 사용법은 1-1,2,3과 동일합니다. -->

#### 2-2. Individual Evaluation  
```bash
python main.py --eval-individual-models --name {모델명} --model-num {모델 개수} --random-seed {랜덤 시드}
```
- finetune을 통해 만든 모델들의 accuracy를 측정하여 기록합니다. 
- `--name` : 저장된 모델명
- `--model-num` : Evaludation할 모델의 개수
- `--random-seed` : 랜덤 시드 
- `--val-ratio`, `--epoch`, `--data-location`, `--model-locatoin` : validation dataset 비율, epoch, 데이터셋 경로, 저장할 모델 경로
<!-- - val_ratio에 None 값을 입력하면, 전체 dataset에 대해 evaludation을 진행합니다. 
- finetune 당시에 random-seed를 설정해주었다면, None값을 넣어주면 안됩니다. -->
- 실행이 완료되면 logs 폴더 안에 각 모델의 accuracy가 적힌 jsonl 파일이 생성됩니다. 

#### 2-3. Greedy Soup
```bash
python main.py --greedy-soup --name {모델명} --model-num {모델 개수} --random-seed {랜덤 시드}
```  
- individual Evaluation에서 저장한 여러 모델의 accuracy정보를 내림차순으로 정렬합니다. 정렬 기준으로 좋은 성능을 내는 모델들을 순서대로 불러와 greedy하게 조합하여(averaging) 더 좋은 성능을 내도록 하는 최종 모델을 생성합니다.
- `--name` : 저장된 모델명
- `--model-num` : Evaludation할 모델의 개수
- `--random-seed` : 랜덤 시드 
- `--val-ratio`, `--epoch`, `--data-location`, `--model-locatoin` : validation dataset 비율, epoch, 데이터셋 경로, 저장할 모델 경로
- 실행 결과 model 폴더 안에 최종 모델이 저장됩니다.
- log 폴더 안에 변수 GREEDY_SOUP_LOG_FILE가 이름임 로그를 저장합니다. 해당 로그에는 averaging된 모델 정보가 저장됩니다.

****
## Inference Step
### 1. 예측 성능 분석 w/ Validation dataset
```bash
python validation.py --model-name {모델명.pt 파일}
```
- Validation set에서 class별로 잘못 예측한 비율을 출력합니다.
- 해당 모델을 학습했을 때, 사용했던 random seed 값을 동일하게 유지해 주어야 정확한 확률과 예측값이 나옵니다.
- `--model-name` : evaluation할 모델명, 
- `--i` : pretrained model의 index
- `--random-seed` : 랜덤 시드


#### 1-1. Weighted Average Ensemble  
- Age class의 분류 성능을 높이고자 Age 속성만을 분류하는 모델을 학습하여, 이를 전체 class(18개) 분류 모델의 예측값과 weighted sum을 하였습니다.
- `--weighted-ensemble` : Age class를 학습한 모델명, Default는 None
```
python finetune_age.py --name {모델명} --i {모델 number} --random-seed {시드 설정}
```
- finetune_age.py는 Age class만을 학습합니다.
- `--name`, `--i`, `--random-seed`는 finetune.py와 동일하게 설정
#### 1-2. Soft voting (Ensemble)  
- 2개의 학습 모델의 각 class의 확률값을 minmax scaling 후 더하는 방법입니다. 
- `--soft-voting` : soft voting할 모델명, Default는 None 
#### 1-3. Hard voting (Ensemble)
- 최종 예측 결과 csv 파일의 여러개를 최종적으로 Hard voting을 수행하여 Ensemble을 진행하였습니다.
- hard_voting.ipynb 을 실행하여, 앙상블을 원하는 csv를 가지고 hard voting을 수행할 수 있습니다. 

아래 그림은 출력 예시입니다.   
<!-- ![image](https://user-images.githubusercontent.com/113486402/234260857-a5175967-8a7c-4c0b-bcfd-a63f7fb1559c.png) -->
<img src="https://user-images.githubusercontent.com/113486402/234260857-a5175967-8a7c-4c0b-bcfd-a63f7fb1559c.png" width="300" height="500" />


### 2. Test w/ Test dataset
```bash
python inference.py --model-name {모델명.pt 파일}
```
- 생성한 모델 파일(.pt)를 이용하여 Test data를 예측하는 부분입니다.  
- `--model-name` : inference할 모델명
- `--weighted-ensemble`, `--soft-voting` : Weighted average ensemble 시 모델명, Soft Voting 시 모델명 
<!-- - argument 내에 pt 파일명을 적고 실행시킵니다.    -->
- 최종 결과 csv 파일이 output 폴더에 저장됩니다. 


****
## Additional Step
### 1. Dataset Relabeling  
![image](https://user-images.githubusercontent.com/113486402/233954582-70a43065-7586-483e-abf5-707e744eebb3.png)  
<!-- - relabeling이 필요한 id 목록을 list에 넣어서 relabel_dict 딕셔너리에 추가하였습니다. -->
- 잘못 라벨링된 데이터 id 목록을 담은 relabel_dict 딕셔너리를 사용하여 Relabeling을 진행하였습니다.
<!-- - maskbasedataset.py 파일에서 추가로 relabeling이 필요한 id가 있다면 간단하게 해당 list에 넣어주기만 하면 relabeling을 수행합니다.   -->

### 2. Hyperparameter Tuning  
```bash
python optuna_script.py
```
- Optuna를 이용하여 Hyper paramter tuning을 진행합니다.
- optuna_script.py 파일에서 hyper parameter tuning을 위한 설정을 아래 사진과 같이 넣어주고 실행합니다.  

****
## Result
- Private score 3rd / F1 score - 0.7613 / Accuracy - 81.3175
- Public score 6th / F1 score - 0.7653 / Accuracy - 81.3968
![화면 캡처 2023-04-26 022440](https://user-images.githubusercontent.com/33598545/234355466-63a4c6c0-1b86-4039-a327-15bcf7758db1.png)


****


## Contributors

|신현준 |                                                  한현민|정현석 |                                                  김지범|오유림|
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [<img src="https://avatars.githubusercontent.com/u/113486402?s=400&v=4" alt="" style="width:100px;100px;">](https://github.com/june95) <br/> | [<img src="https://avatars.githubusercontent.com/u/33598545?s=400&u=d0aaa9e96fd2fa1d0c1aa034d8e9e2c8daf96473&v=4" alt="" style="width:100px;100px;">](https://github.com/Hyunmin-H) <br/> | [<img src="https://avatars.githubusercontent.com/u/72616557?v=4" alt="" style="width:100px;100px;">](https://github.com/hyuns66) <br/> | [<img src="https://avatars.githubusercontent.com/u/91449518?v=4" alt="" style="width:100px;100px;">](https://github.com/jibeomkim7) <br/> |[<img src="https://avatars.githubusercontent.com/u/63313306?s=400&u=094cba544d8029b4f93aa191d036a109d6265fa8&v=4" alt="" style="width:100px;100px;">](https://github.com/jennifer060697) <br/> |

****
## Reference


Model soups : [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/abs/2203.05482).

ViT : https://github.com/google-research/vision_transformer

ContrastiveLoss : https://github.com/KevinMusgrave/pytorch-metric-learning