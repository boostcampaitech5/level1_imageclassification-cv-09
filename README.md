# 안녕하세요! 👨‍👨‍👦‍👦

네이버 부스트캠프 AItech 5기 CV_9팀 level-1(image classification) 프로젝트 공간입니다.

![image](https://user-images.githubusercontent.com/72616557/228166051-e8197cb8-0025-485d-becc-cba4a5c257fd.png)



## Contributors

|신현준 |                                                  한현민|정현석 |                                                  김지범|오유림|
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [<img src="https://avatars.githubusercontent.com/u/113486402?s=400&v=4" alt="" style="width:100px;100px;">](https://github.com/june95) <br/> | [<img src="https://avatars.githubusercontent.com/u/33598545?s=400&u=d0aaa9e96fd2fa1d0c1aa034d8e9e2c8daf96473&v=4" alt="" style="width:100px;100px;">](https://github.com/Hyunmin-H) <br/> | [<img src="https://avatars.githubusercontent.com/u/72616557?v=4" alt="" style="width:100px;100px;">](https://github.com/hyuns66) <br/> | [<img src="https://avatars.githubusercontent.com/u/91449518?v=4" alt="" style="width:100px;100px;">](https://github.com/jibeomkim7) <br/> |[<img src="https://avatars.githubusercontent.com/u/63313306?s=400&u=094cba544d8029b4f93aa191d036a109d6265fa8&v=4" alt="" style="width:100px;100px;">](https://github.com/jennifer060697) <br/> |


해당 프로젝트 repository에서 참고한 reference 목록입니다.

Model soups : [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/abs/2203.05482).

ViT : https://github.com/google-research/vision_transformer

  
## Setting Step
### 1. 가상 환경 설치  
```bash
conda env create -f environment.yml
conda activate model_soups
```
### 2. 추가 패키지 설치
- wandb, albumentations 등 추가 설치  
### 3. pretrained model 다운로드  
```bash
python main.py --download-models --model-location <where models will be stored>  
```
  
## Training Step  
### 1. ViT-B/32 / ViT-B/16
### 1-1. Fine Tuning
```bash
python finetune.py --name {모델명} --i {모델 number} --batch-size {배치 사이즈(ex:256)} --epochs {에폭 수(ex:10)} --random-seed {시드 설정}
```
- ImageNet 등을 이용하여 미리 학습한 모델 parameter를 이용하여, 우리의 데이터셋에 맞게 마지막 layer를 바꿔주고 학습하는 부분입니다.
- 모델 number range는 0~71(72개 입니다.)  
- 저장되는 모델 pt 파일명은 "모델명i_epochs10.pt"
- "--model {ViT-B/32 | ViT-B/16}" argument 이용하여, base 모델 설정이 가능합니다.
- 추가로 learning rate, data-location과 같은 argument들이 있으며, 모든 argument는 default 값을 finetune.py에서 설정할 수 있습니다.
- Tip : 쉘 스크립트를 사용하여 학습 자동화하기 -> training.sh 파일 작성 후 다음 명령어 실행
```bash
bash trining.sh
``` 
#### 1-2. Data oversampling 여부 설정
```bash  
python finetune.py --old-aug True
```  
- 저희는 Old class의 train dataset이 적은 것을 어느정도 해결하기 위해 Old class data만 추가로 over sampling 하는 코드 또한 구현했습니다.  
- finetune 파일을 실행할 때에, "--old-aug True" 로 argument를 추가해주면, Old class data만 한 번 더 추가하여 학습하도록 설계했습니다.
- 해당 data에 augmentation을 따로 설정해주기 위해, maskbasedataset.py 에서 get_transform 함수에 추가로 'train2' augmentation을 추가해 주었습니다.
#### 1-3. Loss Function 설정
```bash
python finetune.py --loss-fn {CrossEntropyLoss | ContrastiveLoss}
```  
- default loss function은 CrossEntropyLoss 이며, ContrastiveLoss를 사용 시에, argument를 설정해주면 됩니다.


### 2. Model Soups
#### 2-1. Fine Tuning
- model soups는 ViT-B/32 모델을 기준으로 하기 때문에, 1-1 의 Fine Tuning에서 "--model ViT-B/32" argument를 이용하여 사용할 수 있습니다.
- 이 외의 사용법은 1-1 과 동일합니다.

#### 2-2. Individual Evaluation  
```bash
python main.py --eval-individual-models --name {모델명}
```
- finetune을 통해 만든 모델들의 accuracy를 측정하여 기록해두는 부분입니다.
- 추가 argument로 모델의 개수(NUM_MODELS), 모델에서 사용한 epoch, val_ratio를 적어줍니다.
- val_ratio에 None 값을 입력하면, 전체 dataset에 대해 evaludation을 진행합니다. 
- finetune 당시에 random-seed를 설정해주었다면, None값을 넣어주면 안됩니다.
- 실행 결과로 logs 폴더 안에 각 모델의 accuracy가 적힌 jsonl 파일이 생성됩니다. 

### 2-3. Greedy Soup
```bash
python main.py --greedy-soup --name {모델명}
```  
- individual Evaluation에서 저장한 여러 모델의 accuracy정보를 내림차순으로 정렬합니다.  
- 정렬 기준으로 좋은 성능을 내는 모델들을 순서대로 불러와 greedy하게 조합하여(averaging) 더 좋은 성능을 내도록 하는 최종 모델을 생성합니다.
- 실행 결과 model 폴더 안에 greedy 모델이 저장됩니다.
- log 폴더 안에 변수 GREEDY_SOUP_LOG_FILE가 이름임 로그를 저장합니다. 해당 로그에는 averaging된 모델 정보가 저장됩니다.


## Inference Step
### 1. Inference Step
```bash
python inference.py --model-name {모델명.pt 파일}
```
- 생성한 모델 파일(.pt)를 이용하여 Test data를 예측하는 부분입니다.  
- argument 내에 pt 파일명을 적고 실행시킵니다.   
- 최종 예측한 csv 파일이 output 폴더에 저장됩니다. 

#### 1-1. Validation 확인
```bash
python validation.py --model-name {모델명.pt 파일}
```
- 우리가 학습한 모델을 가지고 동일한 validation set에서 어떤 class가 예측을 잘못했는지 출력해주는 부분입니다.
- 해당 모델.pt를 학습했을 때, 사용했던 seed 값을 동일하게 유지해 주어야 정확한 확률과 예측값이 나옵니다.
- 아래 그림은 출력 예시입니다.   
![image](https://user-images.githubusercontent.com/113486402/234260857-a5175967-8a7c-4c0b-bcfd-a63f7fb1559c.png)

#### 1-2. Weighted Ensemble  
- "--weighted-ensemble" argument를 이용하여 사용할 수 있습니다.
#### 1-3. Soft voting (Ensemble)  
- "--soft-voting" argument를 이용하여 사용할 수 있습니다. 
#### 1-4. Hard voting (Ensemble)
- inference.py 를 통해 예측된 output.csv 여러개의 결과값을 가지고 최종적으로 hard voting을 수행하는 Ensemble 또한 구현했습니다.  
- hard_voting.ipynb 을 실행하여, 앙상블을 원하는 csv를 가지고 hard voting을 수행할 수 있습니다. 
  
## 추가 기능
### 1. Relabeling  
![image](https://user-images.githubusercontent.com/113486402/233954582-70a43065-7586-483e-abf5-707e744eebb3.png)  
- relabeling이 필요한 id 목록을 list에 넣어서 relabel_dict 딕셔너리에 넣어주었습니다.
- maskbasedataset.py에서 추가로 relabeling이 필요한 id가 있다면 간단하게 해당 list에 넣어주기만 하면 relabeling을 수행합니다.  

### 2. Optuna  
- 
