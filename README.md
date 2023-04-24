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

## 실행 Step  
### 1. Fine Tuning
```bash
python finetune.py --name {모델명} --i {모델 number} --batch-size {배치 사이즈(ex:256)} --epochs {에폭 수(ex:10)} --random-seed {시드 설정}
```
- ImageNet 등을 이용하여 미리 학습한 모델 parameter를 이용하여, 우리의 데이터셋에 맞게 마지막 layer를 바꿔주고 학습하는 부분입니다.
- 모델 number range는 0~71(72개 입니다.)  
- 저장되는 모델 pt 파일명은 "모델명i_epochs10.pt"
- 추가로 learning rate, data-location과 같은 argument들이 있으며, 모든 argument는 default 값을 finetune.py에서 설정할 수 있습니다.
- Tip : 쉘 스크립트를 사용하여 학습 자동화하기 -> training.sh 파일 작성 후 다음 명령어 실행
```bash
bash trining.sh
```

### 2. Individual Evaluation  
```bash
python main.py --eval-individual-models --name {모델명}
```
- finetune을 통해 만든 모델들의 accuracy를 측정하여 기록해두는 부분입니다.
![image](https://user-images.githubusercontent.com/113486402/233948441-7bab18bc-37f8-424b-a0fb-3223a37781b8.png)
- "입력하세요" 부분에 측정할 모델의 개수(NUM_MODELS), 사용할 epoch, val_ratio를 적어줍니다.
- val_ratio에 None 값을 입력하면, 전체 dataset에 대해 evaludation을 진행합니다. 
- finetune 당시에 random-seed를 설정해주었다면, None값을 넣어주면 안됩니다.
- 실행 결과로 logs 폴더 안에 각 모델의 accuracy가 적힌 jsonl 파일이 생성됩니다. 

### 3. Greedy Soup
```bash
python main.py --greedy-soup --name {모델명}
```  
- individual Evaluation에서 저장한 여러 모델의 accuracy정보를 내림차순으로 정렬합니다.  
- 정렬 기준으로 좋은 성능을 내는 모델들을 순서대로 불러와 greedy하게 조합하여(averaging) 더 좋은 성능을 내도록 하는 최종 모델을 생성합니다.
- 실행 결과 model 폴더 안에 greedy 모델이 저장됩니다.
- log 폴더 안에 변수 GREEDY_SOUP_LOG_FILE가 이름임 로그를 저장합니다. 해당 로그에는 averaging된 모델 정보가 저장됩니다.

## Code

There are 5 steps to reproduced the figure above: 1) downloading the models, 2) evaluating the individual models, 3) running the uniform soup, 4) running the greedy soup, and 5) making the plot.

Note that any of these steps can be skipped, i.e, you can immediately generate the plot above via `python main.py --plot`.
You can also run the greedy soup without evaluating the individual models.
This is because we have already completed all of the steps and saved the results files in this repository (i.e., `individual_model_results.jsonl`).
If you do decide to rerun a step, the corresponding results file or plot is deleted and regenerated.

The exception is step 1, downloading the models. If you wish to run steps 2, 3, or 4 you must first run step 1.

### Install dependencies and downloading datasets

To install the dependencies either run the following code or see [environment.md](environment.md) for more information.
```bash
conda env create -f environment.yml
conda activate model_soups
```

To download the datasets see [datasets.md](datasets.md). When required, set `--data-location` to the `$DATA_LOCATION` used in [datasets.md](datasets.md).

### Step 1: Downloading the models

```bash
python main.py --download-models --model-location <where models will be stored>
```
This will store models to `--model-location`.


### Step 2: Evaluate individual models

```bash
python main.py --eval-individual-models --data-location <where data is stored> --model-location <where models are stored>
```
Note that this will first delete then rewrite the file `individual_model_results.jsonl`.

### Step 3: Uniform soup

```bash
python main.py --uniform-soup --data-location <where data is stored> --model-location <where models are stored>
```
Note that this will first delete then rewrite the file `uniform_soup_results.jsonl`.

### Step 4. Greedy soup

```bash
python main.py --greedy-soup --data-location <where data is stored> --model-location <where models are stored>
```
Note that this will first delete then rewrite the file `greedy_soup_results.jsonl`.

### Step 5. Plot

```bash
python main.py --plot
```
Note that this will first delete then rewrite the file `figure.png`.

### Note

If you want, you can all steps with:
```bash
python main.py --download-models --eval-individual-models --uniform-soup --greedy-soup --plot --data-location <where data is stored> --model-location <where models are stored>
```

Also note: if you are interested in running ensemble baselines, check out [the ensemble branch](https://github.com/mlfoundations/model-soups/tree/ensemble).

Also note: if you are interested in running a minial example of [wise-ft](https://arxiv.org/abs/2109.01903), you can run `python wise-ft-example.py --download-models`. 

Also note: if you are interested in running minimal examples of zeroshot/fine-tuning, you can run `python zeroshot.py` or `python finetune.py`. See program arguments (i.e., run with `--help`) for more information. Note that these are minimal examples and do not contain rand-aug, mixup, or LP-FT.

### Questions

If you have any questions please feel free to raise an issue. If there are any FAQ we will answer them here.

### Authors

This project is by the following authors, where * denotes equal contribution (alphabetical ordering):
- [Mitchell Wortsman](https://mitchellnw.github.io/)
- [Gabriel Ilharco](http://gabrielilharco.com/)
- [Samir Yitzhak Gadre](https://sagadre.github.io/)
- [Rebecca Roelofs](https://twitter.com/beccaroelofs)
- [Raphael Gontijo-Lopes](https://raphagl.com/)
- [Ari S. Morcos](http://www.arimorcos.com/)
- [Hongseok Namkoong](https://hsnamkoong.github.io/)
- [Ali Farhadi](https://homes.cs.washington.edu/~ali/)
- [Yair Carmon*](https://www.cs.tau.ac.il/~ycarmon/)
- [Simon Kornblith*](https://simonster.com/)
- [Ludwig Schmidt*](https://people.csail.mit.edu/ludwigs/)


## Citing

If you found this repository useful, please consider citing:
```bibtex
@InProceedings{pmlr-v162-wortsman22a,
  title = 	 {Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time},
  author =       {Wortsman, Mitchell and Ilharco, Gabriel and Gadre, Samir Ya and Roelofs, Rebecca and Gontijo-Lopes, Raphael and Morcos, Ari S and Namkoong, Hongseok and Farhadi, Ali and Carmon, Yair and Kornblith, Simon and Schmidt, Ludwig},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {23965--23998},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/wortsman22a/wortsman22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/wortsman22a.html}
}


```
