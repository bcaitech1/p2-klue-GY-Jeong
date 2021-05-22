  # Boostcamp AI Tech Stage 2 - KLUE (Relation extraction)

  ## Report

 https://www.notion.so/rmsdud/Wrap-up-03a0acb195a84678b89e52b796e40d2d

  ## 소개

 관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다. 관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요합니다. 비구조적인 자연어 문장에서 구조적인 triple을 추출해 정보를 요약하고, 중요한 성분을 핵심적으로 파악할 수 있습니다.

![image](https://user-images.githubusercontent.com/28976984/119237969-adba3d00-bb7a-11eb-8b0b-66b57f947bc8.png)

 위 그림의 예시와 같이 요약된 정보를 사용해 QA 시스템 구축과 활용이 가능하며, 이외에도 요약된 언어 정보를 바탕으로 효율적인 시스템 및 서비스 구성이 가능합니다.

 이번 대회에서는 문장, 엔티티, 관계에 대한 정보를 통해 ,문장과 엔티티 사이의 관계를 추론하는 모델을 학습시킵니다. 이를 통해 우리의 인공지능 모델이 엔티티들의 속성과 관계를 파악하며 개념을 학습할 수 있습니다. 우리의 model이 정말 언어를 잘 이해하고 있는 지, 평가해 보도록 합니다.


=============== 여기서부터 채우자 ==================
  ### 분류 방법

  - 마스크 착용 여부 : 착용 / 잘못된 착용 / 미착용
  - 성별 : 남 / 여
  - 연령 : 30대 미만 / 30대 이상 ~ 60대 미만 / 60대 이상

  총 18개의 label 분류

  ## 모델 소개

  ### Backbone

  - `EfficientNet-b5` (https://github.com/lukemelas/EfficientNet-PyTorch) model fine-tuning
  - `ViT` (Visison Transformer, https://github.com/google-research/vision_transformer) model fine-tuning
    - Ensemble시 다른 특징 적용 가능성

  ### Loss

  - `F1 loss` + `Focal loss` (gamma = 5)

  ### Training time augmentaion

  - `Center crop` (384 * 384)
  - 대비 제한 적응 히스토그램 평활화(`CLAHE`: Contrast-limited adaptive histogram equalization)

  ### Optimizer

  - `AdamP`

  ### Tensorboard log

  </center><img src="https://user-images.githubusercontent.com/28976984/119237935-85324300-bb7a-11eb-8e3b-256ca320617e.png" width="450" height="450"></center>

  ## 모델 성능

  - F1 Score : 0.7660
  - Accuracy : 81.1111%

## Getting Started

### Training

* python train.py

### Inference

* python inference.py --model_dir=[model_path]
* ex) python inference.py --model_dir=./results/checkpoint-500

### Evaluation

* python eval_acc.py
