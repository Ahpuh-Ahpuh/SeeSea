# 🌊Seesea-Growth🌊
이화여자대학교 캡스톤 창업 디자인프로젝트 그로쓰 8팀 어푸어푸입니다


## :trophy: 캡스톤디자인경진대회 창업 아이디어 부문 동상 수상 :trophy:
## :trophy: 캡스톤 포스터세션 우수상 수상 :trophy:

## 프로젝트 소개
드론을 이용한 실시간 영상 인식과 수상 구조
![20230609_144947](https://github.com/Ahpuh-Ahpuh/SeeSea/assets/93649914/528e6d8c-63b9-4298-bf9e-b34f34d61b44)

## 프로젝트 목적

## 요구사항 정의
#### 1) 구조도
![image](https://github.com/Ahpuh-Ahpuh/SeeSea/assets/93649914/c022c4bc-75f6-495d-b4d5-f0f11f6665e2)
![image](https://github.com/Ahpuh-Ahpuh/SeeSea/assets/93649914/668e8f7b-2a6d-4f1f-83e3-97debc05db7c)

#### 2) 흐름도
![image](https://github.com/Ahpuh-Ahpuh/SeeSea/assets/93649914/1f99d7cf-7601-4e4e-90c9-d819ca21f0dd)

## 프로젝트 기술
#### 1) 관리해야 하는 해수욕장의 면적에 따라 드론들을 최적의 위치에 배치
사각지대가 발생하지 않도록 여러 대의 드론을 사각지대가 없게, 최소한의 수로 배치해주는 기술 구현
![image](https://github.com/Ahpuh-Ahpuh/SeeSea/assets/93649914/49ce98b5-9085-440b-bace-45849a9d2630)
![image](https://github.com/Ahpuh-Ahpuh/SeeSea/assets/93649914/68d82ace-c5fa-4b3d-8ea7-d6e41ed00cb1)

#### 2) YOLO v5를 이용하여 실시간 영상 분석
드론으로부터 실시간으로 영상을 받아와서 분석, YOLO v5 커스텀 후 사용
![image](https://github.com/Ahpuh-Ahpuh/SeeSea/assets/93649914/1e85e189-adf9-40a6-98c2-b3948dd51b36)
detection/yolov5/ma.py

#### 3) Depth Estimation을 이용하여 드론과 위험상황에 처한 사람 거리 분석
드론이 촬영하고 있는 영상을 170칸의 구역으로 나누어 bounding box의 중심좌표가 위치한 구역에 따라 거리와 방향 출력
![image](https://github.com/Ahpuh-Ahpuh/SeeSea/assets/93649914/038843c8-fbc0-4b02-a8d3-eacf344e7d33)
depth_estimation.ipynb 
#### 4) 이중 PI 제어로 드론 위치 고정
바람이 심한 바닷가 특성상 드론이 제자리를 고정할 수 있도록 이중 PI제어를 이용
pid 폴더 안에 있습니다.

#### 5) 시연 영상
https://www.youtube.com/watch?v=PKLmfuanSAU

## 어플 사용 방법

#### 1) 주피터 노트북에서 ma.app있는 위치(detection/yolov5/ma.py)에서 uvicorn ma:app --reload 실행

#### 2) 드론과 노트북 연결

#### 3) 안드로이드 스튜디오 에뮬레이터 실행

## 어플 화면
#### 1) 메인화면
![image](https://github.com/Ahpuh-Ahpuh/SeeSea/assets/93649914/2cac52fb-30e3-4163-a6f9-c14ec5d0ad9c)
#### 2) 드론 배치와 연결
![image](https://github.com/Ahpuh-Ahpuh/SeeSea/assets/93649914/08080cb8-715d-475b-b62e-4f0fcbf32254)
#### 3) 데모 영상 페이지
![image](https://github.com/Ahpuh-Ahpuh/SeeSea/assets/93649914/99c302de-a797-49c8-bf1b-322bf80e3ecb)
#### 4) 알람 전송
![image](https://github.com/Ahpuh-Ahpuh/SeeSea/assets/93649914/6aff583b-9b63-473a-b5ff-ff38d6796297)


