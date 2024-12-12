# LangEyE

## 프로젝트 개요

분산된 복지 정보를 통합한 시각장애인을 위한 보이스 챗 봇

## 프로젝트 환경
- Ubuntu 22.04
- cuda 11.8.0
- python 3.10
- RTX A5000 24GB

## 수집 데이터
1. 보건복지 상담 센터
2. 보건복지부(복지로)
3. 국가법령정보센터

## 사용방법

```bash
sh lang-env.sh
```

1. STT
    - [Model Weight](https://drive.google.com/drive/folders/1Adv8kYXV1XGGoLY1XA36EI38kfk0r0WZ) 다운로드
    - App > models > STT > checkpoint 폴더에 `denoiser.th`, `Conformer-CTC-BPE.nemo` 파일 저장

2. RAG
    - src > weights 폴더에 weights 들 저장

```bash
python app_gradio_retrieval.py
```

을 통해 gradio 실행
