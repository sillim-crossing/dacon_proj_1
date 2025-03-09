# pdf문서의 문장들과, train data의 문장들간의 코사인 유사도를 계산하여 가장 참고할만한 pdf를 가져옵니다.

import os
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util

# 📂 파일 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
train_file_path = os.path.join(script_dir, "..", "1.Data", "train_cleaned.csv")  # train 데이터
pdf_processed_file_path = os.path.join(script_dir, "..", "1.Data", "PDF_문장_처리.csv")  # 문장 단위 PDF 데이터

# 🔹 **Sentence-BERT 모델 로드 (KoBERT STS)**
model = SentenceTransformer("jhgan/ko-sbert-sts")

# 🔹 **데이터 로드**
train_df = pd.read_csv(train_file_path)
pdf_df = pd.read_csv(pdf_processed_file_path)

# 🔹 **10개 샘플링 (question만)**
sample_questions = train_df[["question"]].head(10)

# 🔹 **PDF 문장 리스트 및 임베딩 생성**
pdf_sentences = pdf_df["문장"].tolist()
pdf_embeddings = model.encode(pdf_sentences, convert_to_tensor=True)

# 🔹 **유사도 계산 및 매칭**
matched_results = []
for idx, row in sample_questions.iterrows():
    question = row["question"]

    # ✅ "question" 문장 임베딩
    question_embedding = model.encode(question, convert_to_tensor=True)

    # ✅ 코사인 유사도 계산
    similarities = util.pytorch_cos_sim(question_embedding, pdf_embeddings)[0]

    # ✅ 가장 유사한 문장 찾기
    best_match_idx = similarities.argmax().item()
    best_match_sentence = pdf_sentences[best_match_idx]
    best_match_pdf = pdf_df.iloc[best_match_idx]["파일명"]  # 유사한 문장이 있는 PDF 파일명

    # ✅ 결과 저장
    matched_results.append([question, best_match_sentence, best_match_pdf, similarities[best_match_idx].item()])

# 🔹 **결과 DataFrame 변환 및 저장**
result_df = pd.DataFrame(matched_results, columns=["질문", "유사한 PDF 문장", "PDF 파일명", "유사도"])
output_csv = os.path.join(script_dir, "..", "1.Data", "유사도_매칭_결과.csv")  # 결과 저장
result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"✅ 유사도 매칭 완료! 저장 위치: {output_csv}")
