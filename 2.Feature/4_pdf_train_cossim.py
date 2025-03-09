# pdf문서의 문장들과, train data의 문장들간의 코사인 유사도를 계산하여 가장 참고할만한 pdf를 가져옵니다.

import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import re

# 📂 파일 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
train_file_path = os.path.join(script_dir, "..", "1.Data", "train_cleaned.csv")  # train 데이터
pdf_processed_file_path = os.path.join(script_dir, "..", "1.Data", "PDF_문장_처리.csv")  # 문장 단위 PDF 데이터

# 🔹 Sentence-BERT 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("jhgan/ko-sbert-sts", device=device)

# 🔹 데이터 로드
train_df = pd.read_csv(train_file_path)
pdf_df = pd.read_csv(pdf_processed_file_path)

# 🔹 PDF 파일을 문서 단위로 결합 (문장 → 파일별 텍스트 합치기)
pdf_grouped = pdf_df.groupby("파일명")["문장"].apply(lambda x: " ".join(x)).reset_index()

# ✅ **Step 1: train 질문에서 `''` 안의 키워드만 추출**
def extract_keywords(text):
    """질문에서 `''` 안의 키워드만 추출"""
    return re.findall(r"'(.*?)'", text)

train_df["키워드"] = train_df["question"].apply(extract_keywords)

# ✅ **Step 2: PDF에서 키워드가 골고루 포함된 문서 선택**
def compute_keyword_coverage_score(pdf_text, keywords):
    """PDF 문서에서 키워드가 얼마나 골고루 포함되었는지 점수화"""
    unique_matches = set()
    total_matches = 0

    for word in keywords:
        if word in pdf_text:
            unique_matches.add(word)  # 포함된 키워드 개수 계산
            total_matches += pdf_text.count(word)  # 전체 등장 횟수

    if len(keywords) == 0:
        return 0

    score = len(unique_matches) / len(keywords)  # 키워드 균형 점수 (0~1)
    return score

def find_best_pdfs_for_question(keywords):
    """키워드가 골고루 포함된 PDF 파일 3~5개 선택"""
    pdf_scores = {}

    for _, row in pdf_grouped.iterrows():
        pdf_text = row["문장"]
        score = compute_keyword_coverage_score(pdf_text, keywords)
        if score > 0:
            pdf_scores[row["파일명"]] = score  # PDF 파일별 점수 저장

    return sorted(pdf_scores, key=pdf_scores.get, reverse=True)[:5]

train_df["관련 PDF 파일"] = train_df["키워드"].apply(find_best_pdfs_for_question)

# ✅ **Step 3: 선택된 PDF 파일에서 키워드가 포함된 문장 분석**
def extract_relevant_sentences(pdf_df, pdf_filenames, keywords):
    """PDF에서 키워드가 포함된 문장 3~5개 추출"""
    relevant_sentences = []
    
    for _, row in pdf_df.iterrows():
        if row["파일명"] in pdf_filenames:
            for keyword in keywords:
                if keyword in row["문장"]:
                    relevant_sentences.append((row["파일명"], row["문장"]))

    return relevant_sentences[:5]  # 최대 5개 문장 반환

# ✅ **Step 4: 선택된 PDF 파일과 질문 전체를 비교하여 유사도 분석**
matched_results = []
total_questions = len(train_df)

print(f"🔥 전체 {total_questions}개의 질문을 분석 시작...")

for idx, row in train_df.iloc[:200].iterrows():
    question = row["question"]
    keywords = row["키워드"]
    related_pdfs = row["관련 PDF 파일"]

    # 🔹 진행률 출력
    if idx % 50 == 0:
        print(f"🔄 진행 중: {idx}/{total_questions} ({(idx/total_questions)*100:.2f}%) 완료")

    # 🔹 선택된 PDF 파일에서 텍스트 가져오기
    selected_pdf_texts = pdf_grouped[pdf_grouped["파일명"].isin(related_pdfs)]["문장"].tolist()
    selected_pdf_filenames = pdf_grouped[pdf_grouped["파일명"].isin(related_pdfs)]["파일명"].tolist()

    # 🔹 선택된 PDF가 없으면 패스
    if not selected_pdf_texts:
        continue

    # 🔹 질문 임베딩 생성
    question_embedding = model.encode(question, convert_to_tensor=True, device=device)

    # 🔹 PDF 문서(파일 단위) 임베딩 생성
    pdf_embeddings = model.encode(selected_pdf_texts, convert_to_tensor=True, device=device)

    # 🔹 유사도 계산
    similarities = util.pytorch_cos_sim(question_embedding, pdf_embeddings)[0]

    # 🔹 가장 유사한 PDF 찾기
    best_match_idx = similarities.argmax().item()
    best_match_text = selected_pdf_texts[best_match_idx]
    best_match_pdf = selected_pdf_filenames[best_match_idx]
    best_score = similarities[best_match_idx].item()

    # 🔹 **해당 PDF에서 키워드가 포함된 주요 문장 3~5개 추출**
    top_matched_sentences = extract_relevant_sentences(pdf_df, [best_match_pdf], keywords)

    # ✅ 결과 저장
    matched_results.append([question, keywords, best_match_pdf, best_match_text, best_score, top_matched_sentences])

print("✅ 모든 질문 분석 완료!")

# ✅ **결과 저장**
result_df = pd.DataFrame(matched_results, columns=["질문", "키워드", "PDF 파일명", "유사한 PDF 전체 내용", "유사도", "PDF 내 관련 문장"])
output_csv = os.path.join(script_dir, "..", "1.Data", "유사도_매칭_문맥분석_전체.csv")
result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"✅ 문맥 분석 완료! 저장 위치: {output_csv}")
