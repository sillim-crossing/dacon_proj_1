# 합친 pdf문서를 문장 단위로 나누어 전처리 해주는 코드입니다.

import os
import pandas as pd
import re

# 📂 파일 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_file_path = os.path.join(script_dir, "..", "1.Data", "전체_PDF_정제본.csv")  # 기존 PDF 데이터

def split_sentences(text):
    """✅ PDF 문서를 문장 단위로 나누기"""
    text = re.sub(r'\n+', ' ', text)  # 줄바꿈 제거 후 공백 처리
    sentences = re.split(r'(?<=[.!?])\s+', text)  # 문장 분리
    return [sentence.strip() for sentence in sentences if sentence.strip()]  # 빈 문장 제거

# 🔹 **CSV 파일 로드 (PDF 텍스트)**
pdf_df = pd.read_csv(pdf_file_path)

# 🔹 **"텍스트" 컬럼을 문장 단위로 나누고 처리**
processed_data = []
for idx, row in pdf_df.iterrows():
    pdf_name = row["파일명"]
    category = row["카테고리"]
    text = row["텍스트"]

    # 🔸 1. **처음 등장하는 "4."부터 유지 (1,2,3 삭제)**
    section4_match = re.search(r"\n4\.\s*[가-힣]", text)
    if section4_match:
        text = text[section4_match.start():].strip()
    else:
        continue  # "4."가 없는 경우 해당 행 삭제

    # 🔸 2. **문장 단위 분리**
    sentences = split_sentences(text)
 
    # 🔸 3. **빈 문장 제거 후 리스트에 저장**
    for sentence in sentences:
        if sentence.strip():  # 공백이 아닌 경우만 저장
            processed_data.append([pdf_name, category, sentence])

# 🔹 **DataFrame 변환**
pdf_processed_df = pd.DataFrame(processed_data, columns=["파일명", "카테고리", "문장"])

# 🔸 4. **텍스트가 공백인 행 전체 삭제**
pdf_processed_df = pdf_processed_df.dropna(subset=["문장"]).reset_index(drop=True)

# 🔹 **CSV로 저장**
output_csv = os.path.join(script_dir, "..", "1.Data", "PDF_문장_처리.csv")  # 결과 저장
pdf_processed_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"✅ 문장 단위 전처리 완료! 저장 위치: {output_csv}")
