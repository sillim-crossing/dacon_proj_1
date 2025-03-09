# pip install PDFPlumber 
# pdf들을 하나로 합쳐주는 코드입니다.

import numpy as np
import pandas as pd
import os
import re
import pdfplumber


script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_folder = os.path.join(script_dir, "..", "1.Data", "PDF")
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]


# 🏗️ 새로운 카테고리별 키워드 매칭 (파일명에서만 띄어쓰기 수정)
CATEGORY_KEYWORDS = {
    "안전작업 지침": ["안전작업지침", "안전 작업 지침"],
    "안전보건작업 지침": ["안전보건작업지침", "안전 보건 작업 지침"],
    "안전보건작업 기술지침": ["안전보건작업 기술지침", "안전 보건 작업 기술 지침"],
    "기술지침": ["기술지침", "기술 지침"],
    "설계지침": ["설계지침", "설계 지침"],
    "설치지침": ["설치지침", "설치 지침"],
    "사용안전 지침": ["사용안전지침", "사용 안전 지침"],
    "작성지침": ["작성지침", "작성 지침"]
}

def match_category(file_name):
    """파일명을 기반으로 적절한 카테고리를 찾는 함수"""
    normalized_name = file_name.replace(" ", "")  # 파일명에서만 띄어쓰기 제거
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword.replace(" ", "") in normalized_name:
                return category
    return "기타"  # 매칭되지 않는 경우 기타로 분류



###############################################################3

# 🏗️ 주요 항목 리스트 (모든 PDF에서 적용)
MAJOR_SECTIONS = ["목적", "적용범위", "용어의 정의", "안전관리", "작업방법", "설계기준", "안전작업사항"]

def clean_text(text, file_name):
    """불필요한 텍스트 삭제 + 강제 개행 제거 + 주요 항목 개행 정리 + 파일명 삭제"""

    # ✅ (1) "KOSHA GUIDE C - 30 - 2020" 형태 삭제
    text = re.sub(r'KOSHA GUIDE\s*C\s*-\s*\d+\s*-\s*\d{4}', '', text)

    # ✅ (2) `- 1 -` 같은 페이지 넘버 삭제
    text = re.sub(r'-\s*\d+\s*-', '', text)

    # ✅ (3) `<그림 n>` 및 같은 줄의 설명 삭제
    text = re.sub(r'<그림\s*\d+>[^\n]*', '', text)

    # ✅ (4) 주요 항목 번호 + 제목을 한 줄로 유지 (`2. \n적용범위` → `\n\n2. 적용범위\n`)
    major_sections = ["목적", "적용범위", "용어의 정의", "안전관리", "작업방법", "설계기준", "안전작업사항"]
    for section in major_sections:
        text = re.sub(rf'\n*(\d+)\.\s*\n*{section}', rf'\n\n\1. {section}\n', text)

    # ✅ (5) 세부 항목 (`(6)`, `(7)`, `(8)`) 개행 제거 (설명이 자연스럽게 연결되도록)
    text = re.sub(r'\(\s*(\d+)\s*\)\s*\n', r'(\1) ', text)

    # ✅ (6) 주요 항목과 설명 부분은 개행 유지, 설명 내부의 강제 개행 제거
    lines = text.split("\n")
    processed_lines = []
    for i, line in enumerate(lines):
        if re.match(r"^\d+\.\s*[가-힣]", line):  # 주요 항목(1. 목적 등)은 개행 유지
            processed_lines.append("\n" + line)
        elif re.match(r"^\(\d+\)", line):  # 세부 항목 ((6), (7) 등)은 개행 유지
            processed_lines.append("\n" + line)
        elif i > 0 and re.match(r"^\s*[가-힣]", line):  # 설명 내부의 강제 개행 제거
            processed_lines[-1] += " " + line.strip()
        else:
            processed_lines.append(line)

    text = "\n".join(processed_lines)

    # ✅ (7) 파일명과 동일한 단어가 `1. 목적` 전에 나오면 삭제
    normalized_filename = os.path.splitext(file_name)[0].replace(" ", "")
    text_lines = text.split("\n")
    
    new_text_lines = []
    found_purpose = False

    for line in text_lines:
        # `1. 목적`이 나오기 전까지 파일명과 동일한 단어 제거
        if "1. 목적" in line:
            found_purpose = True
        if not found_purpose and normalized_filename in line.replace(" ", ""):
            continue
        new_text_lines.append(line)

    text = "\n".join(new_text_lines)

    return text.strip()

def extract_text_from_pdf(pdf_path, file_name):
    """PDF에서 첫 2페이지 제외하고 나머지 텍스트를 `pdfplumber`로 추출"""
    extracted_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        # 🔹 3페이지를 확인하여 "목차"가 있으면 4페이지부터 추출
        start_page = 2
        if pdf.pages[2].extract_text() and pdf.pages[2].extract_text().strip().startswith("목차"):
            start_page = 3  # 4페이지부터 추출

        # 🔹 start_page부터 마지막 페이지까지 텍스트 추출
        for page_num in range(start_page, len(pdf.pages)):  
            page = pdf.pages[page_num]
            extracted_text += page.extract_text() + "\n" if page.extract_text() else ""

    # ✅ 불필요한 텍스트 삭제 및 문단 구별 적용
    extracted_text = clean_text(extracted_text, file_name)

    return extracted_text

def process_all_pdfs(output_csv):
    """전체 PDF 파일을 처리하고 정제된 텍스트를 CSV로 저장 (카테고리 추가)"""
    data = []

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    if not pdf_files:
        print("⚠️ PDF 파일이 없습니다. 작업을 중지합니다.")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        category = match_category(pdf_file)  # 카테고리 자동 매칭

        print(f"📄 처리 중: {pdf_file} ({category})")

        text = extract_text_from_pdf(pdf_path, pdf_file)

        if not text.strip():
            print(f"⚠️ {pdf_file} (페이지 시작 기준 후 텍스트 없음) - 제외됨")
            continue

        data.append([pdf_file, category, text])

    df = pd.DataFrame(data, columns=["파일명", "카테고리", "텍스트"])
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ 완료! 결과가 {output_csv}에 저장됨")

output_csv = os.path.join(script_dir, "..", "1.Data", "전체_PDF_정제본.csv")
process_all_pdfs(output_csv)
