import os
import re
import jpype
import pandas as pd
from konlpy.tag import Okt

# 🔹 **Okt 객체 생성**
okt = Okt()

# 🔹 **불용어 리스트 (중요 키워드는 포함 X)**
stop_words = ["및", "등", "이상", "이하", "대한", "경우", "제", "그", "이", "저", "이런", "저런", "각종",
              "하는", "하는것", "할", "하고", "한다", "한", "되어", "되며", "하는데", "하는지", "하게", "하면", "하면서",
              "수", "있다", "없는", "있으며", "있어야", "없음", "없다", "때문", "때문에", "하여", "하여야", "하였다"]

# 🔹 **중요 조사는 유지 (문장 구조 개선)**
essential_josa = ["이", "가", "은", "는", "에", "을", "를"]

def preprocessing(text, remove_stopwords=True):
    """
    ✅ 사고원인 및 재발방지대책 전처리 함수 (Okt 기반)
    - 특수기호, 숫자 제거
    - 형태소 분석 후 표제어 추출
    - 불용어 및 필요 없는 조사 제거 후 문장 복원
    """
    if not isinstance(text, str):  # 문자열이 아닌 경우 빈 문자열 반환
        return ""

    # 1️⃣ **한글 + 공백 제외한 문자(숫자, 특수기호, 영어) 제거**
    text = re.sub(r"[^가-힣\s]", "", text)

    # 2️⃣ **형태소 분석 및 표제어 추출**
    words = okt.pos(text, stem=True)  # 품사 태깅 진행

    # 3️⃣ **불용어 및 필요 없는 조사 제거**
    filtered_words = []
    for word, tag in words:
        if remove_stopwords and word in stop_words:
            continue
        if tag == "Josa" and word not in essential_josa:  # 중요한 조사는 유지
            continue
        filtered_words.append(word)

    # 4️⃣ **문장 구조를 자연스럽게 복원**
    cleaned_text = " ".join(filtered_words)

    return cleaned_text
  
# 🔹 **경로 설정**
script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(script_dir, "..", "1.Data", "train.csv")
test_path = os.path.join(script_dir, "..", "1.Data", "test.csv")

# ✅ **(1) `train.csv`와 `test.csv` 불러오기**
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# ✅ **(2) 불필요한 열 삭제**
drop_columns = ["발생일시", "사고인지 시간", "날씨", "기온", "습도", "연면적", "층 정보", "물적사고", "부위"]
train.drop(columns=drop_columns, inplace=True)
test.drop(columns=drop_columns, inplace=True)

# ✅ **(3) `공사종류`, `공종`, `사고객체`를 대분류/중분류로 나누기**
def split_columns(df): 
    df['공종(중분류)'] = df['공종'].str.split(' > ').str[1]
    df['사고객체(대분류)'] = df['사고객체'].str.split(' > ').str[0]
    df['사고객체(중분류)'] = df['사고객체'].str.split(' > ').str[1]
    df['인적사고(대분류)'] = df['인적사고'].str.split('(').str[0].str.strip()
    df['인적사고(중분류)'] = df['인적사고'].str.split('(').str[1].str.replace(')', '').str.strip()

split_columns(train)
split_columns(test)

# ✅ **(4) 사고원인 & 재발방지대책 전처리**
train["사고원인_정제"] = train["사고원인"].apply(lambda x: preprocessing(x, remove_stopwords=True))
train["재발방지대책_정제"] = train["재발방지대책 및 향후조치계획"].apply(lambda x: preprocessing(x, remove_stopwords=True))
test["사고원인_정제"] = test["사고원인"].apply(lambda x: preprocessing(x, remove_stopwords=True))

# ✅ **(5) 훈련 데이터 통합 생성**
# 훈련 데이터 통합 생성
combined_training_data = train.apply(
    lambda row: {
        "question": (  
            f"공종 중분류 '{row['공종(중분류)']}' 작업에서 "
            f"인적사고 대분류 '{row['인적사고(대분류)']}', 중분류 '{row['인적사고(중분류)']}' "
            f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
            f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. "
            f"재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
        ),
        "answer": row["재발방지대책 및 향후조치계획"]
    },
    axis=1
)
# DataFrame으로 변환
combined_training_data = pd.DataFrame(list(combined_training_data))


# 테스트 데이터 통합 생성
combined_test_data = test.apply(
    lambda row: {
        "question": (
            f"공종 중분류 '{row['공종(중분류)']}' 작업에서 "
            f"인적사고 대분류 '{row['인적사고(대분류)']}', 중분류 '{row['인적사고(중분류)']}' "
            f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
            f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. "
            f"재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
        )
    },
    axis=1
)

# DataFrame으로 변환
combined_test_data = pd.DataFrame(list(combined_test_data))


# ✅ **(8) 저장**
train_cleaned_path = os.path.join(script_dir, "..", "1.Data", "train_cleaned.csv")
test_cleaned_path = os.path.join(script_dir, "..", "1.Data", "test_cleaned.csv")

combined_training_data.to_csv(train_cleaned_path, index=False, encoding="utf-8-sig")
combined_test_data.to_csv(test_cleaned_path, index=False, encoding="utf-8-sig")

print(f"\n✅ 전처리 완료! 저장된 파일: {train_cleaned_path}, {test_cleaned_path} 🚀")






