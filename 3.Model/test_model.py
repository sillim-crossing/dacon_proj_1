import os
import re
import json
import ast
import pandas as pd
import numpy as np
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import faiss  # <-- 기존에 벡터화용 라이브러리지만 여기선 안 쓰거나, 상황에 맞춰 제거 가능
import tqdm

# ==============================
# 1. 데이터 전처리 모듈
# ==============================
class DataProcessor:
    def __init__(self):
        pass

    def clean_text(self, text):
        if isinstance(text, str):
            text = re.sub(r'\s+', ' ', text).strip()  # 2개 이상 공백 및 앞뒤 공백 제거
        else:
            text = ""
        return text

    def load_train_data(self, filepath):
        """
        train.csv 예시 컬럼: ['질문', '답변'] 또는
        ['작업프로세스', '사고원인', '재발방지대책 및 향후조치계획'] 등
        """
        df = pd.read_csv(filepath)
        # 예시: '작업프로세스', '사고원인', '재발방지대책' 등을 clean
        # 실제 컬럼명에 맞춰 수정
        for col in ['작업프로세스', '사고원인', '재발방지대책 및 향후조치계획']:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_text)
        return df

    def load_test_data(self, filepath):
        """
        test.csv 예시 컬럼: ['질문'] 혹은
        ['작업프로세스', '사고원인'] 등
        """
        df = pd.read_csv(filepath)
        for col in ['작업프로세스', '사고원인']:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_text)
        return df

# ==============================
# 2. CSV를 활용한 PDF 문장 검색 모듈
# ==============================
class PDFCsvRetriever:
    """
    pdf_info.csv 예시 컬럼:
    - '질문'
    - '키워드'
    - 'PDF 파일명'
    - '유사도'
    - 'PDF 내 관련 문장' (문장 리스트나 문자열)
    """
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        # 'PDF 내 관련 문장'이 리스트 형태로 저장돼있다면 literal_eval로 파싱해야 할 수도 있음
        if isinstance(self.df.iloc[0]['PDF 내 관련 문장'], str):
            try:
                # 샘플 첫 행을 파싱 시도 → 전체를 일괄 처리
                sample_val = ast.literal_eval(self.df.iloc[0]['PDF 내 관련 문장'])
                # 성공 시 전체에 적용
                self.df['PDF 내 관련 문장'] = self.df['PDF 내 관련 문장'].apply(lambda x: ast.literal_eval(x))
            except:
                # 이미 문자열이거나 파싱 불가능하면 그대로 둠
                pass

    def search(self, query, top_k=3):
        """
        이미 CSV에 질문별 유사도가 계산돼 있다면,
        1) query와 동일(또는 유사)한 질문 행만 필터링
        2) 유사도 기준 내림차순 정렬
        3) 상위 top_k 'PDF 내 관련 문장' 반환
        """
        # (1) 질문 일치하는 행만 필터링 (예시: 정확 일치)
        # 만약 실제로는 부분 일치나 Embedding Search를 하고 싶다면 추가 로직 필요
        filtered = self.df[self.df['질문'] == query].copy()

        if len(filtered) == 0:
            # CSV에 없는 새로운 질문일 경우, 빈 리스트 반환
            return []

        # (2) 유사도 높은 순 정렬
        filtered.sort_values(by='유사도', ascending=False, inplace=True)
        top_rows = filtered.head(top_k)

        # (3) PDF 내 관련 문장들을 하나로 합치거나 리스트로 반환
        # 여기서는 여러 행의 여러 문장을 이어붙여 하나의 문장으로 만드는 예시
        merged_context = []
        for row in top_rows.itertuples():
            # 'PDF 내 관련 문장'이 list라면 하나씩 추출
            if isinstance(row._5, list):  # row._5 == row['PDF 내 관련 문장']
                for txt_tuple in row._5:
                    # txt_tuple이 (파일명, 문장) 형태라면 문장만 추출
                    if isinstance(txt_tuple, tuple) and len(txt_tuple) > 1:
                        merged_context.append(txt_tuple[1])
                    else:
                        merged_context.append(str(txt_tuple))
            else:
                # 문자열인 경우 그대로
                merged_context.append(str(row._5))

        # 최종적으로는 하나의 긴 문자열로 합치거나, 리스트 그대로 반환 가능
        final_context = " ".join(merged_context)
        return [final_context]

# ==============================
# 3. 한국어 전용 생성 모듈 (KoBART 기반 RAG)
# ==============================
class KoreanRAGGenerator:
    def __init__(self, passages_file='passages.json', embed_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart")
        self.model = BartForConditionalGeneration.from_pretrained("hyunwoongko/kobart")

        # passages.json이 기존 RAG 용도로 만든 파일이라면
        with open(passages_file, "r", encoding="utf-8") as f:
            self.passages = json.load(f)
        self.passage_texts = [p["text"] for p in self.passages]

        self.embedder = SentenceTransformer(embed_model_name)
        self.passage_embeddings = self.embedder.encode(self.passage_texts, convert_to_tensor=True)

    def retrieve(self, query, top_k=1):
        query_emb = self.embedder.encode(query, convert_to_tensor=True)
        sims = util.cos_sim(query_emb, self.passage_embeddings)[0]
        top_results = torch.topk(sims, k=top_k)
        indices = top_results.indices.tolist()
        retrieved_texts = [self.passage_texts[i] for i in indices]
        return retrieved_texts

    def generate(self, query, max_length=200, num_beams=5):
        retrieved = self.retrieve(query, top_k=1)
        context = retrieved[0] if retrieved else ""
        final_query = query + " " + context
        inputs = self.tokenizer(final_query, return_tensors="pt", max_length=512, truncation=True)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            num_beams=num_beams,
            max_length=max_length,
            early_stopping=True
        )
        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return output

    @staticmethod
    def create_passages(train_df, passages_file='passages.json'):
        """
        기존 RAG 방식: train_df의 일부 컬럼을 passage로 만들어 저장
        """
        passages = []
        for idx, row in train_df.iterrows():
            passage = {
                "title": f"doc_{idx}",
                "text": row.get('작업프로세스', ''),  # 혹은 질문(쿼리)로 할 수도 있음
                "id": idx
            }
            passages.append(passage)
        with open(passages_file, "w", encoding="utf-8") as f:
            json.dump(passages, f, ensure_ascii=False, indent=4)
        print(f"Passages saved to {passages_file}")

# ==============================
# 4. LLM 프롬프트 엔지니어링 기반 생성 모듈 (T5)
# ==============================
class LLMGenerator:
    def __init__(self, model_name="t5-base"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def create_prompt(self, few_shot_examples, new_input):
        prompt = (
            "너는 건설공사 안전 전문가야. 다음 사고 정보를 바탕으로, 효과적인 사고 대책을 작성해줘.\n\n"
            f"{few_shot_examples}"
            f"사고 상황 및 원인: {new_input}\n대책:"
        )
        return prompt

    def generate(self, prompt, max_length=150, num_beams=5):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded

    def get_few_shot_examples(self, train_df, n_shot=3):
        """
        예시: train_df에서 상위 n_shot개를 추출하여 prompt에 넣을 few-shot 템플릿 생성
        """
        examples = ""
        for idx, row in train_df.head(n_shot).iterrows():
            # 실제 컬럼명에 맞춰 수정 (예: 질문 + 대책)
            example = f"사고 상황 및 원인: {row.get('작업프로세스','')} / {row.get('사고원인','')}\n대책: {row.get('재발방지대책 및 향후조치계획','')}\n\n"
            examples += example
        return examples

# ==============================
# 5. 지도학습 기반 모델 모듈 (Supervised, T5)
# ==============================
class SupervisedModel:
    def __init__(self, model_name="t5-base"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def generate(self, input_text, max_length=150, num_beams=5):
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        generated_ids = self.model.generate(
            input_ids=inputs['input_ids'],
            max_length=max_length,
            num_beams=num_beams
        )
        output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return output

# ==============================
# 6. Ensemble 예측 모듈
# ==============================
class EnsemblePredictor:
    def __init__(self, embed_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.embedder = SentenceTransformer(embed_model_name)

    def cosine_sim(self, text1, text2):
        emb1 = self.embedder.encode(text1, convert_to_tensor=True)
        emb2 = self.embedder.encode(text2, convert_to_tensor=True)
        cos_sim = util.cos_sim(emb1, emb2).item()
        return cos_sim

    def ensemble(self, preds):
        """
        단순 앙상블:
        S-BERT Cosine 유사도를 측정하여
        가장 '서로 간의' 유사도가 높은(공통적으로 비슷한) 예측을 선택
        """
        n = len(preds)
        scores = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    scores[i] += self.cosine_sim(preds[i], preds[j])
        best_idx = int(np.argmax(scores))
        return preds[best_idx]

# ==============================
# 7. 최종 파이프라인 및 Submission 생성
# ==============================
def main():
    # 파일 경로 설정 (사용 환경에 맞춰 수정)
    train_path = os.path.join(os.getcwd(), "..", "1.Data", "train.csv")
    test_path = os.path.join(os.getcwd(), "..", "1.Data", "test.csv")
    sample_sub_path = os.path.join(os.getcwd(), "..", "1.Data", "sample_submission.csv")
    pdf_info_path = os.path.join(os.getcwd(), "..", "1.Data", "pdf_info.csv")  # PDF 정보가 담긴 CSV

    # 1. 데이터 로드
    dp = DataProcessor()
    train_df = dp.load_train_data(train_path)
    test_df = dp.load_test_data(test_path)

    # 2. PDF CSV 기반 검색기 준비
    pdf_retriever = PDFCsvRetriever(pdf_info_path)

    # 3. RAG용 passages 생성 (옵션)
    passages_file = "passages.json"
    KoreanRAGGenerator.create_passages(train_df, passages_file)
    rag_generator = KoreanRAGGenerator(passages_file)

    # 4. LLM(T5) 기반 생성 준비 (Few-shot 예시 포함)
    llm_generator = LLMGenerator("t5-base")
    few_shot_examples = llm_generator.get_few_shot_examples(train_df, n_shot=3)

    # 5. 지도학습 기반 모델 준비 (T5)
    supervised_model = SupervisedModel("t5-base")

    # 6. Ensemble 예측 모듈 준비
    ensemble_predictor = EnsemblePredictor("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # 추론 및 예측 결과 저장
    predictions = []
    for idx, row in tqdm.tqdm(test_df.iterrows(), total=len(test_df)):
        # test_df 컬럼 예시: ['작업프로세스', '사고원인', ...] → 질문 구성
        # 또는 row['질문']이 직접 있을 수도 있음
        # 여기서는 임의로 accident_info를 만든 예시
        accident_info = f"작업프로세스: {row.get('작업프로세스','')}, 사고원인: {row.get('사고원인','')}"

        # (1) CSV에서 해당 accident_info와 일치(또는 유사)하는 질문 행 찾아서 문맥 가져오기
        pdf_contexts = pdf_retriever.search(accident_info, top_k=1)
        domain_context = pdf_contexts[0] if pdf_contexts else ""

        # (2) 최종 입력: 도메인 지식 포함
        final_input = accident_info + " " + domain_context

        # (3) 각 모듈별 예측 생성
        pred_rag = rag_generator.generate(final_input)
        prompt = llm_generator.create_prompt(few_shot_examples, final_input)
        pred_llm = llm_generator.generate(prompt)
        pred_supervised = supervised_model.generate(final_input)

        # (4) 앙상블: 3가지 예측 중 서로 가장 비슷한(유사도 높은) 예측 선택
        pred_candidates = [pred_rag, pred_llm, pred_supervised]
        final_pred = ensemble_predictor.ensemble(pred_candidates)
        predictions.append(final_pred)

        print(f"Test idx {idx} 예측 완료. → {final_pred[:80]}...")

    # 7. Submission 파일 생성
    submission = pd.read_csv(sample_sub_path)
    # sample_submission.csv의 '재발방지대책 및 향후조치계획' 컬럼에 결과 할당 (컬럼명 맞춰야 함)
    submission['재발방지대책 및 향후조치계획'] = predictions
    submission.to_csv("final_submission.csv", index=False, encoding="utf-8-sig")
    print("Submission 파일 생성 완료: final_submission.csv")

if __name__ == "__main__":
    main()
