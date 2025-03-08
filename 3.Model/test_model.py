# submission.py
import os
import re
import json
import pandas as pd
import numpy as np
import torch
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration, T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import faiss
import pdfplumber  # PDF 파싱을 위한 라이브러리

# ==============================
# 1. 데이터 전처리 모듈
# ==============================
class DataProcessor:
    def __init__(self):
        pass

    def clean_text(self, text):
        if isinstance(text, str):
            text = re.sub(r'\s+', ' ', text).strip()
        else:
            text = ""
        return text

    def load_train_data(self, filepath):
        df = pd.read_csv(filepath)
        # 컬럼명은 train.csv에 따라 조정 (예: '온도', '습도', '사고발생상황', '원인', '대책')
        for col in ['사고발생상황', '원인', '대책']:
            df[col] = df[col].apply(self.clean_text)
        return df

    def load_test_data(self, filepath):
        df = pd.read_csv(filepath)
        # test 데이터에도 동일한 전처리 적용
        for col in ['사고발생상황', '원인']:
            df[col] = df[col].apply(self.clean_text)
        return df

# ==============================
# 2. PDF 건설안전지침 벡터 DB 구축 모듈
# ==============================
class PDFVectorizer:
    def __init__(self, pdf_folder, embed_model_name='jhgan/ko-sbert-sts'):
        self.pdf_folder = pdf_folder
        self.embedder = SentenceTransformer(embed_model_name)
        self.texts = []
        self.ids = []
        self.index = None

    def parse_pdf(self, pdf_path):
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + " "
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
        return text.strip()

    def vectorize_pdfs(self):
        file_list = [f for f in os.listdir(self.pdf_folder) if f.endswith('.pdf')]
        for i, file in enumerate(file_list):
            full_path = os.path.join(self.pdf_folder, file)
            txt = self.parse_pdf(full_path)
            if txt:
                self.texts.append(txt)
                self.ids.append(i)
        if self.texts:
            embeddings = self.embedder.encode(self.texts, convert_to_numpy=True)
            d = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(embeddings)
            print(f"Indexed {len(self.texts)} PDF documents.")
        else:
            print("No PDF texts found.")

    def search(self, query, top_k=3):
        if self.index is None:
            return []
        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_emb, top_k)
        results = [self.texts[i] for i in I[0] if i < len(self.texts)]
        return results

# ==============================
# 3. RAG 기반 생성 모듈
# ==============================
class RAGGenerator:
    def __init__(self, passages_file='passages.json'):
        # passages_file는 train 데이터의 사고발생상황을 JSON으로 저장한 파일
        self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        self.retriever = RagRetriever.from_pretrained("facebook/rag-token-nq",
                                                      index_name="custom",
                                                      passages_path=passages_file)
        self.model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")

    def generate(self, query, max_length=200, num_beams=5):
        inputs = self.tokenizer(query, return_tensors="pt")
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            num_beams=num_beams,
            max_length=max_length
        )
        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return output

    @staticmethod
    def create_passages(train_df, passages_file='passages.json'):
        passages = []
        for idx, row in train_df.iterrows():
            passage = {
                "title": f"doc_{idx}",
                "text": row['사고발생상황'],
                "id": idx
            }
            passages.append(passage)
        with open(passages_file, "w", encoding="utf-8") as f:
            json.dump(passages, f, ensure_ascii=False, indent=4)
        print(f"Passages saved to {passages_file}")

# ==============================
# 4. LLM 프롬프트 엔지니어링 기반 생성 모듈
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
        examples = ""
        # n_shot 만큼 예시 추출 (상황, 원인, 대책)
        for idx, row in train_df.head(n_shot).iterrows():
            example = f"사고 상황 및 원인: {row['사고발생상황']} / {row['원인']}\n대책: {row['대책']}\n\n"
            examples += example
        return examples

# ==============================
# 5. 지도학습 기반 모델 모듈 (Supervised)
# ==============================
class SupervisedModel:
    def __init__(self, model_name="t5-base"):
        # 실제로는 미세조정된 모델을 로드할 것
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def generate(self, input_text, max_length=150, num_beams=5):
        inputs = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(
            inputs, 
            max_length=max_length, 
            num_beams=num_beams, 
            early_stopping=True
        )
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result

# ==============================
# 6. Ensemble 예측 모듈
# ==============================
class EnsemblePredictor:
    def __init__(self, embed_model_name='jhgan/ko-sbert-sts'):
        self.embedder = SentenceTransformer(embed_model_name)

    def cosine_sim(self, text1, text2):
        emb1 = self.embedder.encode(text1, convert_to_tensor=True)
        emb2 = self.embedder.encode(text2, convert_to_tensor=True)
        cos_sim = util.cos_sim(emb1, emb2).item()
        return cos_sim

    def ensemble(self, preds):
        # 단순 앙상블: S-BERT Cosine 유사도를 측정하여 가장 공통된(유사한) 예측 선택
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
    # 파일 경로 설정
    train_path = "train.csv"
    test_path = "test.csv"
    sample_sub_path = "sample_submission.csv"
    pdf_folder = "./pdf_guidelines"  # 건설안전지침 PDF들이 저장된 폴더

    # 1. 데이터 로드
    dp = DataProcessor()
    train_df = dp.load_train_data(train_path)
    test_df = dp.load_test_data(test_path)

    # 2. PDF 벡터 DB 구축 (옵션: PDF가 있을 경우)
    pdf_vectorizer = PDFVectorizer(pdf_folder)
    pdf_vectorizer.vectorize_pdfs()

    # 3. RAG용 passages 생성 (train 데이터의 사고발생상황 활용)
    passages_file = "passages.json"
    RAGGenerator.create_passages(train_df, passages_file)
    rag_generator = RAGGenerator(passages_file)

    # 4. LLM 기반 생성 준비 (Few-shot 예시 포함)
    llm_generator = LLMGenerator("t5-base")
    few_shot_examples = llm_generator.get_few_shot_examples(train_df, n_shot=3)

    # 5. 지도학습 기반 모델 준비
    supervised_model = SupervisedModel("t5-base")

    # 6. Ensemble 예측 모듈 준비
    ensemble_predictor = EnsemblePredictor('jhgan/ko-sbert-sts')

    predictions = []
    for idx, row in test_df.iterrows():
        # 사고 상황 및 원인, 환경정보를 하나의 입력 텍스트로 구성
        accident_info = f"온도: {row.get('온도', '')}, 습도: {row.get('습도', '')}, 사고발생상황: {row.get('사고발생상황', '')}, 원인: {row.get('원인', '')}"

        # PDF 도메인 지식 활용: 입력과 관련된 건설안전지침 검색
        pdf_guidelines = pdf_vectorizer.search(accident_info, top_k=1)
        domain_context = pdf_guidelines[0] if pdf_guidelines else ""

        # 최종 입력: 도메인 지식 포함
        final_input = accident_info + " " + domain_context

        # 각 모듈별 예측 생성
        pred_rag = rag_generator.generate(final_input)
        prompt = llm_generator.create_prompt(few_shot_examples, final_input)
        pred_llm = llm_generator.generate(prompt)
        pred_supervised = supervised_model.generate(final_input)

        # Ensemble: 3가지 예측 중 S-BERT 유사도가 가장 높은(상호 일치하는) 예측 선택
        pred_candidates = [pred_rag, pred_llm, pred_supervised]
        final_pred = ensemble_predictor.ensemble(pred_candidates)
        predictions.append(final_pred)
        print(f"Test idx {idx} 예측 완료.")

    # 7. Submission 파일 생성
    submission = pd.read_csv(sample_sub_path)
    submission['대책'] = predictions  # sample_submission에 맞게 '대책' 컬럼 명 확인
    submission.to_csv("final_submission.csv", index=False, encoding="utf-8-sig")
    print("Submission 파일 생성 완료: final_submission.csv")

if __name__ == "__main__":
    main()
