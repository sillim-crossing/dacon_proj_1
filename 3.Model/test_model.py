import os
import re
import json
import ast
import pandas as pd
import numpy as np
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import tqdm
import multiprocessing

MAX_LENGTH = 23000

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
        for col in ['작업프로세스', '사고원인', '재발방지대책 및 향후조치계획']:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_text)
        return df

    def load_test_data(self, filepath):
        df = pd.read_csv(filepath)
        for col in ['작업프로세스', '사고원인']:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_text)
        return df

# ==============================
# 2. CSV를 활용한 PDF 문장 검색 모듈
# ==============================
class PDFCsvRetriever:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        if isinstance(self.df.iloc[0]['PDF 내 관련 문장'], str):
            try:
                sample_val = ast.literal_eval(self.df.iloc[0]['PDF 내 관련 문장'])
                self.df['PDF 내 관련 문장'] = self.df['PDF 내 관련 문장'].apply(lambda x: ast.literal_eval(x))
            except:
                pass

    def search(self, query, top_k=3):
        filtered = self.df[self.df['질문'] == query].copy()
        if len(filtered) == 0:
            return []
        filtered.sort_values(by='유사도', ascending=False, inplace=True)
        top_rows = filtered.head(top_k)
        merged_context = []
        for row in top_rows.itertuples():
            if isinstance(row._5, list):
                for txt_tuple in row._5:
                    if isinstance(txt_tuple, tuple) and len(txt_tuple) > 1:
                        merged_context.append(txt_tuple[1])
                    else:
                        merged_context.append(str(txt_tuple))
            else:
                merged_context.append(str(row._5))
        final_context = " ".join(merged_context)
        return [final_context]

# ==============================
# 3. 한국어 전용 생성 모듈 (KoBART 기반 RAG)
# ==============================
class KoreanRAGGenerator:
    def __init__(self, passages_file='passages.json', embed_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", max_length=MAX_LENGTH):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart")
        self.model = BartForConditionalGeneration.from_pretrained("hyunwoongko/kobart")
        with open(passages_file, "r", encoding="utf-8") as f:
            self.passages = json.load(f)
        self.passage_texts = [p["text"] for p in self.passages]
        self.embedder = SentenceTransformer(embed_model_name)
        self.passage_embeddings = self.embedder.encode(self.passage_texts, convert_to_tensor=True)
        self.max_length = max_length

    def retrieve(self, query, top_k=1):
        query_emb = self.embedder.encode(query, convert_to_tensor=True)
        sims = util.cos_sim(query_emb, self.passage_embeddings)[0]
        top_results = torch.topk(sims, k=top_k)
        indices = top_results.indices.tolist()
        retrieved_texts = [self.passage_texts[i] for i in indices]
        return retrieved_texts

    def generate(self, query, max_length=None, num_beams=5):
        max_length = max_length or self.max_length
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

# ==============================
# 4. 최종 파이프라인 및 Submission 생성
# ==============================
def process_chunk(data_chunk, processor, func):
    return [func(chunk) for chunk in data_chunk]

def save_results_in_chunks(results, filename="generated_results.csv", chunk_size=1000):
    all_results = []
    for idx, result in enumerate(results):
        all_results.append(result)
        if (idx + 1) % chunk_size == 0:
            df = pd.DataFrame(all_results, columns=["Query", "Response"])
            df.to_csv(filename, index=False, encoding="euc-kr", mode='a', header=not bool(idx))
            all_results = []
    if all_results:  # 남은 결과 저장
        df = pd.DataFrame(all_results, columns=["Query", "Response"])
        df.to_csv(filename, index=False, encoding="euc-kr", mode='a', header=not bool(idx))
    print(f"Results saved to {filename}!")

def main():
    # 파일 경로 설정
    train_path = "D:/sillim_crossing_1/dacon_proj_1/1.Data/train.csv"
    #train_path = os.path.join(os.getcwd(), "..", "1.Data", "train.csv")
    #test_path = os.path.join(os.getcwd(), "..", "1.Data", "test.csv")
    
    processor = DataProcessor()
    train_data = processor.load_train_data(train_path)
    
    test_data = train_data.iloc[2400:2500]
    train_data = train_data.iloc[:2400] # 테스트를 위해 각각 일부만 설정
    #test_data = processor.load_test_data(test_path)
    print("데이터 로딩 완료")

    # 전처리된 데이터 처리
    num_workers = min(multiprocessing.cpu_count(), 4)
    print("사용할 코어 갯수:", num_workers)
    chunks = np.array_split(test_data, num_workers)
    pool = multiprocessing.Pool(processes=num_workers)
    results = pool.starmap(process_chunk, [(chunk, processor, processor.clean_text) for chunk in chunks], chunksize=50)
    
    all_results = [pd.DataFrame(chunks[i], columns = results[i]) for i in range(len(chunks))]
    
    print("전처리 완료. 처리된 데이터 수:", len(results))
    
    # RAG 모델을 이용해 답변 생성
    rag_generator = KoreanRAGGenerator()
    rag_results = []

    for result_df in all_results:
        for _, row in tqdm.tqdm(result_df.iterrows(), total=result_df.shape[0]):
            # 각 열의 값을 기반으로 query 작성
            query = (
                f"기온: {row['기온']}, 습도: {row['습도']}, 사고원인: {row['사고원인']}, "
                f"사고 발생 장소: {row['장소']}, 사고 부위: {row['부위']}, 사고 유형: {row['사고객체']}, "
                f"작업프로세스: {row['작업프로세스']}에 대한 사고 예방 대책을 작성해줘."
            )
            
            # RAG 모델을 이용해 답변 생성
            generated_text = rag_generator.generate(query)
            print("query:", query)
            print("generated_text:", generated_text)
            
            # 결과를 rag_results에 저장
            rag_results.append({"Query": query, "Response": generated_text})

    # 최종 결과 저장
    save_results_in_chunks(rag_results)

if __name__ == "__main__":
    main()
