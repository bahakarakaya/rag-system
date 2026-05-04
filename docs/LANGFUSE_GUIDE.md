# Langfuse Rehberi (RAG System)

Bu dokuman, bu projedeki Langfuse entegrasyonunu anlamaniz ve etkin kullanmaniz icin pratik bir rehberdir. Teknik terimler (Trace, Span, Observation, Generation, Latency, Metadata) bilincli olarak korunmustur.

## 1) Baglamsal Yorumlama: Bilesenler ve Langfuse Haritasi

### Cekirdek bilesenler (bu projede)
- Ingestion: Dosya yukleme, parcalama ve vektor/dokuman indeksleme akisidir.
- Retrieval: Query embed edilip vektor + BM25 ile arama yapilir, RRF ve reranker ile siralama tamamlanir.
- Generation: LLM (OpenAI veya Ollama) ile yanit uretilir.

### Langfuse ile birebir esleme
- Trace: Bir kullanici isteginin uctan uca akisi.
  - Query icin: "rag_query" Trace
  - Ingestion icin: "ingestion" Trace
  - Giris noktasi: [main.py](main.py) ve [streamlit_app.py](streamlit_app.py)

- Span: Trace icindeki her ana adim.
  - Ingestion adimlari: [rag/pipeline/ingestion.py](rag/pipeline/ingestion.py)
    - `ingestion_pipeline.run` (tum ingestion)
    - `ingest_document` (dokuman bazli Span)
  - Query adimlari: [rag/pipeline/query.py](rag/pipeline/query.py)
    - `query_pipeline.run`
  - Generation adimi: [rag/pipeline/generation.py](rag/pipeline/generation.py)
    - `generation_pipeline.run`
  - Retrieval adimlari:
    - `hybrid_retriever.retrieve` ([rag/stores/hybrid.py](rag/stores/hybrid.py))
    - `faiss.search` ([rag/stores/faiss.py](rag/stores/faiss.py))
    - `bm25.search` ([rag/stores/bm25.py](rag/stores/bm25.py))
    - `cross_encoder.rerank` ([rag/pipeline/reranker.py](rag/pipeline/reranker.py))
  - Embedding adimi:
    - `embed_chunks` ([rag/ingestion/embedders/sentence_trans.py](rag/ingestion/embedders/sentence_trans.py))

- Observation / Generation:
  - OpenAI cagrisi "generation" olarak loglanir: `openai.chat.completions`
    - Kaynak: [rag/generation/gpt.py](rag/generation/gpt.py)
  - Ollama cagrisi "generation" olarak loglanir: `ollama.generate`
    - Kaynak: [rag/generation/ollama.py](rag/generation/ollama.py)
  - Bu Generation kayitlari prompt, model ve output icin dogrudan izlenebilir olur.

### Prompt ve Context akisi
- Prompt olusturma: [rag/generation/gpt.py](rag/generation/gpt.py)
  - `prompt = prompt.format(context=context, query=query)`
- Context iceriği: `retrieved_chunks` ile olusturulan birleştirilmis metin.
- Bu nedenle Langfuse icinde Trace > Span > Generation hiyerarsisi, pratikte "query -> retrieval -> rerank -> prompt -> LLM" zincirini temsil eder.

## 2) Stratejik Kullanim Senaryolari (Evaluations / Scores + Prompt Management)

1) Retrieval kalitesi icin otomatik Score
   - Query sonucunda gelen chunk sayisini ve son cevabin dayandigi kaynaklari kullanarak basit bir "relevance" skoru yaratabilirsiniz.
   - Ornek: `result_count` dusukken ve cevap bosken negatif skor verin.
   - Neden: Retrieval hatalari LLM kalitesini direkt etkiliyor.

2) Reranker etkisi icin A/B Score
   - `CrossEncoderReranker` acik/kapali A/B senaryosu kurun.
   - Aynı query icin yanitlari karsilastirip kullanıcı memnuniyet skorunu (manuel veya otomatik) kaydedin.
   - Neden: Reranker Latency artirir, kaliteyi olcmeden ROI goremezsiniz.

3) Prompt Management ile versiyonlama
   - Promptu Langfuse prompt management ile versiyonlayin.
   - Ornek: "prompt_v1" vs "prompt_v2" ve hangi versiyonun daha az hallucination urettigini Score ile takip edin.
   - Neden: Prompt degisimi, cevap kalitesinde en hizli etkendir.

4) Maliyet optimizasyonu icin token ve Latency Score
   - OpenAI usage verileri zaten Generation icinde yakalaniyor.
   - "cost_per_answer" veya "latency_budget" Score ekleyip model secimini optimize edebilirsiniz.
   - Neden: `gpt-3.5-turbo` ile `gpt-4o-mini` farkini sayisal olarak gormek kolaylasir.

5) Ingestion basarisini izleyen kalite Score
   - `ingestion` Trace icinde `processed_docs`, `skipped_docs`, `total_chunks` alanlari var.
   - Belirli esiklerin altinda otomatik uyarı skoru verin (or. 0 chunks).
   - Neden: Veri kalitesi dusukse Retrieval deger kaybeder.

## 3) ROI ve Faydalar (bu kod uzerinden orneklerle)

- Latency ayiklama:
  - `hybrid_retriever.retrieve` ile `cross_encoder.rerank` arasindaki sure farkini Span seviyesinde gorursunuz.
  - Ornek: Reranker gecikmesi coksa `top_k` veya `top_k*2` azaltilabilir.

- Maliyet takibi:
  - OpenAI Generation icinde `usage_details` ile token bazli metrikler hazir.
  - Ornek: `prompt` icindeki context uzunlugu artinca maliyet arttigini net gorursunuz.

- Prompt degisimi etkisi:
  - Prompt versiyonlarini Langfuse Prompt Management ile yonetip Score ile karsilastirabilirsiniz.
  - Ornek: Daha kisa prompt, Latency ve maliyeti azaltirken dogrulukta dusus yaratabilir.

- Retrieval darboğazlari:
  - `faiss.search` ve `bm25.search` Span sayisi ve sonuc sayisi farklari, index kalitesi hakkinda sinyal verir.
  - Ornek: BM25 sonuclari cok zayifsa ingestion chunking kurallari yeniden ayarlanabilir.

- Uctan uca gorunurluk:
  - [main.py](main.py) ve [streamlit_app.py](streamlit_app.py) girislerinden baslayan Trace, butun pipeline’i görünur kilıyor.
  - Ornek: Hangi query icin hangi chunklar kullanildi, hangi model ile cevap verildi netlesir.

## Kisa Oneri
- Baslangic icin: Trace ve Generation loglarini izleyin, sonra 1-2 basit Score ile kalite takibini baslatin.
- Prompt Management ile iki prompt versiyonu acip A/B karsilastirma yapin.

Bu rehber, mevcut entegrasyonun ustune dogrudan uygulanabilir ve kisa surede geri donus (ROI) saglar.
