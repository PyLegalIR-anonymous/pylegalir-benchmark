# PyLegalIR: A Benchmark for Spanish Legal Information Retrieval from Paraguayan Supreme Court Cases
PyLegalIR is the first benchmark dataset for legal information retrieval in Spanish, built from expert annotations of real-world judicial decisions issued by the Criminal Chamber of the Supreme Court of Paraguay.

This repository contains the dataset, synthetic silver labels, and code to evaluate retrieval models on PyLegalIR.

## 📂 Repository Structure
```
data/
│
├── corpus.jsonl                   # Full corpus of 5,000 legal rulings in JSONL format
├── queries_54.tsv                 # Original 54 expert-written queries
├── qrels_54.tsv                   # Human-annotated relevance labels
│
├── inpars/                        # Silver data (InPars-style)
│   ├── queries_inpars.tsv
│   ├── qrels_inpars_mistral-small-2501.tsv
│   ├── qids_inpars_{train,dev,test}.txt
│
└── synthetic_llm_dense/          # Silver data (LLM-labeled dense annotations)
    ├── queries_synthetic.tsv
    ├── qrels_synthetic_mistral-small-2501.tsv
    ├── qids_synthetic_{train,dev,test}.txt

src/
│
├── models/                        # For loading models
├── scripts/                       # Scripts for MRR evaluation
├── utils/                         # Utility functions for retrieval, scoring
│   [...]
│
├── eval_class.py                  # Main evaluation class

.gitignore
```

## 📝 Dataset Overview

- Documents: 5,000 Supreme Court rulings (2011–2023)
- Language: Spanish
- Queries: 54 expert-written queries
- Annotations: ~30 documents per query (1,597 query-doc pairs), 4-level graded relevance
- Synthetic Labels:
    - InPars-style queries per document
    - LLM-annotated queries + relevance/evidence spans

## ⚙️ Getting Started
### 1. Clone and create a virtual environment
```
git clone https://github.com/PyLegalIR-anonymous/pylegalir-benchmark.git
cd pylegalir-benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run evaluation
You can evaluate a retrieval model by instantiating the `Evaluator` class from `eval_class.py`.

### Example usage:
```
cd src/
python3 eval_class.py
```

This default run will:

- Use the "legal-54" dataset (expert-annotated queries)
- Evaluate the "bm25" retriever
- Output metrics like nDCG, Recall@100, MRR, and more

### Customizing the Evaluation
To run other models, pass one of the following model_name values when initializing the Evaluator:
| Model Name             | Description                               |
| ---------------------- | ----------------------------------------- |
| `bm25`                 | Classic BM25 sparse retrieval             |
| `bge`                  | BGE-m3 dense retriever                    |
| `bge-sliding`          | BGE with sliding window                   |
| `bge-sparse`           | See **                                    |
| `jina`                 | jina-embeddings-v3                        |
| `sentence-transformer` | Any sbert model instance                  |

** Will run evaluation for BGE-Sparse, BGE-ColBERT, and all their combination as they were reported in the paper

You can also:

- Set rerank=True, max_length (only for rerankers) and reranker_model_type to "bge" for BAAI/bge-reranker-v2-gemma or "sbert" for cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 to apply a cross-encoder reranker
- Set rerank_chunkwise=True for reranking by chunks
- Switch datasets via ds argument: "legal-54", "legal-inpars", "legal-synthetic"
- Provide a custom model_instance, tokenizer, or checkpoint

Note: The dual encoder retrievers pull max_lengths from config/config.py, but for the cross-encoder rerankers the max_length param has to be specified.

### Evaluator instanciation:
```
evaluator = Evaluator(
        ds="legal-54",
        model_name="bm25",
        metric_names={'ndcg', 'ndcg_cut.10', 'recall_1000', 'recall_100', 'recall_10', 'recip_rank', 'map'},
        rerank=False,
        # max_length=128,
        # rerank_chunkwise=True,
        # reranker_model_type="sbert"
    )
    evaluator.evaluate()
```

## 📊 Evaluation
Models are evaluated using:

- nDCG@10
- Recall@100
- MRR@10

Evaluation is based on the queries_54.tsv and qrels_54.tsv files. These should not be used for training. Use silver labels from inpars/ and synthetic_llm_dense/ instead.

## License

The **code** (`src/`) in this repository is released under the [MIT License](https://opensource.org/licenses/MIT).  
You are free to use, modify, and distribute it with proper attribution.

The **dataset** (`data/`) is released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).  
This means you may share and adapt the data for any purpose, including commercial use, as long as you give appropriate credit.

Please cite our paper if you use this benchmark in your work.


## 📜 Citation

If you use PyLegalIR in your research, please cite:

[Citation coming soon after paper acceptance.]
