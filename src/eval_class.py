import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import pandas as pd
import sys
from datasets import load_dataset

def configure_python_path():
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Apply the path tweak before any project imports
configure_python_path()

import src.scripts.msmarco_eval_ranking as msmarco_eval_ranking
from config.config import STORAGE_DIR

from src.utils.retrieval_utils import (
    embed_bge,
    embed_jina,
    embed_mamba,
    retrieve_bm25,
    embed_s_transformers,
    rerank_cross_encoder,
    embed_qwen,
    get_eval_metrics,
    create_results_file,
    get_legal_dataset,
    get_legal_queries,
    create_predictions_file,
    embed_chunkwise,
    get_sim_bge,
    chunk_by_paragraphs,
    embed_bge_sparse,
    embed_bge_sliding_window,
    rerank_cross_encoder_chunked
)

from src.models.model_setup import get_bge_m3_model, get_jinja_model, get_mamba_model


class Evaluator:
    """
        Initialize the evaluator with required parameters.

        Parameters
        ----------
        ds : str
            The dataset to use (e.g., 'msmarco', 'legal').
        model_name : str
            The model identifier (e.g., 'bm25', 'bge', 'jinja', etc.).
        metric_names : set
            A set of metric names for evaluation.
        limit : int
            The maximum number of documents to process.
        reuse_run : bool
            Flag to determine if a cached run should be reused.
        model_instance : object
            The model instance for embedding (if applicable).
        rerank : bool
            Whether to perform reranking.
        tokenizer : object, optional
            Tokenizer instance for models that require it.
        reranker_model : object, optional
            Reranker model instance.
    """
    def __init__(self, ds, model_name, metric_names, limit=None, reuse_run=False, model_instance=None, rerank=False, tokenizer=None, reranker_model=None, reranker_model_type=None, max_length=-1, rerank_chunkwise=False, checkpoint=None):
        self.ds = ds
        self.model_name = model_name
        self.metric_names = metric_names
        self.limit = limit
        self.reuse_run = reuse_run
        self.model_instance = model_instance
        self.rerank = rerank
        self.tokenizer = tokenizer
        self.reranker_model = reranker_model
        self.reranker_model_type = reranker_model_type
        self.max_length = max_length
        self.rerank_chunkwise = rerank_chunkwise
        self.checkpoint = checkpoint

        self.device = torch.device("cuda:0")
        self.docs = None
        self.doc_ids = None
        self.queries = None
        self.query_ids = None
        self.qrels_dev_df = None
        self.run = None
        self.path_to_reference_qrels = None

    def load_data(self):
        if self.ds in ("legal-54", "legal-inpars", "legal-synthetic"):
            self.doc_ids, self.docs = get_legal_dataset(os.path.join("..", "data", "corpus.jsonl"))
            self.doc_dict = {doc_id: doc for doc_id, doc in zip(self.doc_ids, self.docs)}
            
            if self.ds == "legal-54":
                queries_path = os.path.join("..", "data", "queries_54.tsv")
                self.path_to_reference_qrels = os.path.join("..", "data", "qrels_54.tsv")
                test_qids_path = None
            elif self.ds == "legal-inpars":
                queries_path = os.path.join("..", "data", "inpars", "queries_inpars.tsv")
                self.path_to_reference_qrels = os.path.join("..", "data", "inpars", "qrels_inpars_mistral-small-2501.tsv")
                test_qids_path = os.path.join("..", "data", "inpars", "qids_inpars_test.txt")
            elif self.ds == "legal-synthetic":
                queries_path = os.path.join("..", "data", "synthetic_llm_dense", "queries_synthetic.tsv")
                self.path_to_reference_qrels = os.path.join("..", "data", "synthetic_llm_dense", "qrels_synthetic_mistral-small-2501.tsv")
                test_qids_path = os.path.join("..", "data", "synthetic_llm_dense", "qids_synthetic_test.txt")

            self.query_ids, self.queries = get_legal_queries(queries_path)

            self.qrels_dev_df = pd.read_csv(
                self.path_to_reference_qrels,
                sep="\t",                # TREC qrels are usually tab-separated
                names=["query_id", "iteration", "doc_id", "relevance"],
                header=None,            # There's no header in qrels files
                dtype={"query_id": str, "iteration": int, "doc_id": str, "relevance": int}
            )

            if test_qids_path is not None:
                with open(test_qids_path, "r") as f:
                    test_qids = f.readlines()
                test_qids = [qid.strip() for qid in test_qids]

                # filter queries and query_ids to only include the ones in the test set
                self.queries = [query for query_id, query in zip(self.query_ids, self.queries) if query_id in test_qids]
                self.query_ids = [query_id for query_id in self.query_ids if query_id in test_qids]

                # filter qrels to only include the ones in the test set
                self.qrels_dev_df = self.qrels_dev_df[self.qrels_dev_df["query_id"].isin(test_qids)]

                # Right after filtering qrels_dev_df:
                filtered_qrels_path = os.path.join(STORAGE_DIR,
                    "legal_ir", "data", "annotations", f"{self.ds}_qrels_test.tsv")
                self.qrels_dev_df.to_csv(
                    filtered_qrels_path,
                    sep="\t",
                    header=False,
                    index=False
                )
                self.path_to_reference_qrels = filtered_qrels_path

            # create a dictionary of query_id to query
            self.query_dict = {query_id: query for query_id, query in zip(self.query_ids, self.queries)}
        else:
            raise ValueError("Dataset not supported.")
        print("Data prepared.")

    def _process_run(self, compute_fn):
        """
        Checks if the run file exists; if not, computes the run using the provided function,
        then pickles the result.

        Parameters
        ----------
        run_path : str
            File path for storing/loading the run.
        compute_fn : callable
            Function that computes the run.
        """
        self.run = compute_fn()
        
    def get_run(self):
        """
        Retrieves or computes the run based on the model name.
        """
        # Mapping of model names to their respective computation lambdas.
        model_mapping = {
            "bm25": lambda: retrieve_bm25(self.docs, self.queries, self.doc_ids, self.query_ids),
            "bge-sliding": lambda: embed_bge_sliding_window(
                get_bge_m3_model('BAAI/bge-m3'),
                self.doc_dict, self.query_dict, self.reuse_run
            ),
            "bge": lambda: embed_bge(
                get_bge_m3_model('BAAI/bge-m3') if self.checkpoint is None else get_bge_m3_model(self.checkpoint),
                self.docs, self.queries, self.doc_ids, self.query_ids, self.reuse_run
            ),
            "bge-sparse": lambda: embed_bge_sparse(
                get_bge_m3_model('BAAI/bge-m3'),
                self.docs, self.queries, self.doc_ids, self.query_ids, self.reuse_run
            ),
            "jina": lambda: embed_jina(
                get_jinja_model().to(self.device),
                self.docs, self.queries, self.doc_ids, self.query_ids
            ),
            "mamba": lambda: embed_mamba(
                *get_mamba_model(), self.docs, self.queries, self.doc_ids, self.query_ids
            ),
            "bge-finetuned": lambda: embed_bge(
                get_bge_m3_model('test_encoder_only_m3_bge-m3_sd'),
                self.docs, self.queries, self.doc_ids, self.query_ids, self.reuse_run
            ),
            "sentence-transformer": lambda: embed_s_transformers(
                self.model_instance, self.docs, self.queries, self.doc_ids, self.query_ids
            ),
            "qwen": lambda: embed_qwen(
                self.model_instance, self.tokenizer, self.docs, self.queries, self.doc_ids, self.query_ids
            ),
            "bge-chunkwise": lambda: embed_chunkwise(get_bge_m3_model('BAAI/bge-m3'), get_sim_bge, self.docs, self.queries, self.doc_ids, self.query_ids, chunk_func=chunk_by_paragraphs, window_size=256)
        }

        if self.model_name not in model_mapping:
            raise ValueError("Model not supported.")

        self._process_run(model_mapping[self.model_name])

    def write_run_to_tsv(self, qid: str, out_path):
        run_Q1 = self.run[qid]
        run_Q1 = dict(sorted(run_Q1.items(), key=lambda x: x[1], reverse=True))
        with open(out_path, "w") as f:
            f.write("doc_id\tscore\n")
            for did, score in run_Q1.items():
                f.write(f"{did}\t{str(score)}\n")

    def rerank_run(self):
        # self.write_run_to_tsv(qid="1", out_path="run_Q1_before.tsv")
        # for each query rerank the top 100 docs
        if self.rerank_chunkwise:
            self.run = rerank_cross_encoder_chunked(self.reranker_model, self.reranker_model_type, self.tokenizer, self.run, 100, self.query_dict, self.doc_dict,
                                                    max_length=self.max_length, stride=self.max_length//2, aggregator="top10")
        else:
            self.run = rerank_cross_encoder(self.reranker_model, self.reranker_model_type, self.tokenizer, self.run, 50, self.query_dict, self.doc_dict,
                                    max_length=self.max_length)
        # self.write_run_to_tsv(qid="1", out_path="run_Q1_after.tsv")

    def get_metrics(self):
        run_qids = set(self.run.keys())
        ref_qids = set(self.qrels_dev_df["query_id"].astype(str).unique())
        print(f"Run queries:   {len(run_qids)}")
        print(f"QREL queries:  {len(ref_qids)}")
        print(f"Intersection:  {len(run_qids & ref_qids)}")
        self.metrics = get_eval_metrics(self.run, self.qrels_dev_df, self.doc_ids, self.metric_names)
        result_paths = create_results_file(self.run)
        prediction_paths = create_predictions_file(self.run)   # create TREC style qrel file (contains same info as results.txt)
        if type(result_paths) == str:
            msmarco_eval_ranking.main(self.path_to_reference_qrels, [result_paths])
        elif type(result_paths) == list:
            msmarco_eval_ranking.main(self.path_to_reference_qrels, result_paths)
        else:
            raise Exception("Invalid result paths")


    def create_run_df(self, run, top_k=5):
        """
        Create a DataFrame with queries as columns and rows representing the top_k retrieved documents.

        For each query in the input dictionary, the documents are sorted in descending order
        according to their scores. Only the top_k document identifiers are kept. In the returned
        DataFrame, each column corresponds to a query and each row represents the rank (e.g., Rank 1,
        Rank 2, ..., Rank top_k). If a query has fewer than top_k documents, the missing entries are
        set to None.

        Parameters
        ----------
        run : dict
            A dictionary where each key is a query identifier and each value is a dictionary of document
            identifiers mapped to their corresponding scores. Example format:
            {
                "query1": {"doc1": score1, "doc2": score2, ...},
                "query2": {"doc3": score3, "doc4": score4, ...},
                ...
            }
        top_k : int, optional
            The number of top documents to retrieve for each query (default is 5).

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame with queries as columns and rows representing the top_k document ranks.
            The index labels (e.g., 'Rank 1', 'Rank 2', etc.) indicate the rank order.

        Raises
        ------
        ValueError
            If the input run is not a dictionary, or if its values are not dictionaries.
        """
        if not isinstance(run, dict):
            raise ValueError("The run input must be a dictionary.")
        
        data = {}
        for query_id, doc_scores in run.items():
            if not isinstance(doc_scores, dict):
                raise ValueError("Each value in the run dictionary must be a dictionary of document scores.")
            # Sort documents for the given query based on their scores (higher first)
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            top_docs = [doc for doc, score in sorted_docs[:top_k]]
            # Pad with None if fewer than top_k documents exist
            if len(top_docs) < top_k:
                top_docs.extend([None] * (top_k - len(top_docs)))
            data[query_id] = top_docs

        # Create DataFrame and label the rows by rank.
        df = pd.DataFrame(data)
        df.index = [f"Rank {i + 1}" for i in range(top_k)]
        return df

    def evaluate(self):
        self.load_data()
        self.get_run()
        if self.rerank:
            self.rerank_run()
        self.get_metrics()


def get_messirve_corpus(country):
    ds = load_dataset("spanish-ir/messirve", country)
    docs = ds["test"]["docid_text"]
    queries = ds["test"]["query"]
    doc_ids = ds["test"]["docid"]
    query_ids = ds["test"]["id"]
    return docs, queries, doc_ids, query_ids


if __name__ == "__main__":
    # Evaluate IR metrics.
    evaluator = Evaluator(
        ds="legal-54",
        model_name="bge",
        metric_names={'ndcg', 'ndcg_cut.10', 'recall_1000', 'recall_100', 'recall_10', 'recip_rank', 'map'},
        rerank=False,
        # max_length=512,
        # rerank_chunkwise=False,
        # reranker_model_type="sbert"
    )
    evaluator.evaluate()
