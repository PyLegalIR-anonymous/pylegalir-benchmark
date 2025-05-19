import os
import numpy as np
from tqdm import tqdm
import torch
import pytrec_eval
from .train_utils import tokenize_with_manual_eos, get_eos_embeddings
from config.config import MAX_QUERY_LEN, MAX_DOC_LEN
import torch.nn.functional as F
try:
    import faiss
except ImportError:
    pass
import pandas as pd
import pickle
from src.utils.cross_encoder_scorer import CrossEncoderScorer


def build_faiss_index(embeddings, use_cosine=False):
    """
    Build a FAISS index from document embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Array of document embeddings with shape (num_docs, emb_dim).
    use_cosine : bool, optional
        If True, normalize embeddings for cosine similarity (default is False).

    Returns
    -------
    index : faiss.Index
        A FAISS index built from the provided embeddings.
    """
    if use_cosine:
        faiss.normalize_L2(embeddings)
    emb_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(emb_dim)  # Using inner product for dot-product similarity.
    index.add(embeddings)
    return index


def search_faiss_index(index, query_embeddings, top_k, use_cosine=False):
    """
    Search the FAISS index to retrieve the top-k nearest neighbors for each query.

    Parameters
    ----------
    index : faiss.Index
        The FAISS index built from document embeddings.
    query_embeddings : np.ndarray
        Array of query embeddings with shape (num_queries, emb_dim).
    top_k : int
        Number of nearest neighbors to retrieve per query.
    use_cosine : bool, optional
        If True, normalize query embeddings for cosine similarity (default is False).

    Returns
    -------
    distances : np.ndarray
        Array of shape (num_queries, top_k) containing similarity scores.
    indices : np.ndarray
        Array of shape (num_queries, top_k) containing indices of the nearest documents.
    """
    if use_cosine:
        faiss.normalize_L2(query_embeddings)
    distances, indices = index.search(query_embeddings, top_k)
    return distances, indices


def batch_encode_jina(model, data, batch_size, task, max_length):
    """
    Encode data in batches using the provided Jina model.
    Returns a list of all embeddings (on CPU).
    """
    from tqdm import trange

    embeddings = []
    for start_idx in trange(0, len(data), batch_size, desc="Batch encoding data"):
        batch = data[start_idx : start_idx + batch_size]
        batch_embeddings = model.encode(batch, task=task, max_length=max_length)
        batch_embeddings = np.array(batch_embeddings, dtype=np.float32)
        embeddings.extend(batch_embeddings)
    return embeddings


def retrieve(query_ids, doc_ids, similarity):
    run = {}
    for i in tqdm(range(len(query_ids)), desc="Creating run"):
        query_sim = similarity[i]
        
        run[str(query_ids[i])] = {str(doc_ids[j]): float(query_sim[j]) for j in range(len(doc_ids))}
    print("Run created.")
    return run

def compute_similarity(embeddings_queries, embeddings_docs, sim_type='dot'):
    """
    Given query and document embeddings, compute the similarity between queries and documents
    and return the similarity matrix.
    Dot-product and cosine similarity are supported.

    Parameters
    ----------
    embeddings_queries : torch.Tensor
        Tensor of query embeddings. Can be either 2D [num_queries, emb_dim] or 1D [emb_dim].
    embeddings_docs : torch.Tensor
        Tensor of document embeddings of shape [num_docs, emb_dim].
    sim_type : str, optional
        Type of similarity to use. Options: 'dot', 'cosine'. (default is 'dot').

    Returns
    -------
    torch.Tensor
        Similarity matrix of shape [num_queries, num_docs].
    """
    print("Computing similarity...", end="")

    # If there is just one query, unsqueeze to make it 2D
    if embeddings_queries.dim() == 1:
        embeddings_queries = embeddings_queries.unsqueeze(0)

    if sim_type == 'dot':
        similarities = []
        batch_size = 32
        for query_batch in torch.split(embeddings_queries, batch_size, dim=0):
            sim_batch = query_batch @ embeddings_docs.T
            similarities.append(sim_batch)
        similarity = torch.cat(similarities, dim=0)
    elif sim_type == 'cosine':
        # Normalize embeddings along the embedding dimension
        embeddings_queries = F.normalize(embeddings_queries, p=2, dim=1)
        embeddings_docs = F.normalize(embeddings_docs, p=2, dim=1)
        similarity = (embeddings_queries @ embeddings_docs.T) * 100
    else:
        raise ValueError(f"Invalid similarity type: {sim_type}")
    
    print("Done.")
    return similarity



def compute_similarity_streaming(doc_embeddings_iterator, embeddings_queries,
                                 top_k=10, sim_type='dot'):
    """
    Compute similarity between queries and documents in a streaming fashion without
    loading all document embeddings into memory. Returns the top_k document IDs and
    their similarity scores for each query.

    Parameters
    ----------
    doc_embeddings_iterator : iterator
        Iterator that yields tuples of (doc_embeddings, doc_ids) for each batch.
    embeddings_queries : torch.Tensor
        Tensor of query embeddings with shape (num_queries, embedding_dim).
    top_k : int, optional
        Number of top similar documents to retrieve for each query (default is 10).
    sim_type : str, optional
        Similarity type to use: 'dot' for dot product or 'cosine' for cosine similarity
        (default is 'dot').

    Returns
    -------
    tuple
        A tuple (top_doc_ids, top_scores) where:
            top_doc_ids : list of lists
                Each sublist contains the top_k document IDs for the corresponding query.
            top_scores : torch.Tensor
                Tensor of shape (num_queries, top_k) with the similarity scores.
    """
    num_queries = embeddings_queries.size(0)
    device = embeddings_queries.device

    # Initialize top scores and corresponding doc IDs for each query
    top_scores = torch.full((num_queries, top_k), float('-inf'), device=device)
    top_doc_ids = [[None] * top_k for _ in range(num_queries)]

    if sim_type == 'cosine':
        embeddings_queries = F.normalize(embeddings_queries, p=2, dim=1)

    for doc_batch_embeddings, doc_batch_ids in doc_embeddings_iterator:
        doc_batch_embeddings = doc_batch_embeddings.to(device)
        if sim_type == 'cosine':
            doc_batch_embeddings = F.normalize(doc_batch_embeddings, p=2, dim=1)

        # Compute similarity between all queries and the current doc batch.
        sim_batch = embeddings_queries @ doc_batch_embeddings.T  # (num_queries, batch_size)

        # Concatenate current top_scores with new batch scores along dimension=1.
        combined_scores = torch.cat([top_scores, sim_batch], dim=1)
        new_topk_scores, indices = combined_scores.topk(k=top_k, dim=1)

        # Update top_doc_ids based on the new indices.
        new_top_doc_ids = []
        for i in range(num_queries):
            # Build a combined list of previous doc IDs and current batch doc IDs.
            combined_ids = top_doc_ids[i] + doc_batch_ids
            # Select the doc IDs corresponding to the topk indices.
            new_ids = [combined_ids[idx] for idx in indices[i].tolist()]
            new_top_doc_ids.append(new_ids)
        top_scores = new_topk_scores
        top_doc_ids = new_top_doc_ids

    return top_doc_ids, top_scores


def embed_jina_faiss(
    model,
    docs,
    queries,
    doc_ids,
    query_ids,
    top_k=1000,
    sim_type="dot"
):
    """
    Embed queries and documents using the Jina embeddings model, build a FAISS index
    for document embeddings on CPU in a streaming fashion, and retrieve the top-k most
    similar documents for each query.

    Parameters
    ----------
    model : object
        The Jina embeddings model.
    docs : list
        List of document texts.
    queries : list
        List of query texts.
    doc_ids : list
        List of document IDs corresponding to docs.
    query_ids : list
        List of query IDs.
    top_k : int, optional
        Number of top similar documents to retrieve for each query (default is 1000).
    sim_type : str, optional
        Similarity type: 'dot' for dot product or 'cosine' for cosine similarity
        (default is 'dot').

    Returns
    -------
    dict
        Run dictionary mapping each query ID to a dictionary of document IDs and similarity scores.
    """
    # 1) Encode Queries in Memory
    print("Encoding queries...")
    embeddings_queries = batch_encode_jina(
        model, queries, batch_size=128, task="retrieval.query", max_length=MAX_QUERY_LEN
    )
    # Convert to CPU tensor (float32)
    embeddings_queries = torch.tensor(np.array(embeddings_queries), dtype=torch.float32, device="cpu")
    print("Done encoding queries.")

    # 2) Prepare for streaming document embeddings
    print("Initializing FAISS CPU index for doc embeddings...")
    emb_dim = None
    cpu_index = None  # Will initialize once we know the embedding dimension
    all_doc_ids = []

    # 3) Stream Document Embeddings and Add to the CPU Index
    doc_iter = batch_encode_jina_stream(
        model, data=docs, ids=doc_ids, batch_size=128,
        task="retrieval.passage", max_length=MAX_DOC_LEN
    )

    first_batch = True
    for batch_embeddings, batch_ids in doc_iter:
        # batch_embeddings is already a float32 torch.Tensor on CPU
        batch_embeddings_np = batch_embeddings.numpy()

        # If using cosine similarity, L2-normalize before indexing
        if sim_type == "cosine":
            faiss.normalize_L2(batch_embeddings_np)

        if first_batch:
            # Figure out embedding dimension from first batch
            emb_dim = batch_embeddings_np.shape[1]
            # Create a CPU index for IP
            cpu_index = faiss.IndexFlatIP(emb_dim)
            first_batch = False

        # Add to the CPU index
        cpu_index.add(batch_embeddings_np)

        # Keep track of doc IDs in the order they were added
        all_doc_ids.extend(batch_ids)

    print("Done building CPU index.")

    # 4) Convert queries to numpy for FAISS search
    query_embeddings_np = embeddings_queries.numpy()
    if sim_type == "cosine":
        faiss.normalize_L2(query_embeddings_np)

    print(f"Searching FAISS index for top {top_k} docs per query...")
    distances, faiss_indices = cpu_index.search(query_embeddings_np, top_k)
    print("FAISS search complete.")

    # 5) Build Run Dictionary
    run = {}
    for i, qid in enumerate(query_ids):
        run[str(qid)] = {}
        for rank in range(top_k):
            doc_idx = faiss_indices[i, rank]
            if doc_idx < 0 or doc_idx >= len(all_doc_ids):
                continue
            doc_id = all_doc_ids[doc_idx]
            score = distances[i, rank]
            run[str(qid)][str(doc_id)] = float(score)

    return run


def embed_jina(model, docs, queries, doc_ids, query_ids):
    """
    Embed the queries and documents using the Jinja embeddings model and compute the similarity between queries and documents.
    Calls the retrieve function.

    Args:
        model: Jinja embeddings model.
        docs (dict): Dictionary with document_id as key and text as value.
        queries (list): List of queries.
        top_k (int): Number of most similar documents to retrieve.
    
    Returns:
        dict: Dictionary with query as key and a list of tuples of (similarity, document text, doc_id) as value.
    """
    # When calling the `encode` function, you can choose a `task` based on the use case:
    # 'retrieval.query', 'retrieval.passage', 'separation', 'classification', 'text-matching'
    # Alternatively, you can choose not to pass a `task`, and no specific LoRA adapter will be used.
    print("Encoding queries...", end="")
    embeddings_queries = batch_encode_jina(
        model, queries, batch_size=128, task="retrieval.query", max_length=MAX_QUERY_LEN
    )
    print("Done.")

    print("Encoding docs...", end="")
    embeddings_docs = batch_encode_jina(
        model, docs, batch_size=128, task="retrieval.passage", max_length=MAX_DOC_LEN
    )
    print("Done.")

    embeddings_queries = torch.tensor(np.array(embeddings_queries), dtype=torch.float32)
    embeddings_docs = torch.tensor(np.array(embeddings_docs), dtype=torch.float32)

    # Compute similarities
    similarity = compute_similarity(embeddings_queries, embeddings_docs)
    run = retrieve(query_ids, doc_ids, similarity)
    return run


def batch_encode_jina_stream(model, data, ids, batch_size, task, max_length):
    """
    Generator that encodes data in batches using the provided Jina model,
    yielding (embeddings, ids) on each iteration.

    Parameters
    ----------
    model : object
        The Jina embeddings model.
    data : list
        List of texts to be encoded.
    ids : list
        List of IDs, same length as data.
    batch_size : int
        Number of items to encode per batch.
    task : str
        Task identifier for the encoding (e.g., 'retrieval.passage').
    max_length : int
        Maximum token length for the encoding.

    Yields
    ------
    (torch.Tensor, list)
        A tuple with (embeddings for the batch, IDs for that batch).
    """
    from tqdm import trange

    n = len(data)
    for start_idx in trange(0, n, batch_size, desc="Batch streaming data"):
        end_idx = min(start_idx + batch_size, n)
        batch = data[start_idx:end_idx]
        batch_embeddings = model.encode(batch, task=task, max_length=max_length)
        # Convert to CPU float32 tensor
        batch_embeddings = torch.tensor(np.array(batch_embeddings), dtype=torch.float32, device="cpu")

        batch_ids = ids[start_idx:end_idx]
        yield batch_embeddings, batch_ids


def embed_bge(model, docs, queries, doc_ids, query_ids, reuse_run):
    """
    Embed the queries and documents using the BAAI embeddings models and compute the similarity between queries and documents.
    Calls the retrieve function.

    Args:
        model: BAAI embeddings model.
        docs (dict): Dictionary with document_id as key and text as value.
        queries (list): List of queries.
        top_k (int): Number of most similar documents to retrieve.

    Returns:
        dict: Dictionary with query as key and a list of tuples of (similarity, document text, doc_id) as value.
    """
    embeddings_queries = model.encode(queries, batch_size=8, max_length=MAX_QUERY_LEN)['dense_vecs']
    # Embed entire corpus if file does not exist
    path = 'corpus/embeddings_train_corpus_bge-m3.npy'
    if not os.path.exists(path) or not reuse_run:
        print("Embedding docs...", end="")
        embeddings_docs = model.encode(docs, batch_size=8, max_length=MAX_DOC_LEN)['dense_vecs']
        print("Done.")
        # save embeddings
        # print("Saving embeddings...", end="")
        # np.save(path, embeddings_docs)
        # print("Done.")
    else:
        # Load embeddings
        embeddings_docs = np.load(path)

    # Compute similarities
    similarity = compute_similarity(torch.tensor(embeddings_queries, dtype=torch.float32), torch.tensor(embeddings_docs, dtype=torch.float32))
    run = retrieve(query_ids, doc_ids, similarity)
    return run


from transformers import AutoTokenizer
def embed_bge_sliding_window(model, doc_dict, query_dict, reuse_run):
    """
    Embed queries and chunked documents with BGE-M3 dense head,
    apply sliding-window chunking, and compute max-pooled similarity per doc.

    Parameters
    ----------
    model : BGEM3FlagModel
        BGE-M3 model instance.
    docs : dict
        Mapping from document_id to full document text.
    queries : list of str
        Query strings.
    doc_ids : list
        Ordered list of document identifiers (keys of `docs`).
    query_ids : list
        Ordered list of query identifiers.
    reuse_run : bool
        Unused here.

    Returns
    -------
    dict
        Mapping from run name to retrieval result dict.
    """
    # sliding window parameters
    chunk_size = 512
    stride     = 256

    doc_ids = list(doc_dict.keys())
    query_ids = list(query_dict.keys())
    query_texts = list(query_dict.values())

    # instantiate a fast tokenizer (no special tokens, so chunks align simply)
    tokenizer = AutoTokenizer.from_pretrained(
        "BAAI/bge-m3",
        use_fast=True,
    )

    # 1) build a flat list of (doc_id, chunk_text)
    chunk_map = []
    for doc_id in tqdm(doc_ids):
        text = doc_dict[doc_id]
        toks = tokenizer.encode(text, add_special_tokens=False)
        for start in range(0, len(toks), stride):
            window = toks[start : start + chunk_size]
            if not window:
                break
            chunk_map.append((doc_id, tokenizer.decode(window, skip_special_tokens=True)))

    # unzip
    chunk_doc_ids = [did for did, _ in chunk_map]
    chunk_texts  = [ct  for _,  ct in chunk_map]

    # 2) encode queries and chunks (dense head only)
    print("Encoding queries...", end="")
    q_emb = model.encode(
        query_texts,
        batch_size=8,
        max_length=MAX_QUERY_LEN
    )["dense_vecs"]
    print("Done.")

    print("Encoding chunks...", end="")
    c_emb = model.encode(
        chunk_texts,
        batch_size=8,
        max_length=chunk_size
    )["dense_vecs"]
    print("Done.")

    # to tensors
    q_t = torch.tensor(q_emb, dtype=torch.float32)       # (Q, D)
    c_t = torch.tensor(c_emb, dtype=torch.float32)       # (C, D)

    # 3) raw sim: (Q x C)
    raw_sim = q_t @ c_t.T

    print("Computing max-pool chunk sims...", end="")
    # 4) max-pool chunk sims up to doc level
    Q, D = raw_sim.size(0), len(doc_dict)
    sim = torch.full((Q, D), -np.inf)
    pools = {}
    for idx, did in enumerate(tqdm(chunk_doc_ids)):
        pools.setdefault(did, []).append(idx)

    pooling_strategy = "top_3"

    for j, did in enumerate(tqdm(doc_ids)):
        idxs = pools.get(did, [])
        if idxs:
            if pooling_strategy == "max":
                sim[:, j], _ = raw_sim[:, idxs].max(dim=1)
            elif pooling_strategy == "top_3":
                vals, _ = raw_sim[:, idxs].topk(3, dim=1)
                sim[:, j] = vals.mean(dim=1)
    print("Done.")

    # 5) retrieve using your existing function
    run = retrieve(list(query_dict.keys()), list(doc_dict.keys()), sim)
    return run


def compute_sparse_similarity(model, q_lex_list, d_lex_list):
    """
    Compute pairwise sparse similarity matrix using model.compute_lexical_matching_score.

    Parameters
    ----------
    model : BGEM3FlagModel
        BGE-M3 model instance.
    q_lex_list : list of dict
        Query lexical_weights.
    d_lex_list : list of dict
        Document lexical_weights.

    Returns
    -------
    torch.Tensor
        Similarity matrix of shape (num_queries, num_docs).
    """
    num_q, num_d = len(q_lex_list), len(d_lex_list)
    sim = torch.zeros(num_q, num_d, dtype=torch.float32)
    print("Computing sparse similarity...", end="")
    for i, qw in tqdm(enumerate(q_lex_list)):
        for j, dw in enumerate(d_lex_list):
            sim[i, j] = float(model.compute_lexical_matching_score(qw, dw))
    print("Done.")
    return sim


def compute_colbert_similarity(q_col_list, d_col_list, model):
    """
    Compute pairwise ColBERT similarity matrix using the model's colbert_score.

    Parameters
    ----------
    q_col_list : list of array-like, shape (Lq, 128)
        Query token vectors.
    d_col_list : list of array-like, shape (Ld, 128)
        Document token vectors.
    model : BGEM3FlagModel
        Model with `colbert_score` method.

    Returns
    -------
    torch.Tensor
        Similarity matrix of shape (num_queries, num_docs).
    """
    print("Computing ColBERT similarity...", end="")
    num_q = len(q_col_list)
    num_d = len(d_col_list)
    sim = torch.zeros(num_q, num_d, dtype=torch.float32)
    for i in tqdm(range(num_q)):
        for j in range(num_d):
            sim[i, j] = float(model.colbert_score(q_col_list[i], d_col_list[j]))
    print("Done.")
    return sim


def per_query_zscore(sim: torch.Tensor) -> torch.Tensor:
    mu = sim.mean(dim=1, keepdim=True)
    sigma = sim.std(dim=1, unbiased=False, keepdim=True) + 1e-9
    return (sim - mu) / sigma


def per_query_minmax(sim: torch.Tensor) -> torch.Tensor:
    minv = sim.min(dim=1, keepdim=True).values
    maxv = sim.max(dim=1, keepdim=True).values
    return (sim - minv) / (maxv - minv + 1e-9)


def rrf(sim: torch.Tensor, k: int = 60) -> torch.Tensor:
    # Higher score = better → get ranks (1‑based)
    ranks = sim.argsort(dim=1, descending=True).argsort(dim=1) + 1
    return 1.0 / (k + ranks.float())


def embed_bge_sparse(model, docs, queries, doc_ids, query_ids, reuse_run):
    """
    Embed queries and docs with BGE-M3 heads and compute zero-shot runs.

    Parameters
    ----------
    model : BGEM3FlagModel
        BGE-M3 model instance.
    docs : dict of {hashable: str}
        Mapping from document IDs to document texts.
    queries : list of str
        Query strings.
    doc_ids : list
        Ordered list of document identifiers.
    query_ids : list
        Ordered list of query identifiers.
    reuse_run : bool
        If True, attempt to reuse existing runs (not used).

    Returns
    -------
    dict of str -> dict
        Mapping from run name to retrieval results
        (query -> list of (score, text, doc_id)).
    """
    # 1) Encode queries and docs
    print("Embedding queries...", end="")
    q_out = model.encode(
        queries,
        batch_size=16,
        max_length=MAX_QUERY_LEN,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    print("Done.")
    print("Embedding docs...", end="")
    d_out = model.encode(
        docs,
        batch_size=16,
        max_length=MAX_DOC_LEN,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    print("Done.")

    # 2) Compute individual similarity matrices
    dense_score = compute_similarity(
        torch.tensor(q_out["dense_vecs"], dtype=torch.float32),
        torch.tensor(d_out["dense_vecs"], dtype=torch.float32),
    )
    sparse_sim = compute_sparse_similarity(
        model,
        q_out["lexical_weights"],
        d_out["lexical_weights"],
    )
    colbert_sim = compute_colbert_similarity(
        q_out["colbert_vecs"],
        d_out["colbert_vecs"],
        model,
    )

    NORM = "z"

    # after you've built dense_score, sparse_sim, colbert_sim …
    if NORM == "z":
        dense_n  = per_query_zscore(dense_score)
        sparse_n = per_query_zscore(sparse_sim)
        colbert_n = per_query_zscore(colbert_sim)
    elif NORM == "minmax":
        dense_n  = per_query_minmax(dense_score)
        sparse_n = per_query_minmax(sparse_sim)
        colbert_n = per_query_minmax(colbert_sim)
    elif NORM == "rrf":
        dense_n  = rrf(dense_score)
        sparse_n = rrf(sparse_sim)
        colbert_n = rrf(colbert_sim)
    else:  # raw
        dense_n, sparse_n, colbert_n = dense_score, sparse_sim, colbert_sim

    # then fuse the *normalized* matrices
    dense_sparse_sim  = 0.5 * dense_n + 0.5 * sparse_n
    dense_colbert_sim = 0.5 * dense_n + 0.5 * colbert_n
    sparse_colbert_sim = 0.5 * sparse_n + 0.5 * colbert_n
    all_three_sim = (dense_n + sparse_n + colbert_n) / 3.0

    # 4) Retrieve runs for each similarity
    runs = {
        "dense": retrieve(query_ids, doc_ids, dense_score),
        "sparse": retrieve(query_ids, doc_ids, sparse_sim),
        "colbert": retrieve(query_ids, doc_ids, colbert_sim),
        "dense_sparse": retrieve(query_ids, doc_ids, dense_sparse_sim),
        "dense_colbert": retrieve(query_ids, doc_ids, dense_colbert_sim),
        "sparse_colbert": retrieve(query_ids, doc_ids, sparse_colbert_sim),
        "all_three": retrieve(query_ids, doc_ids, all_three_sim),
    }
    return runs

def get_sim_bge(model, docs, queries):
    embeddings_queries = model.encode(queries, batch_size=8, max_length=MAX_QUERY_LEN)['dense_vecs']
    print("Embedding docs...", end="")
    embeddings_docs = model.encode(docs, batch_size=8, max_length=MAX_DOC_LEN)['dense_vecs']
    print("Done.")
    
    # Compute similarities
    similarity = compute_similarity(torch.tensor(embeddings_queries, dtype=torch.float32), torch.tensor(embeddings_docs, dtype=torch.float32))
    return similarity


def embed_qwen(model, tokenizer, docs, queries, doc_ids, query_ids):
    """
    Embed the queries and documents using the BAAI embeddings models and compute the similarity between queries and documents.
    Calls the retrieve function.

    Args:
        model: BAAI embeddings model.
        docs (dict): Dictionary with document_id as key and text as value.
        queries (list): List of queries.
        top_k (int): Number of most similar documents to retrieve.

    Returns:
        dict: Dictionary with query as key and a list of tuples of (similarity, document text, doc_id) as value.
    """
    from torch.utils.data import DataLoader, TensorDataset
    batch_size = 8
    device = next(model.parameters()).device  # Automatically detect model's device
    print("Device: ", device)

    inputs_docs = tokenize_with_manual_eos(tokenizer, docs, max_length=MAX_DOC_LEN)
    inputs_queries = tokenize_with_manual_eos(tokenizer, queries, max_length=MAX_QUERY_LEN)

    print("Docs and queries tokenized.")

    # Create DataLoaders for batching
    doc_input_ids = torch.tensor(inputs_docs["input_ids"], dtype=torch.long)
    doc_attention_mask = torch.tensor(inputs_docs["attention_mask"], dtype=torch.long)
    doc_dataset = TensorDataset(doc_input_ids, doc_attention_mask)
    
    query_input_ids = torch.tensor(inputs_queries["input_ids"], dtype=torch.long)
    query_attention_mask = torch.tensor(inputs_queries["attention_mask"], dtype=torch.long)
    query_dataset = TensorDataset(query_input_ids, query_attention_mask)

    doc_loader = DataLoader(doc_dataset, batch_size=batch_size, pin_memory=True)
    query_loader = DataLoader(query_dataset, batch_size=batch_size, pin_memory=True)

    print("Embedding docs and queries...", end="")
    embeddings_queries = []
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(query_loader, desc="Embedding queries"):
            query_embeds = get_eos_embeddings(model, input_ids.to(device), attention_mask.to(device), tokenizer)
            embeddings_queries.append(query_embeds)
    embeddings_queries = torch.cat(embeddings_queries, dim=0)  # Combine batches
    print("Done.")

    embeddings_docs = []
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(doc_loader, desc="Embedding docs"):
            doc_embeds = get_eos_embeddings(model, input_ids.to(device), attention_mask.to(device), tokenizer)
            embeddings_docs.append(doc_embeds)
    embeddings_docs = torch.cat(embeddings_docs, dim=0)  # Combine batches

    similarity = compute_similarity(embeddings_queries, embeddings_docs)
    run = retrieve(query_ids, doc_ids, similarity)
    return run


def embed_qwen_faiss(
    model,
    tokenizer,
    docs,
    doc_ids,
    queries,
    query_ids,
    top_k=1000,
    sim_type="dot",
    max_doc_len=150,
    max_query_len=20,
    batch_size=128
):
    """
    Embed queries and docs using Qwen, build a CPU FAISS index for doc embeddings,
    then retrieve the top_k docs per query.

    Parameters
    ----------
    model : torch.nn.Module
        The Qwen model in inference mode.
    tokenizer : PreTrainedTokenizer
        Tokenizer for Qwen.
    docs : list of str
        List of document texts.
    doc_ids : list
        List of doc IDs (parallel to docs).
    queries : list of str
        List of query texts.
    query_ids : list
        List of query IDs.
    top_k : int
        Number of docs to retrieve for each query.
    sim_type : str
        'dot' or 'cosine' similarity.
    max_doc_len : int
        Max token length for documents (including appended EOS).
    max_query_len : int
        Max token length for queries (including appended EOS).
    batch_size : int
        Batch size for tokenization + embedding.

    Returns
    -------
    run : dict
        TREC-style dictionary: run[qid][doc_id] = similarity_score.
    """
    # 1) Embed Queries in Memory
    print("Embedding queries in memory...")
    all_query_embeddings = []
    all_query_ids = []

    # We can re-use the same streaming logic for queries, or just do them in one pass if memory allows.
    query_generator = batch_embed_qwen_stream(
        model, tokenizer, queries, query_ids,
        batch_size=batch_size, max_length=max_query_len
    )

    for emb, qids in query_generator:
        all_query_embeddings.append(emb)
        all_query_ids.extend(qids)

    # Combine into a single tensor
    query_embeddings = torch.cat(all_query_embeddings, dim=0)  # (num_queries, hidden_dim)

    print("Queries embedded. Building FAISS index for docs...")

    # 2) Initialize the index after we see the first doc-embedding batch dimension
    cpu_index = None
    all_docids_ordered = []
    first_batch = True

    # 3) Stream doc embeddings
    doc_generator = batch_embed_qwen_stream(
        model, tokenizer, docs, doc_ids,
        batch_size=batch_size, max_length=max_doc_len
    )

    for doc_embeds, dids in doc_generator:
        doc_embeds_np = doc_embeds.numpy()  # shape (batch_size, hidden_dim)

        # If 'cosine', L2 normalize
        if sim_type == "cosine":
            faiss.normalize_L2(doc_embeds_np)

        if first_batch:
            emb_dim = doc_embeds_np.shape[1]
            # We set up an IndexFlatIP for dot-product
            # and let 'cosine' be handled by normalization
            cpu_index = faiss.IndexFlatIP(emb_dim)
            first_batch = False

        cpu_index.add(doc_embeds_np)
        all_docids_ordered.extend(dids)

    # 4) Convert query embeddings to NumPy for searching
    query_embeds_np = query_embeddings.numpy()
    if sim_type == "cosine":
        faiss.normalize_L2(query_embeds_np)

    print(f"Searching for top-{top_k} documents per query...")
    distances, faiss_indices = cpu_index.search(query_embeds_np, top_k)
    print("Search complete.")

    # 5) Build run dictionary: run[qid][docid] = float(similarity)
    run = {}
    for q_idx, qid in enumerate(all_query_ids):
        run[str(qid)] = {}
        for rank in range(top_k):
            doc_idx = faiss_indices[q_idx, rank]
            if doc_idx < 0 or doc_idx >= len(all_docids_ordered):
                continue
            doc_id = all_docids_ordered[doc_idx]
            score = distances[q_idx, rank]
            run[str(qid)][str(doc_id)] = float(score)

    return run


from tqdm import trange

def batch_embed_qwen_stream(
    model,
    tokenizer,
    texts,
    ids,
    batch_size,
    max_length
):
    """
    Generator that yields (embeddings, ids) for Qwen in CPU float32 batches.

    Parameters
    ----------
    model : torch.nn.Module
        Qwen model.
    tokenizer : PreTrainedTokenizer
        The tokenizer corresponding to the Qwen model.
    texts : list of str
        List of texts (queries or documents).
    ids : list
        List of corresponding IDs (query_ids or doc_ids).
    batch_size : int
        Number of samples per batch.
    max_length : int
        Maximum token length for each text in the batch.

    Yields
    ------
    tuple of (torch.Tensor, list)
        A tuple (embeddings of shape (batch_size, hidden_dim), list of IDs).
    """
    n = len(texts)
    for start_idx in trange(0, n, batch_size, desc="Embedding in batches"):
        end_idx = min(start_idx + batch_size, n)
        batch_texts = texts[start_idx:end_idx]
        batch_ids = ids[start_idx:end_idx]

        # Tokenize
        tokenized = tokenize_with_manual_eos(tokenizer, batch_texts, max_length=max_length)
        input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long)

        # Move to GPU or CPU (depending on your model device)
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            embeddings = get_eos_embeddings(model, input_ids, attention_mask, tokenizer)
        # Move embeddings to CPU and float32
        embeddings = embeddings.cpu().float()

        yield embeddings, batch_ids


def embed_mamba(model, tokenizer, docs, queries, doc_ids, query_ids):
    from torch.utils.data import DataLoader, TensorDataset
    batch_size = 8
    device = next(model.parameters()).device  # Automatically detect model's device
    print("Device: ", device)

    inputs_docs = tokenizer(
        docs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_DOC_LEN
    )
    inputs_queries = tokenizer(
        queries,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_QUERY_LEN
    )
    print("Docs and queries tokenized.")
    inputs_docs = {key: tensor.to(device) for key, tensor in inputs_docs.items()}  # Move to device
    inputs_queries = {key: tensor.to(device) for key, tensor in inputs_queries.items()}  # Move to device

    # eos_token_id = tokenizer.eos_token_id
    # eos_tokens = torch.full((inputs_docs["input_ids"].size(0), 1), eos_token_id, dtype=torch.long).to(device)
    # attention_tokens = torch.ones((inputs_docs["attention_mask"].size(0), 1), dtype=torch.long).to(device)

    # inputs_docs["input_ids"] = torch.cat([inputs_docs["input_ids"], eos_tokens], dim=1)
    # inputs_docs["attention_mask"] = torch.cat([inputs_docs["attention_mask"], attention_tokens], dim=1)    

    # eos_token_id = tokenizer.eos_token_id
    # eos_tokens = torch.full((inputs_queries["input_ids"].size(0), 1), eos_token_id, dtype=torch.long).to(device)
    # attention_tokens = torch.ones((inputs_queries["attention_mask"].size(0), 1), dtype=torch.long).to(device)

    # inputs_queries["input_ids"] = torch.cat([inputs_queries["input_ids"], eos_tokens], dim=1)
    # inputs_queries["attention_mask"] = torch.cat([inputs_queries["attention_mask"], attention_tokens], dim=1)
    
    # Create DataLoaders for batching
    doc_dataset = TensorDataset(inputs_docs["input_ids"], inputs_docs["attention_mask"])
    query_dataset = TensorDataset(inputs_queries["input_ids"], inputs_queries["attention_mask"])
    doc_loader = DataLoader(doc_dataset, batch_size=batch_size)
    query_loader = DataLoader(query_dataset, batch_size=batch_size)

    print("Embedding docs and queries...")
    embeddings_docs = []
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(doc_loader):
            batch = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}
            output = model(**batch)
            embeddings_docs.append(output.logits.mean(dim=1))  # Mean pooling
    embeddings_docs = torch.cat(embeddings_docs, dim=0)  # Combine batches

    embeddings_queries = []
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(query_loader):
            batch = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}
            output = model(**batch)
            embeddings_queries.append(output.logits.mean(dim=1))  # Mean pooling
    embeddings_queries = torch.cat(embeddings_queries, dim=0)  # Combine batches
    print("Embeddings done.")

    run = compute_similarity(query_ids, doc_ids, embeddings_queries, embeddings_docs)
    return run


from contextlib import redirect_stdout

def embed_chunkwise_old(model, get_sim_func, docs, queries, doc_ids, query_ids, window_size=256):
    # Chunk the documents into smaller pieces.
    docs_chunked = []
    for doc in docs:
        doc_words = doc.split(" ")
        doc_chunked = [" ".join(doc_words[i:i+window_size]) 
                       for i in range(0, len(doc_words), window_size)]
        docs_chunked.append(doc_chunked)

    # Calculate the similarity between all queries and each document's chunks in parallel.
    similarities = []  # will store one similarity vector (one per query) for each document
    for doc_chunks in tqdm(docs_chunked, desc="Calculating similarities"):
        # get_sim_func returns a similarity matrix of shape [n_queries, n_chunks]
        with open(os.devnull, "w") as fnull, redirect_stdout(fnull):
            sim_matrix = get_sim_func(model, doc_chunks, queries)
        # For each query, take the maximum similarity over all chunks
        doc_sims, _ = sim_matrix.max(dim=1)  # shape: [n_queries]
        similarities.append(doc_sims)
    
    # Stack all document similarity vectors to form a [n_queries, n_docs] tensor.
    similarities = torch.stack(similarities, dim=1)
    
    return retrieve(query_ids, doc_ids, similarities)


def chunk_by_paragraphs(doc, window_size=256):
    """
    Chunk a document by paragraphs without splitting inside a paragraph.
    Paragraphs are defined by two consecutive newlines ("\n\n"). Consecutive paragraphs
    are merged until adding another paragraph would exceed the window_size in terms of word count.

    Parameters
    ----------
    doc : str
        The document text.
    window_size : int, optional
        Maximum number of words allowed per chunk (default is 256).

    Returns
    -------
    list of str
        List of text chunks, each containing one or more complete paragraphs.
    """
    # Split document into paragraphs and remove empty ones.
    paragraphs = [p.strip() for p in doc.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = []
    current_count = 0

    for para in paragraphs:
        word_count = len(para.split())
        # If adding this paragraph exceeds the window and we already have a chunk,
        # flush the current chunk.
        if current_chunk and (current_count + word_count > window_size):
            chunks.append(" ".join(current_chunk))
            current_chunk = [para]
            current_count = word_count
        else:
            current_chunk.append(para)
            current_count += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def embed_chunkwise(model, get_sim_func, docs, queries, doc_ids, query_ids, chunk_func, window_size=256):
    """
    Chunk documents using the provided chunking function and compute the similarity between all queries and 
    each document's chunks in one batch. Then, for each document, take the maximum similarity across its chunks.

    Parameters
    ----------
    model : object
        The model used for encoding.
    get_sim_func : callable
        A function that computes similarity given (model, docs, queries) and returns a similarity matrix.
        In your case, this is something like get_sim_bge.
    docs : list of str
        List of documents.
    queries : list of str
        List of queries.
    doc_ids : list
        List of document IDs.
    query_ids : list
        List of query IDs.
    chunk_func : callable
        Function that takes a document (str) and window_size, and returns a list of chunks.
    window_size : int, optional
        Parameter to pass to the chunking function (default is 256).

    Returns
    -------
    dict
        Retrieval run in the expected format (as returned by the retrieve function).
    """
    # Step 1: Chunk each document using the provided chunking function.
    docs_chunked = []
    doc_chunk_counts = []  # keep track of the number of chunks per document
    for doc in docs:
        chunks = chunk_func(doc, window_size)
        docs_chunked.append(chunks)
        doc_chunk_counts.append(len(chunks))
    
    # Step 2: Flatten all chunks into a single list.
    flatten_chunks = []
    for chunks in docs_chunked:
        flatten_chunks.extend(chunks)
    
    # Step 3: Compute similarity between all queries and all document chunks in one call.
    # sim_all will have shape [num_queries, total_chunks]
    sim_all = get_sim_func(model, flatten_chunks, queries)
    
    # Step 4: For each document, select its corresponding chunk columns and take max over chunks.
    num_queries = sim_all.shape[0]
    num_docs = len(docs)
    sim_matrix = torch.empty((num_queries, num_docs))
    
    start_idx = 0
    for doc_idx, count in enumerate(doc_chunk_counts):
        # For this document, select its corresponding columns.
        sim_doc = sim_all[:, start_idx:start_idx+count]  # shape: [num_queries, count]
        # Maximum similarity for each query for this document.
        sim_max, _ = sim_doc.max(dim=1)
        sim_matrix[:, doc_idx] = sim_max
        start_idx += count
    
    return retrieve(query_ids, doc_ids, sim_matrix)


def get_sim_sbert(model, docs, queries):
    embeddings_queries = model.encode(queries)
    embeddings_docs = model.encode(docs)
    similarity = model.similarity(embeddings_queries, embeddings_docs)
    return similarity


def embed_s_transformers(model, docs, queries, doc_ids, query_ids):
    embeddings_queries = model.encode(queries)
    embeddings_docs = model.encode(docs)
    similarity = model.similarity(embeddings_queries, embeddings_docs)
    return retrieve(query_ids, doc_ids, similarity)


def embed_s_transformers_faiss(
    model,
    docs: list,
    doc_ids: list,
    queries: list,
    query_ids: list,
    top_k: int = 1000,
    batch_size: int = 512,
    similarity: str = "dot",
):
    """
    Embed documents + queries with SentenceTransformers in a streaming fashion,
    build a FAISS index of doc embeddings, and retrieve top_k results per query.

    Parameters
    ----------
    model : SentenceTransformer
        The SentenceTransformers model for embedding.
    docs : list of str
        List of document texts.
    doc_ids : list
        List of document IDs, parallel to docs.
    queries : list of str
        List of query texts.
    query_ids : list
        List of query IDs.
    top_k : int, optional
        Number of docs to retrieve per query (default 1000).
    batch_size : int, optional
        Batch size for embedding docs (default 512).
    similarity : str, optional
        Either "dot" or "cosine". Defaults to "dot".

    Returns
    -------
    run : dict
        A run dict mapping query_id -> {doc_id: score, ...}
        suitable for TREC-style evaluation (pytrec_eval).
    """
    # ---- 1) Embed Queries in memory ----
    print("Embedding queries in memory...")
    query_embeddings = model.encode(queries, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
    query_embeddings = query_embeddings.astype(np.float32)  # FAISS prefers float32

    if similarity == "cosine":
        # L2-normalize queries to mimic cosine with IndexFlatIP
        faiss.normalize_L2(query_embeddings)

    print("Finished embedding queries. Now building doc index...")

    # ---- 2) Create a FAISS index in CPU memory ----
    # For dot product or "IP", we can keep it simple with IndexFlatIP
    # If you're truly at 8.8M docs, consider IVFPQ or HNSW for memory usage
    index = None
    emb_dim = None
    all_doc_ids = []

    # ---- 2a) Stream doc embeddings in BATCHES ----
    # We'll embed docs in chunks to avoid OOM
    num_docs = len(docs)
    for start_idx in tqdm(range(0, num_docs, batch_size), desc="Indexing docs"):
        end_idx = min(start_idx + batch_size, num_docs)
        doc_batch = docs[start_idx:end_idx]
        doc_id_batch = doc_ids[start_idx:end_idx]

        # Embed the documents
        doc_embeddings = model.encode(doc_batch, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        doc_embeddings = doc_embeddings.astype(np.float32)

        if similarity == "cosine":
            # L2-normalize doc embeddings if using "cosine"
            faiss.normalize_L2(doc_embeddings)

        # On first batch, init the index
        if index is None:
            emb_dim = doc_embeddings.shape[1]
            index = faiss.IndexFlatIP(emb_dim)  # Dot-product index
        # Add embeddings
        index.add(doc_embeddings)
        all_doc_ids.extend(doc_id_batch)

    print(f"Done building index with {index.ntotal} documents.")

    # ---- 3) Search for top_k docs per query ----
    print("Searching index for each query...")
    distances, faiss_indices = index.search(query_embeddings, top_k)
    print("Done searching.")

    # ---- 4) Build the run dictionary (for TREC eval) ----
    run = {}
    for q_idx, qid in enumerate(query_ids):
        run[str(qid)] = {}
        for rank in range(top_k):
            doc_idx = faiss_indices[q_idx, rank]
            if doc_idx < 0 or doc_idx >= len(all_doc_ids):
                continue
            doc_id = all_doc_ids[doc_idx]
            score = distances[q_idx, rank]
            run[str(qid)][str(doc_id)] = float(score)

    return run


def retrieve_bm25(docs, queries, doc_ids, query_ids):
    """
    Embed the queries and documents using the BM25 model and compute the similarity between queries and documents.

    Args:
        docs (dict): Dictionary with document_id as key and text as value.
        queries (list): List of queries.
        doc_ids (list): List of document IDs.
        top_k (int): Number of most similar documents to retrieve.

    Returns:
        dict: Dictionary with (query_id, doc_id) as key and a list of tuples of score as value.
    """
    from rank_bm25 import BM25Okapi

    print("Tokenizing corpus...", end="")
    # Simple space-based tokenization
    tokenized_corpus = [doc.lower().split() for doc in tqdm(docs)]
    print("Done.")

    print("Creating BM25 model...", end="")
    # Create BM25 model
    bm25 = BM25Okapi(tokenized_corpus)
    print("Done.")

    print("Tokenizing queries...", end="")
    # Queries
    tokenized_queries = [query.lower().split() for query in tqdm(queries)]
    print("Done.")

    # key: query, value: (similarity, text, doc_id)
    run = {}
    for tokenized_query, query_id in tqdm(zip(tokenized_queries, query_ids), total=len(tokenized_queries)):
        # Calcular las puntuaciones BM25 para la consulta en cada documento
        scores = bm25.get_scores(tokenized_query)

        run_query = {}
        for doc_id, score in zip(doc_ids, scores):
            run_query[str(doc_id)] = score
        run[str(query_id)] = run_query
    return run


def merge_reranked_into_full_run(full_run, reranked_run):
    merged_run = {}
    for qid, orig_scores in full_run.items():
        rerank_scores = reranked_run.get(qid, {})

        if rerank_scores:
            # find absolute minimum CE score so we can dominate it
            min_ce = min(rerank_scores.values())
            # offset = max_original - min_ce + ε  (ε=1 e‑3 keeps ties broken)
            offset = max(orig_scores.values()) - min_ce + 1e-3
        else:
            offset = 0

        updated = {}
        for did, score in orig_scores.items():
            updated[did] = (
                rerank_scores[did] + offset if did in rerank_scores else score
            )

        # store sorted by score desc
        merged_run[qid] = dict(sorted(updated.items(), key=lambda x: x[1], reverse=True))
    return merged_run


def rerank_cross_encoder_chunked(model,
                                 model_type,
                                 tokenizer,
                                 run,
                                 top_k,
                                 query_dict,
                                 doc_dict,
                                 max_length,
                                 stride,
                                 aggregator="max",
                                 batch_size=512):
    """
    Rerank the provided run using a cross-encoder with sliding-window chunking.

    This function splits each document into overlapping chunks using the
    tokenizer's `return_overflowing_tokens` feature, scores each chunk,
    aggregates the chunk scores into a single document-level score, and
    merges the reranked scores back into the full run.

    Parameters
    ----------
    model : torch.nn.Module
        The trained cross-encoder model (or placeholder for BGE/SBERT).
    model_type : str
        One of ['binary', 'fine_grained', 'ranknet', 'bge', 'sbert'].
    tokenizer : transformers.PreTrainedTokenizer
        Corresponding tokenizer.
    run : dict
        Initial retrieval run mapping query_id → {doc_id: score}.
    top_k : int
        Number of top candidates to keep per query.
    query_dict : dict
        Mapping from query_id → query text.
    doc_dict : dict
        Mapping from doc_id → full document text.
    max_length : int
        Maximum token length per chunk.
    stride : int
        Number of overlapping tokens between chunks.
    aggregator : {'max', 'mean'}, optional
        How to aggregate chunk scores into one document score.
        Default is 'max'.
    batch_size : int, optional
        Batch size for scoring chunks. Default is 8.

    Returns
    -------
    dict
        Merged run mapping each query_id → {doc_id: aggregated_score}.
    """
    device = torch.device("cuda")
    # instantiate or prepare model exactly as in your normal reranker
    if model_type == "bge":
        from FlagEmbedding import FlagLLMReranker
        model = FlagLLMReranker('BAAI/bge-reranker-v2-gemma', use_bf16=True)
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-gemma')
    elif model_type == "sbert":
        from sentence_transformers import CrossEncoder
        model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
        tokenizer = AutoTokenizer.from_pretrained('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

    reranked_run = {}

    for qid, scores_dict in tqdm(run.items(), desc="Queries", leave=False):
        # Pick top_k docs for this query
        top_docs = sorted(
            scores_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        doc_ids = [doc_id for doc_id, _ in top_docs]
        docs = [doc_dict[str(doc_id)] for doc_id in doc_ids]
        query_texts = [query_dict[qid]] * len(docs)

        # Tokenize all (query, doc) pairs with overflowing chunks
        encoding = tokenizer(
            query_texts,
            docs,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_tensors="pt"
        ).to(device)

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        token_type_ids = encoding.get("token_type_ids")
        overflow_map = encoding["overflow_to_sample_mapping"].tolist()

        # Collect chunk scores per document
        doc_chunk_scores = {doc_id: [] for doc_id in doc_ids}
        total_chunks = input_ids.size(0)

        # Batch-score all chunks
        for start in range(0, total_chunks, batch_size):
            end = start + batch_size
            batch = {
                "input_ids": input_ids[start:end],
                "attention_mask": attention_mask[start:end]
            }
            if token_type_ids is not None:
                batch["token_type_ids"] = token_type_ids[start:end]

            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits

            # Extract scores based on head
            if model_type == "binary" and logits.size(-1) == 2:
                scores = torch.softmax(logits, dim=-1)[:, 1]
            elif model_type == "binary":
                scores = logits.squeeze(-1)
            elif model_type == "fine_grained":
                probs = torch.softmax(logits, dim=-1)
                idx = torch.arange(
                    logits.size(-1), device=device, dtype=torch.float
                )
                scores = (probs * idx).sum(dim=-1)
            elif model_type == "ranknet":
                scores = logits[:, 0]
            else:
                scores = logits.squeeze(-1)

            # Map chunk scores back to documents
            for idx, score in enumerate(scores.cpu().tolist(), start=start):
                doc_idx = overflow_map[idx]
                doc_id = doc_ids[doc_idx]
                doc_chunk_scores[doc_id].append(score)

        # Aggregate scores per document
        reranked_run[qid] = {}
        for doc_id, chunk_scores in doc_chunk_scores.items():
            if not chunk_scores:
                agg_score = float("-inf")
            elif aggregator == "max":
                agg_score = max(chunk_scores)
            elif aggregator == "mean":
                agg_score = sum(chunk_scores) / len(chunk_scores)
            elif aggregator == "top3":
                top3 = sorted(chunk_scores, reverse=True)[:3]
                agg_score = sum(top3) / len(top3)
            elif aggregator == "top10":
                top10 = sorted(chunk_scores, reverse=True)[:10]
                agg_score = sum(top10) / len(top10)
            else:
                raise ValueError(f"Unknown aggregator: {aggregator}")
            reranked_run[qid][doc_id] = float(agg_score)

    # Merge with original run if needed
    return merge_reranked_into_full_run(run, reranked_run)


def rerank_cross_encoder(model, model_type, tokenizer, run, top_k, query_dict, doc_dict, max_length, batch_size=8):
    """
    Rerank the provided run using a cross-encoder scorer. This function creates a 
    CrossEncoderScorer instance that abstracts the underlying model's scoring mechanism,
    then processes each query in batches, computing a single relevance score per query–document 
    pair.

    Parameters
    ----------
    model : torch.nn.Module
        The trained cross-encoder model.
    model_type : str
        The type of the model (e.g., "bge", "ranknet").
    tokenizer : transformers.PreTrainedTokenizer
        The corresponding tokenizer.
    run : dict
        Initial retrieval run, where each query_id maps to a dict of doc_id: score.
    top_k : int
        The maximum number of candidate documents to consider per query.
    queries : list of str
        List of query texts.
    query_ids : list
        List of query identifiers.
    docs : list of str
        List of document texts.
    doc_ids : list
        List of document identifiers.
    max_length : int
        Maximum sequence length for tokenization.
    batch_size : int, optional
        Batch size for evaluation (default is 8).

    Returns
    -------
    dict
        A merged run with the reranked scores.
    """
    if model_type == "bge":
        from FlagEmbedding import FlagLLMReranker
        # model = FlagLLMReranker('BAAI/bge-reranker-v2-m3', use_bf16=False)
        model = FlagLLMReranker('BAAI/bge-reranker-v2-gemma', use_bf16=False)
    elif model_type == "sbert":
        from sentence_transformers import CrossEncoder
        model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

    # Limit each query's candidates to top_k.
    reranked_run = {}
    for query_id in run:
        reranked_run[query_id] = dict(sorted(run[query_id].items(), key=lambda x: x[1], reverse=True)[:top_k])

    # Instantiate the scorer.
    scorer = CrossEncoderScorer(model, tokenizer, head_type=model_type)

    # Iterate over each query.
    for query_id, doc_score_dict in tqdm(reranked_run.items(), total=len(reranked_run)):
        curr_doc_ids = list(doc_score_dict.keys())
        query_scores = []
        query_text = query_dict[query_id]
        # Process in batches.
        for start_idx in range(0, len(curr_doc_ids), batch_size):
            end_idx = min(start_idx + batch_size, len(curr_doc_ids))
            batch_doc_ids = curr_doc_ids[start_idx:end_idx]
            batch_doc_texts = [doc_dict[str(doc_id)] for doc_id in batch_doc_ids]
            scores = scorer.score([query_text]*len(batch_doc_texts), batch_doc_texts, max_length=max_length)
            # score = random.random()
            query_scores.extend(scores)
        # Build a new dict mapping doc_ids to their new scores.
        reranked_run[query_id] = {curr_doc_ids[i]: float(query_scores[i]) for i in range(len(curr_doc_ids))}

    return merge_reranked_into_full_run(run, reranked_run)   


def get_eval_metrics(run, qrels_dev_df, all_docids, metrics):
    # Evaluate BM25
    # qrels = {query_id: {doc_id: relevance, ...},
    #          query_id: {doc_id: relevance, ...}, ...},
    # convert qrels_dev_df to qrels dict
    qrels = {}
    for _, row in qrels_dev_df.iterrows():
        # row headers "query_id", "iteration", "doc_id", "relevance"
        query_id = str(row["query_id"])
        doc_id = str(row["doc_id"])
        relevance = int(row["relevance"])

        if query_id not in qrels:
            qrels[query_id] = {}
        qrels[query_id][doc_id] = relevance
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    # check instance of run
    if len(run) < 10:
        metrics_all = {}
        for run_name, run_ in run.items():
            results = evaluator.evaluate(run_)
    
            result_values = list(results.values())
            metric_names = list(result_values[0].keys())      # because some result names change e.g. from ndcg_cut.10 to ndcg_cut_10
            metric_sums = {metric_name: 0 for metric_name in metric_names}
            for metrics_ in results.values():
                for metric in metric_names:
                    metric_sums[metric] += metrics_[metric]
            
            # Average metrics over all queries
            avg_metrics = {metric_name: metric_sums[metric_name]/len(results) for metric_name in metric_names}
            
            metrics_all[run_name] = avg_metrics

            print(f"\nResults for {run_name}:")
            for metric_name, metric_value in avg_metrics.items():
                print(f"Average {metric_name}: {metric_value}")

            print("\n")
    
    else:
        results = evaluator.evaluate(run)

        with open("metrics_per_query.pkl", "wb") as f:
            pickle.dump(results, f)
    
        result_values = list(results.values())
        metric_names = list(result_values[0].keys())      # because some result names change e.g. from ndcg_cut.10 to ndcg_cut_10
        metric_sums = {metric_name: 0 for metric_name in metric_names}
        for metrics_ in results.values():
            for metric in metric_names:
                metric_sums[metric] += metrics_[metric]
        
        # Average metrics over all queries
        # assert len(results) == len(query_ids)
        avg_metrics = {metric_name: metric_sums[metric_name]/len(results) for metric_name in metric_names}
        
        print("\nResults:")
        for metric_name, metric_value in avg_metrics.items():
            print(f"Average {metric_name}: {metric_value}")

        print("\n")

        return avg_metrics


def create_results_file(run):
    """
    run has the following stucture:
    run[str(query_ids[i])] = {str(doc_ids[j]): float(query_sim[j]) for j in range(len(doc_ids))}
    {query_id: {doc_id: similarity, ...}, ...}
    """
    if len(run) > 10:
        # sort run dict by similarity
        for query_id in run:
            run[query_id] = dict(sorted(run[query_id].items(), key=lambda x: x[1], reverse=True)[:10])
        
        with open("results.txt", "w") as f:
            for query_id, doc_dict in run.items():
                for i, (doc_id, similarity) in enumerate(doc_dict.items()):
                    f.write(f"{query_id}\t{doc_id}\t{i+1}\n")
        return "results.txt"
    else:
        out_paths = []
        for type_, type_run in run.items():
            for query_id in type_run:
                type_run[query_id] = dict(sorted(type_run[query_id].items(), key=lambda x: x[1], reverse=True)[:10])
            
            out_path = f"results_{type_}.txt"
            out_paths.append(out_path)
            with open(out_path, "w") as f:
                for query_id, doc_dict in type_run.items():
                    for i, (doc_id, similarity) in enumerate(doc_dict.items()):
                        f.write(f"{query_id}\t{doc_id}\t{i+1}\n")
        return out_paths
    

def create_predictions_file(run, run_id="my_run"):
    """
    Generates a TREC-formatted predictions file from the run dictionary.

    The run dictionary has the following structure:
        {
            str(query_id): {str(doc_id): float(score), ...},
            ...
        }
    
    The TREC format for a result is:
        <query_id> Q0 <doc_id> <rank> <score> <run_id>
    
    Parameters
    ----------
    run : dict
        Dictionary mapping query IDs to dictionaries of document IDs and their scores.
    run_id : str, optional
        Identifier for the run (default is "my_run").
    
    Returns
    -------
    None
        Writes the results to a file named "predictions.tsv".
    """
    if len(run) > 10:
        with open("predictions.tsv", "w") as f:
            for query_id, doc_scores in run.items():
                # Sort documents by score in descending order and take top 10
                sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
                    # Write in TREC format: query_id, Q0, doc_id, rank, score, run_id
                    f.write(f"{query_id}\tQ0\t{doc_id}\t{rank}\t{score:.4f}\t{run_id}\n")
        return "predictions.tsv"
    else:
        out_paths = []
        for type_, type_run in run.items():
            out_path = f"predictions_{type_}.tsv"
            out_paths.append(out_path) 
            with open(f"predictions_{type_}.tsv", "w") as f:
                for query_id, doc_scores in type_run.items():
                    # Sort documents by score in descending order and take top 10
                    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                    for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
                        # Write in TREC format: query_id, Q0, doc_id, rank, score, run_id
                        f.write(f"{query_id}\tQ0\t{doc_id}\t{rank}\t{score:.4f}\t{run_id}\n")
        return out_paths
    

import json
def get_legal_dataset(path):
    """
    Load the legal dataset from a JSON or CSV file.
    
    Args:
        path (str): Path to the JSON or CSV file.
    
    Returns:
        tuple: A tuple containing two lists:
            - List of document IDs (Codigo).
            - List of document texts (text).
    """
    if path.endswith(".json"):
        with open(path, 'r', encoding='utf-8') as f:
            corpus_dict = json.load(f)
        df = pd.DataFrame(list(corpus_dict.items()), columns=["Codigo", "text"])
        # rename columns
        df.rename(columns={"Codigo": "id"}, inplace=True)
    elif path.endswith(".jsonl"):
        # Load the dataset
        with open(path, 'r', encoding='utf-8') as f:
            corpus_dict = [json.loads(line) for line in f]
        df = pd.DataFrame(corpus_dict)
        # drop columns
        df.drop(columns=["title"], inplace=True)
    else:
        raise ValueError("Path must end with .json or .jsonl")
    # convert Codigo column datatype to str
    df["id"] = df["id"].astype(str)
    return df["id"].tolist(), df["text"].tolist()


def get_legal_queries(path):
    if not path.endswith(".tsv"):
        raise ValueError("Path must end with .tsv")
    df = pd.read_csv(path, sep="\t", header=0, names=["id", "query"])
    # convert topic_id column to list
    df["id"] = df["id"].astype(str)
    return df["id"].tolist(), df["query"].tolist()