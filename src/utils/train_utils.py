from config.config import MAX_QUERY_LEN, MAX_DOC_LEN
from tqdm import tqdm
import os
import pickle
from datasets import load_dataset
import torch
from config.config import STORAGE_DIR
from torch.nn import functional as F


def get_msmarco_queries():
    print("Loading MS MARCO queries...", end="")
    save_path = os.path.join(STORAGE_DIR, "ms_marco_passage", "data", "qid_to_query.pkl")
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            qid_to_query = pickle.load(f)
    else:
        query_dataset = load_dataset("sentence-transformers/msmarco-corpus", "query", split="train")
        qid_to_query = dict(zip(query_dataset["qid"], query_dataset["text"]))
        # print(qid_to_query[571018])
        # => "what are the liberal arts?"
        with open(save_path, "wb") as f:
            pickle.dump(qid_to_query, f)
    print("Done")
    return qid_to_query


def get_msmarco_passages():
    print("Loading MS MARCO passages...", end="")
    save_path = os.path.join(STORAGE_DIR, "ms_marco_passage", "data", "pid_to_passage.pkl")
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            pid_to_passage = pickle.load(f)
    else:
        passage_dataset = load_dataset("sentence-transformers/msmarco-corpus", "passage", split="train")
        pid_to_passage = dict(zip(passage_dataset["pid"], passage_dataset["text"]))
        # print(pid_to_passage[7349777])
        # => "liberal arts. 1. the academic course of instruction at a college 
        with open(save_path, "wb") as f:
            pickle.dump(pid_to_passage, f)
    print("Done")
    return pid_to_passage


def get_msmarco_hard_negatives(num_negs, reload=False):
    print("Loading hard negatives...", end="")
    save_path = os.path.join(STORAGE_DIR, "ms_marco_passage", "data", f"negatives_{num_negs}_msmarco.pkl")
    if os.path.exists(save_path) and not reload:
        with open(save_path, "rb") as f:
            negs_ds = pickle.load(f)
    else:
        negs_ds = load_dataset("sentence-transformers/msmarco-msmarco-distilbert-base-tas-b", "triplet-50-ids")
        negs_ds = negs_ds["train"]
        remove_cols = [f"negative_{i+1}" for i in range(num_negs, 50)]
        negs_ds = negs_ds.map(lambda x: x, remove_columns=remove_cols)
        # save to disk
        with open(save_path, "wb") as f:
            pickle.dump(negs_ds, f)
    print("Done")
    return negs_ds


def tokenize_train_ds_msmarco(tokenizer, train_ds, qid_to_query, pid_to_passage, num_negs, reuse=False):
    print("Tokenizing train dataset...")
    save_path = os.path.join(STORAGE_DIR, "ms_marco_passage", "data", f"train_ds_msmarco_{num_negs}negs_50k.pkl")
    if os.path.exists(save_path) and reuse:
        with open(save_path, "rb") as f:
            train_ds = pickle.load(f)
    else:
        train_ds = train_ds.map(lambda x: tokenize_with_hard_negatives_msmarco(tokenizer, x, qid_to_query, pid_to_passage, num_negs, MAX_QUERY_LEN, MAX_DOC_LEN), batched=True)
        with open(save_path, "wb") as f:
            pickle.dump(train_ds, f)
    print("Done")
    return train_ds


def tokenize_test_ds_msmarco(tokenizer, test_ds, qid_to_query, pid_to_passage, num_negs, reuse=False):
    print("Tokenizing test dataset...")
    save_path = os.path.join(STORAGE_DIR, "ms_marco_passage", "data", f"test_ds_msmarco_{num_negs}negs_50k.pkl")
    if os.path.exists(save_path) and reuse:
        with open(save_path, "rb") as f:
            test_ds = pickle.load(f)
    else:
        test_ds = test_ds.map(lambda x: tokenize_with_hard_negatives_msmarco(tokenizer, x, qid_to_query, pid_to_passage, num_negs, MAX_QUERY_LEN, MAX_DOC_LEN), batched=True)
        with open(save_path, "wb") as f:
            pickle.dump(test_ds, f)
    print("Done")
    return test_ds


def get_eos_embeddings(model, input_ids, attention_mask, tokenizer):
    """
    Extract L2-normalized embeddings of exactly one EOS token per sequence.
    Assumes:
      - Each sequence in `input_ids` has exactly one EOS token.
      - That EOS token is `tokenizer.eos_token_id`.
    Raises an error if any sequence has zero or multiple EOS tokens.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False
    )
    hidden_states = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)

    batch_size = input_ids.size(0)      # for hard negs it is batch_size*n_neg
    eos_id = tokenizer.eos_token_id

    # 1) Find positions of all eos_id tokens in the batch
    # eos_positions is (row_indices, col_indices), each shape (#eos_tokens_found,)
    eos_positions = (input_ids == eos_id).nonzero(as_tuple=True)
    row_indices, col_indices = eos_positions

    # 2) Assert that the number of EOS tokens found == batch_size
    if row_indices.size(0) != batch_size:
        raise ValueError(
            f"Expected exactly 1 EOS token per sequence, "
            f"but found {row_indices.size(0)} matches for {batch_size} sequences."
        )

    # 3) Check that each sequence index appears exactly once
    #    i.e., no sequence is missing an EOS or has multiple
    unique_rows, counts = row_indices.unique(return_counts=True)
    if unique_rows.size(0) < batch_size:
        raise ValueError("At least one sequence does not contain any EOS token.")
    if (counts > 1).any():
        raise ValueError("At least one sequence has multiple EOS tokens.")

    # 4) Gather the embeddings for the single EOS token per sequence
    eos_embeds = hidden_states[row_indices, col_indices, :]  # shape (batch_size, hidden_dim)

    # # Use attention mask to get the index of the last non-pad token in each sequence.
    # last_token_idx = attention_mask.sum(dim=1) - 1
    # batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
    
    # last_token_ids = input_ids[batch_indices, last_token_idx]
    # assert (last_token_ids == tokenizer.eos_token_id).all(), "Not all sequences end with the EOS token."
    
    # eos_embeds = hidden_states[batch_indices, last_token_idx, :]

    # 5) Normalize embeddings to unit L2 norm
    eos_embeds = F.normalize(eos_embeds, p=2, dim=-1)

    return eos_embeds


def tokenize_with_manual_eos(tokenizer, text_list, max_length):
    """
    Tokenize each item in text_list to max_length - 1, 
    then manually append EOS, then (optional) pad up to max_length.
    """
    # 1) Tokenize with max_length - 1, no padding.
    #    We rely on a custom approach to handle final EOS and padding.
    partial_encoded = tokenizer(
        text_list,
        truncation=True,
        max_length=max_length - 1,
        padding=False,
    )

    # 2) Manually append EOS for each sequence
    eos_id = tokenizer.eos_token_id
    new_input_ids = []
    new_attention_masks = []

    for inp_ids, att_mask in zip(partial_encoded["input_ids"], partial_encoded["attention_mask"]):
        # Append EOS token
        inp_ids.append(eos_id)
        att_mask.append(1)

        # 3) (Optional) Now pad if we want a fixed size == max_length
        #    If you truly want each sequence to be exactly max_length:
        pad_len = max_length - len(inp_ids)
        if pad_len > 0:
            inp_ids.extend([tokenizer.pad_token_id] * pad_len)
            att_mask.extend([0] * pad_len)

        new_input_ids.append(inp_ids)
        new_attention_masks.append(att_mask)

    return {
        "input_ids": new_input_ids,
        "attention_mask": new_attention_masks,
    }


def tokenize_with_hard_negatives_messirve(tokenizer, examples, append_eos=False):
    # Flatten the list of hard negatives for tokenization
    flattened_negatives = []
    for hard_neg_docs in examples["hard_negatives"]:
        flattened_negatives.extend(hard_neg_docs)
    
    # Tokenize queries
    tokenized_queries = tokenize_with_manual_eos(tokenizer, examples["query"], max_length=MAX_QUERY_LEN)
    
    # Tokenize positive documents
    tokenized_docs = tokenize_with_manual_eos(tokenizer, examples["docid_text"], max_length=MAX_DOC_LEN)
    
    # Tokenize hard negatives (flattened)
    tokenized_all_negatives = tokenize_with_manual_eos(tokenizer, flattened_negatives, max_length=MAX_DOC_LEN)
    
    # Rolling index to rebuild the structure
    rolling_index = 0
    neg_input_ids = []
    neg_attention_masks = []
    for neg_list in examples["hard_negatives"]:
        length = len(neg_list)
        neg_input_ids.append(
            tokenized_all_negatives["input_ids"][rolling_index : rolling_index + length]
        )
        neg_attention_masks.append(
            tokenized_all_negatives["attention_mask"][rolling_index : rolling_index + length]
        )
        rolling_index += length
    
    return {
        "query_input_ids": tokenized_queries["input_ids"],
        "query_attention_mask": tokenized_queries["attention_mask"],
        "doc_input_ids": tokenized_docs["input_ids"],
        "doc_attention_mask": tokenized_docs["attention_mask"],
        "neg_input_ids": neg_input_ids,
        "neg_attention_mask": neg_attention_masks,
    }


def tokenize_with_hard_negatives_msmarco(tokenizer, examples: dict, qid_to_query, pid_to_passage, num_negs, max_query_len, max_doc_len):
    """
    Due to dataset.map(batched=True) parameter, examples is a dictionary where
    each key corresponds to a column in your dataset and the value is a list
    of items for that column.
    Example:
    {
        "query": ["how to make pizza", "how to make pasta"],
        "positive": ["To make pizza, you need...", "To make pasta, you need..."],
        "negative_1": ["To make a cake, you need...", "To make a salad, you need..."],
        "negative_2": ["To make a sandwich, you need...", "To make a burger, you need..."],
        ...
    }
    """
    queries = [qid_to_query[qid] for qid in examples["query"]]
    positives = [pid_to_passage[pid] for pid in examples["positive"]]

    # Flatten the list of hard negatives for tokenization
    flattened_negatives = []
    for i in range(num_negs):
        neg_pids = examples[f"negative_{i+1}"]
        flattened_negatives.extend([pid_to_passage[neg_pid] for neg_pid in neg_pids])
    
    # Tokenize queries
    tokenized_queries = tokenize_with_manual_eos(tokenizer, queries, max_length=max_query_len)

    # Tokenize positive documents
    tokenized_docs = tokenize_with_manual_eos(tokenizer, positives, max_length=max_doc_len)

    # Tokenize hard negatives (flattened)
    tokenized_all_negatives = tokenize_with_manual_eos(tokenizer, flattened_negatives, max_length=max_doc_len)

    # Rolling index to rebuild the structure
    rolling_index = 0
    neg_input_ids = []
    neg_attention_masks = []
    for i in range(len(examples["negative_1"])):
        neg_input_ids.append(
            tokenized_all_negatives["input_ids"][rolling_index : rolling_index + num_negs]
        )
        neg_attention_masks.append(
            tokenized_all_negatives["attention_mask"][rolling_index : rolling_index + num_negs]
        )
        rolling_index += num_negs
    
    return {
        "query_input_ids": tokenized_queries["input_ids"],
        "query_attention_mask": tokenized_queries["attention_mask"],
        "doc_input_ids": tokenized_docs["input_ids"],
        "doc_attention_mask": tokenized_docs["attention_mask"],
        "neg_input_ids": neg_input_ids,     # list of lists
        "neg_attention_mask": neg_attention_masks,      # list of lists
    }


def tokenize_function(tokenizer, examples, append_eos=False):
    # Append the EOS token to queries and documents
    if append_eos:
        examples["query"] = [q + tokenizer.eos_token for q in examples["query"]]
        examples["docid_text"] = [d + tokenizer.eos_token for d in examples["docid_text"]]

    tokenized_queries = tokenizer(examples["query"], truncation=True, padding=True, max_length=MAX_QUERY_LEN)
    tokenized_docs = tokenizer(examples["docid_text"], truncation=True, padding=True, max_length=MAX_DOC_LEN)

    # Return tokenized queries and documents
    return {
        "query_input_ids": tokenized_queries["input_ids"],
        "query_attention_mask": tokenized_queries["attention_mask"],
        "doc_input_ids": tokenized_docs["input_ids"],
        "doc_attention_mask": tokenized_docs["attention_mask"],
    }


def custom_data_collator(batch):
    """
    Since tokenize_with_manual_eos() already pads each sequence to a uniform max_length,
    here we just stack them into tensors. We assume:
      - Each example in 'batch' has identical shapes for query, doc, etc.
      - The number of negatives (e.g. n_neg=5) is the same for all examples.

      QUESTION: Are we assuming that the query length is the same as the doc length?
    """
    # -- Queries --
    query_input_ids = torch.stack(
        [torch.tensor(example["query_input_ids"], dtype=torch.long) for example in batch],
        dim=0
    )  # shape: (batch_size, query_seq_len)
    query_attention_mask = torch.stack(
        [torch.tensor(example["query_attention_mask"], dtype=torch.long) for example in batch],
        dim=0
    )  # shape: (batch_size, query_seq_len)

    # -- Positive Docs --
    doc_input_ids = torch.stack(
        [torch.tensor(example["doc_input_ids"], dtype=torch.long) for example in batch],
        dim=0
    )  # shape: (batch_size, doc_seq_len)
    doc_attention_mask = torch.stack(
        [torch.tensor(example["doc_attention_mask"], dtype=torch.long) for example in batch],
        dim=0
    )  # shape: (batch_size, doc_seq_len)

    # -- Hard Negatives --
    # Each example["neg_input_ids"] is a list of length n_neg,
    # where each item is a list of length doc_seq_len (already padded).
    # So example["neg_input_ids"] => shape (n_neg, doc_seq_len)
    neg_input_ids = torch.stack(
        [torch.tensor(example["neg_input_ids"], dtype=torch.long) for example in batch],
        dim=0
    )  # shape: (batch_size, n_neg, doc_seq_len)

    neg_attention_mask = torch.stack(
        [torch.tensor(example["neg_attention_mask"], dtype=torch.long) for example in batch],
        dim=0
    )  # shape: (batch_size, n_neg, doc_seq_len)

    return {
        "query_input_ids": query_input_ids,
        "query_attention_mask": query_attention_mask,
        "doc_input_ids": doc_input_ids,
        "doc_attention_mask": doc_attention_mask,
        "neg_input_ids": neg_input_ids,
        "neg_attention_mask": neg_attention_mask,
    }