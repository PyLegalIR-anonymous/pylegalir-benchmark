import torch
import logging
import contextlib
import sys
import os


@contextlib.contextmanager
def silence_stdout():
    """
    Context manager that temporarily redirects both standard output (stdout) and
    standard error (stderr) to os.devnull, effectively muting any print statements
    or error outputs.

    Yields
    -------
    None
        Control is yielded back to the context block where output is silenced.

    Examples
    --------
    >>> def noisy_function():
    ...     print("You won't see this.")
    ...     raise Exception("Nor this error message!")
    ...
    >>> with silence_stdout():
    ...     try:
    ...         noisy_function()
    ...     except Exception:
    ...         pass
    >>> print("Output restored.")
    """
    # Open the null device file which discards all input
    null_f = open(os.devnull, 'w')
    # Save the current stdout and stderr so we can restore them later
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = null_f, null_f
    try:
        yield
    finally:
        # Restore the original stdout and stderr even if an error occurs
        sys.stdout, sys.stderr = old_stdout, old_stderr
        null_f.close()


class CrossEncoderScorer:
    """
    A wrapper class for scoring a list of query–document pairs using a cross-encoder.
    This class abstracts away the differences between models trained with different
    objectives (binary classification, fine-grained classification, RankNet, etc.).
    
    Parameters
    ----------
    model : torch.nn.Module or custom scoring object
        The trained cross-encoder model.
    tokenizer : transformers.PreTrainedTokenizer
        The corresponding tokenizer.
    head_type : str, optional
        The head type used during training. This can be:
          - "binary" for binary classification,
          - "fine_grained" for fine-grained classification,
          - "ranknet" for pairwise ranking training,
          - "bge" for models like Sentence Transformers using Bi-encoder architectures,
          - "sbert" for SBERT‑style scoring.
        Default is "binary".
    """
    def __init__(self, model, tokenizer, head_type="binary"):
        self.model = model
        self.tokenizer = tokenizer
        self.head_type = head_type
        # put model in eval mode
        # check if model has attribute 'eval'
        if hasattr(self.model, 'eval'):
            self.model.eval()

    def score(self, query, doc, max_length=512):
        """
        Score one or more query–document pairs.
        
        Parameters
        ----------
        query : str or list of str
            The query text or list of queries.
        doc : str or list of str
            The document text or list of documents.
            When passing lists, the lengths must match.
        max_length : int, optional
            Maximum sequence length for tokenization. Default is 512.
        
        Returns
        -------
        float or list of float
            A single numerical score or a list of scores for each query–document pair.
        """
        # Ensure that query and doc are lists.
        if isinstance(query, str):
            query = [query]
        if isinstance(doc, str):
            doc = [doc]
        
        # Check that both lists have the same length.
        if len(query) != len(doc):
            raise ValueError("The list of queries and the list of documents must have the same length.")
        
        scores = []
        # Use different evaluation patterns based on the head type.
        if self.head_type in ["binary", "fine_grained", "ranknet"]:
            # Tokenize all pairs at once.
            inputs = self.tokenizer(
                query, doc, truncation=True, padding="max_length",
                max_length=max_length, return_tensors="pt"
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits  # shape: [batch_size, num_labels]
            
            if self.head_type == "fine_grained":
                # For each example, use softmax then weighted average by class indices.
                probs = torch.softmax(logits, dim=-1)
                indices = torch.arange(logits.size(-1), device=logits.device, dtype=torch.float)
                scores = (probs * indices).sum(dim=-1).tolist()
            elif self.head_type == "binary":
                # If two classes, return positive class probability. Otherwise, return raw score.
                if logits.size(-1) == 2:
                    scores = torch.softmax(logits, dim=-1)[:, 1].tolist()
                else:
                    scores = logits.squeeze(-1).tolist()  # Ensure proper shape.
            elif self.head_type == "ranknet":
                # For RankNet, return the raw logit (assumed to be at index 0).
                scores = logits[:, 0].tolist()
        
        elif self.head_type == "bge":
            # For BGE-style models, assume model.compute_score supports batches of pairs.
            # logging.disable(logging.CRITICAL)
            with silence_stdout():
                query_doc_pairs = [[f"Caso legal con el siguiente tema: {q}", d] for q, d in zip(query, doc)]
                scores = self.model.compute_score(query_doc_pairs)
            # logging.disable(logging.NOTSET)
        
        elif self.head_type == "sbert":
            # Assuming model.predict supports batch input.
            scores = self.model.predict(list(zip(query, doc)))
        
        else:
            # Default: if no special head type, tokenize and then take argmax.
            inputs = self.tokenizer(
                query, doc, truncation=True, padding="max_length",
                max_length=max_length, return_tensors="pt"
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits
            scores = logits.argmax(dim=-1).tolist()
        
        # If only one pair was provided, return a single score.
        if len(scores) == 1:
            return scores[0]
        return scores
