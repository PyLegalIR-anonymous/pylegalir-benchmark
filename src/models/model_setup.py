def get_bge_m3_model(checkpoint):
    """ Load BAAI embeddings model."""
    from FlagEmbedding import BGEM3FlagModel
    print("Loading BAAI embeddings model from checkpoint:", checkpoint)
    # model = BGEM3FlagModel(checkpoint, use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    model = BGEM3FlagModel(checkpoint) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    return model


def get_jinja_model():
    """ Load Jinja embeddings model."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
    return model


def get_mamba_model():
    """ Load Mamba embeddings model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # load model from
    path = "/home/leon/tesis/mamba-ir/results_300M_diverse_shuffle_75train/mamba-130m-spanish-legal-300M-tokens-diverse"
    model = AutoModelForCausalLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer


def get_auto_model(checkpoint):
    # Load model directly
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
    return model, tokenizer