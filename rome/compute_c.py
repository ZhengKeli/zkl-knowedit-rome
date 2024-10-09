import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from .layer_stats import layer_stats
from .tok_dataset import TokenizedDataset

inv_mom2_cache = {}


def compute_c_inv(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    cache_dir: str,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    global inv_mom2_cache

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    if key not in inv_mom2_cache:
        c = compute_c(model, tokenizer, layer_name, mom2_dataset, mom2_dtype, mom2_n_samples, cache_dir)
        inv_mom2_cache[key] = torch.inverse(c).float()  # Cast back to float32

    return inv_mom2_cache[key]


def compute_c(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_dtype: str,
    mom2_n_samples: str,
    cache_dir: str
):
    raw_ds = load_dataset(
        mom2_dataset,
        dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en")[mom2_dataset],
        trust_remote_code=True)
    maxlen = model.config.n_positions
    dataset = TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    stat = layer_stats(
        model,
        layer_name,
        cache_dir,
        dataset,
        to_collect=["mom2"],
        sample_size=mom2_n_samples,
        precision=mom2_dtype,
    )
    c = stat.mom2.moment()
    return c
