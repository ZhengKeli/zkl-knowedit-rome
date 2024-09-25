from dataclasses import dataclass

from .compute_v_delta import RomeComputeVDeltaHparams


@dataclass(kw_only=True)
class RomeHparams:
    rewrite_module_name: str

    # prefixes
    context_template_length_params: list[list[int]]

    # k
    mom2_adjustment: bool
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str

    # v delta
    v_delta: RomeComputeVDeltaHparams
