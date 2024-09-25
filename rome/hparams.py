from dataclasses import dataclass

from .compute_v_delta import RomeComputeVDeltaHparams


@dataclass(kw_only=True)
class RomeHparams:
    # Method
    context_template_length_params: list[list[int]]

    # Module templates
    layer: int
    rewrite_module_tmp: str

    # v delta
    v_delta: RomeComputeVDeltaHparams

    # Statistics
    mom2_adjustment: bool
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
