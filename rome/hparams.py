from dataclasses import dataclass

from rome.compute_v_delta import RomeComputeVDeltaHparams


@dataclass(kw_only=True)
class ROMEHyperParams:
    # v delta
    v_delta: RomeComputeVDeltaHparams

    # Method
    context_template_length_params: list[list[int]]

    # Module templates
    layer: int
    rewrite_module_tmp: str

    # Statistics
    mom2_adjustment: bool
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
