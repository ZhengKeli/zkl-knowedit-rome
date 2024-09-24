from dataclasses import dataclass


@dataclass(kw_only=True)
class ROMEHyperParams:
    # Method
    v_num_grad_steps: int
    v_lr: float
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    context_template_length_params: list[list[int]]

    # Module templates
    layer: int
    rewrite_module_tmp: str

    # Statistics
    mom2_adjustment: bool
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
