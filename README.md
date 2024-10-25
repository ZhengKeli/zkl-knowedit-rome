# ZKL-ROME

This project is a reimplementation of [ROME](https://github.com/kmeng01/rome) with better engineering.

ROME (Rank-One Model Editing) is a method for editing large language models (LLMs).
It is described in paper [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)
and implemented in GitHub repository [https://github.com/kmeng01/rome](https://github.com/kmeng01/rome).

However, the original implementation code is totally research-oriented.
With a lot of non-necessary code, hard-coded behaviors, inconvenient interfaces,
the original code base is not suitable for production use.

In this project, the ROME is reimplemented with better engineering.
Thanks to our better design, our implementation offers a range of practical [features](#features).

Yet, it is important to note that this implementation is not fully equivalent to the original one,
due to the differences in some details. If this is a concern, please refer [below](#note-on-equivalence).

## Getting Started

This project is currently not published on PyPI yet.
You can install it via url.

```shell
pip install git+https://gitee.com/zhengkeli/zkl-knowedit-rome
```

An example of simple usage can be found in [scripts/edit_simple.py](scripts/edit_simple.py).
You can just run the following command to inspect the effect.

```shell
python scripts/edit_simple.py
```

An example of detailed usage can be found in [scripts/edit_detailed.py](scripts/edit_detailed.py).
You can just run the following command to inspect the effect.

```shell
python scripts/edit_detailed.py
```

Please notice that the above scripts require additional development dependency `datasets`,
which is not installed by default via pip installation.
Since this project is managed by poetry,
you can conveniently set up the development environment with the following command:

```shell
poetry install
```

## Features

### Token Level Editing

The original ROME is totally text-based, making it impossible to operate accurately on tokens.

With this new implementation, we can perform editing directly with sequence of tokens.

This can benefit formated-text tasks, multimodal tasks, etc.

### Customizable Prefixes

In the original ROME, the prefixes generation is hard-coded as "wikipedia" or "wikidata".

With this new implementation, we can specify any list of token strings as prefixes.

This is more favorable for models that are not trained just on Wikipedia.

### Customizable Preserving

In the original ROME, a kl_loss is computed to ensure locality of the editing.
However, it is hard-coded with template `"{subject} is a"`.

With this new implementation, we can specify any list of token strings as preserving prefixes.

### Flexible Application

In the original ROME, the computed rank-one patch is applied to a specified parameter of a `Linear` module.

More generally, we can also apply it to modules with different types,
as long as it can be treated as a linear mapping.

In this implementation, we implemented `apply_left_right_to_module()`,
which can apply the rank-one patch to any module with correct input/output shapes.

### Faster and More Efficient

In the original ROME, there are many redundant operations,
such as repetitive tokenization, recomputing k,v when computing the right vector.

In this new implementation, we eliminated these redundant operations,
making the whole process more efficient.

### Installable as Python Package

The original ROME is a project even without a `requirements.txt` file,
which is inconvenient for other projects to use as a dependency.

With this new implementation, we can install it directly as a Python package,
with all dependencies resolved automatically.

## Note on Equivalence

When reimplementing it, we tried to change the algorithm as little as possible.
We accepted some differences for efficiency and engineering reasons,
which leads to the fact that they are not fully equivalent.

Nevertheless, we've been keeping track of them and focused on how big the difference is.
We kept track of the cosine similarity of key intermediates of the algorithm.

There is results of the comparison:

```text
prompt="Steve Jobs is the founder of"
subject="Steve Jobs"
target=" Microsoft"

c_sim=0.9489483833312988
c_inv_sim=0.7663295269012451
left_sim=0.8766450881958008
right_sim=0.9692411422729492
w_delta_sim=0.8496803641319275
```

With `w_delta_sim=0.85` we can say that this implementation is "very close" to the original one.

The comparison is performed by [scripts/compare.py](scripts/compare.py).
If it concerns, you can run your own comparison with other configurations.
