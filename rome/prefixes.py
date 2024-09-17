from .utils.generate import generate_fast

CONTEXT_TEMPLATES_CACHE = None


def get_context_templates(model, tok, length_params):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        generated_prefixes_groups = [
            generate_fast(model, tok, ["<|endoftext|>"], n_gen_per_prompt=n_gen, max_out_len=length)
            for length, n_gen in length_params]
        generated_prefixes = [prefix for group in generated_prefixes_groups for prefix in group]

        CONTEXT_TEMPLATES_CACHE = ["{}"] + [x + ". {}" for x in generated_prefixes]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
