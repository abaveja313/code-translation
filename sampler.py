import ctypes
import math

import llama_cpp
from loguru import logger
from prefect import task
from prefect.tasks import task_input_hash
import prefect.runtime


def load_model(*, model_path: str, gpu_layers: int = 20):
    logger.info("Initializing Llama Backend")
    llama_cpp.llama_backend_init(numa=False)
    params = llama_cpp.llama_model_default_params()
    params.n_gpu_layers = gpu_layers
    model = llama_cpp.llama_load_model_from_file(
        bytes(model_path, encoding="utf-8"), params
    )
    if not model:
        raise Exception(f"Failed to load model {model_path}")
    logger.info(f"Loaded Model from {model_path}")
    return model


def cleanup_model(model) -> None:
    logger.warning("Releasing Model Memory")
    llama_cpp.llama_free_model(model)
    logger.warning("Releasing Llama backend memory")
    llama_cpp.llama_backend_free()


def format_prompt(python_code: str) -> str:
    return (
        "[INST] Your task is to convert Python to functionally equivalent Java, obeying by the following "
        "constraints. (1) The Java code should be your only output, (2) it must be between the [JAVA] and "
        "[/JAVA] tags, (3) it should contain all necessary imports, (4) should be inside a class named"
        " Solution with a main(string[] args) method.\n"
        f"[PYTHON]\n{python_code}\b[/PYTHON][/INST]"
    )


def get_context_with_model(
    *,
    model,
    context_size: int,
    num_samples: int,
    seed: int,
    num_threads_per_batch: int,
):
    llama_context = llama_cpp.llama_context_default_params()
    llama_context.seed = seed
    llama_context.n_ctx = context_size
    llama_context.batch_size = max(context_size, num_samples)
    llama_context.n_threads_batch = num_threads_per_batch
    return llama_cpp.llama_new_context_with_model(model, llama_context)


def estimate_max_length(prompt_tokens: int):
    # todo test
    return math.ceil(2.75 * prompt_tokens)


def get_tokenized_prompt(model, python_code: str, prompt_context_size: int):
    prompt = format_prompt(python_code)
    logger.debug(f"Prompt: {prompt}")

    tokens = (llama_cpp.llama_token * prompt_context_size)()
    tokens_len = llama_cpp.llama_tokenize(
        model, str.encode(prompt), len(prompt), tokens, len(tokens), True, True
    )
    return tokens, tokens_len


def configure_batch(tokens_used: int, num_samples: int, tokens, context):
    batch = llama_cpp.llama_batch_init(max(tokens_used, num_samples), 0, 1)
    batch.n_tokens = tokens_used
    for i in range(tokens_used):
        batch.token[i] = tokens[i]
        batch.pos[i] = i
        batch.seq_id[i][0] = 0
        batch.n_seq_id[i] = 1
        batch.logits[i] = False

    batch.logits[batch.n_tokens - 1] = True

    if llama_cpp.llama_decode(context, batch) != 0:
        logger.error("Error decoding batch")

    # Configure the cache for each batch
    for j in range(num_samples):
        llama_cpp.llama_kv_cache_seq_cp(context, 0, j, 0, batch.n_tokens)

    return batch


def calculate_key_value_cache(num_tokens_used: int, max_tokens: int, num_samples: int):
    return num_samples + (max_tokens - num_tokens_used) * num_samples


def decode_token_to_string(model, token_id):
    buf = (ctypes.c_char * 32)()
    outlen = llama_cpp.llama_token_to_piece(model, token_id, buf, len(buf))
    return bytes(buf[:outlen]).decode("utf-8")


def update_batch(batch, new_token_id, n_cur, sample_index):
    batch.token[batch.n_tokens] = new_token_id
    batch.pos[batch.n_tokens] = n_cur
    batch.seq_id[batch.n_tokens][0] = sample_index
    batch.n_seq_id[batch.n_tokens] = 1
    batch.logits[batch.n_tokens] = True


def generate_streams(
    model,
    context,
    batch,
    num_samples: int,
    max_tokens: int,
    top_k: int,
    top_p: float,
    temp: float,
    batch_id: int,
):
    streams = [""] * num_samples
    batch_indices = [batch.n_tokens - 1] * num_samples
    n_cur = batch.n_tokens
    n_decode = 0

    while n_cur <= max_tokens:
        batch.n_tokens = 0
        for i in range(num_samples):
            if batch_indices[i] < 0:
                continue

            logits = llama_cpp.llama_get_logits_ith(context, batch_indices[i])
            n_vocab = llama_cpp.llama_n_vocab(model)

            candidates = (llama_cpp.llama_token_data * n_vocab)()
            for token_id in range(n_vocab):
                candidates[token_id].id = token_id
                candidates[token_id].logit = logits[token_id]
                candidates[token_id].p = 0.0

            candidates_p = llama_cpp.llama_token_data_array(candidates, len(candidates), False)
            llama_cpp.llama_sample_top_k(context, ctypes.byref(candidates_p), top_k, 1)
            llama_cpp.llama_sample_top_p(context, ctypes.byref(candidates_p), top_p, 1)
            llama_cpp.llama_sample_temp(context, ctypes.byref(candidates_p), temp)
            new_token_id = llama_cpp.llama_sample_token(context, ctypes.byref(candidates_p))

            if (
                new_token_id == llama_cpp.llama_token_eos(context)
                or n_cur >= max_tokens
            ):
                logger.info(
                    f"Stream {batch_id} completed generating with {n_cur} tokens"
                )
                logger.debug(f"Stream 0: {streams[0]}")
                batch_indices[i] = -1
                continue

            streams[i] += decode_token_to_string(model, new_token_id)
            update_batch(batch, new_token_id, n_cur, i)
            batch_indices[i] = batch.n_tokens
            batch.n_tokens += 1
            n_decode += 1

        if batch.n_tokens == 0:
            break

        n_cur += 1
        if llama_cpp.llama_decode(context, batch) != 0:
            logger.error("Error decoding")
            break

    llama_cpp.llama_batch_free(batch)
    return streams


# @task(
#     name="process_batch",
#     task_run_name="process-batch-{experiment_id}-{submission_id}-{batch_id}",
#     cache_key_fn=task_input_hash,
# )
def process_batch(
    *,
    model_path: str,
    gpu_layers: int,
    python_code: str,
    prompt_context_size: int,
    seed: int,
    experiment_id: str,
    submission_id: str,  # for caching
    batch_id: int,
    top_k: int,
    threads: int,
    top_p: float,
    temp: float,
    num_samples: int,
):
    logger.info(
        f"Processing batch {batch_id} for experiment {experiment_id}, submission {submission_id}"
    )
    model = load_model(model_path=model_path, gpu_layers=gpu_layers)

    tokens, num_tokens_used = get_tokenized_prompt(
        model=model, python_code=python_code, prompt_context_size=prompt_context_size
    )
    max_tokens = estimate_max_length(num_tokens_used)
    logger.info(
        f"Prompt Size: {num_tokens_used}, estimated max tokens to be {max_tokens}"
    )
    key_value_cache_requested = calculate_key_value_cache(
        num_tokens_used, max_tokens, num_samples
    )
    context = get_context_with_model(
        model=model,
        context_size=key_value_cache_requested,
        num_samples=num_samples,
        num_threads_per_batch=threads,
        seed=seed,
    )
    batch = configure_batch(
        tokens_used=num_tokens_used,
        num_samples=num_samples,
        tokens=tokens,
        context=context,
    )

    streams = generate_streams(
        model, context, batch, num_samples, max_tokens, top_k, top_p, temp, batch_id
    )
    cleanup_model(model)
    return streams


def sample(
    *,
    model_path: str,
    experiment_id: str,
    submission_id: str,
    python_code: str,
    gpu_layers: int,
    num_threads_batch: int,
    num_samples: int,
    top_p: float,
    top_k: int,
    temp: float,
    batch_size: int,
    prompt_context_size: int,
    seed: int,
):
    logger.info(f"Processing Experiment {experiment_id}, submission {submission_id}")

    assert (
        num_samples % batch_size == 0
    ), f"num_samples={num_samples} must be multiple of batch_size={batch_size}"

    samples = []

    for batch_id in range(num_samples // batch_size):
        samples.append(
            process_batch(
                experiment_id=experiment_id,
                submission_id=submission_id,
                batch_id=batch_id,
                model_path=model_path,
                gpu_layers=gpu_layers,
                python_code=python_code,
                prompt_context_size=prompt_context_size,
                seed=seed,
                threads=num_threads_batch,
                top_k=top_k,
                top_p=top_p,
                temp=temp,
                num_samples=batch_size,
            )
        )

    return samples


if __name__ == "__main__":
    process_batch(
        model_path="codellama-13b-inst.gguf",
        gpu_layers=30,
        python_code="print(\"Hello World\")",
        prompt_context_size=800,
        seed=42,
        experiment_id="dummy",
        submission_id="dummy",
        batch_id=1,
        top_k=10,
        threads=3,
        top_p=0.93,
        temp=0.4,
        num_samples=10,
    )