import ctypes
import math

import llama_cpp
from loguru import logger
from prefect import task
from prefect.tasks import task_input_hash


class Sampler:
    def __init__(self, *, model_path: str, gpu_layers: int):
        self.model_path = model_path
        self.gpu_layers = gpu_layers
        params = llama_cpp.llama_model_load_default_params()
        self.model = llama_cpp.llama_load_model_from_file(bytes(model_path), params)

    @staticmethod
    def format_prompt(python_code: str) -> str:
        return (
            "[INST] Your task is to convert Python to functionally equivalent Java, obeying by the following "
            "constraints. (1) The Java code should be your only output, (2) it must be between the [JAVA] and "
            "[/JAVA] tags, (3) it should contain all necessary imports, (4) should be inside a class named"
            " Solution with a main(string[] args) method.\n"
            f"[PYTHON]\n{python_code}\b[/PYTHON][/INST]"
        )

    def estimate_max_length(self, prompt_tokens: int):
        return math.ceil(1.25 * prompt_tokens)

    def get_context_with_model(
        self,
        *,
        seed: int,
        context_size: int,
        num_samples: int,
        num_threads_per_batch: int,
    ):
        llama_context = llama_cpp.llama_context_default_params()
        llama_context.seed = seed
        llama_context.n_ctx = context_size
        llama_context.batch_size = max(context_size, num_samples)
        llama_context.n_threads_batch = num_threads_per_batch
        return llama_cpp.llama_new_context_with_model(self.model, llama_context)

    def get_tokenized_prompt(self, python_code: str, prompt_context_size: int):
        prompt = self.format_prompt(python_code)

        tokens = (llama_cpp.llama_token * prompt_context_size)()
        tokens_len = llama_cpp.llama_tokenize(
            self.model, str.encode(prompt), len(prompt), tokens, len(tokens), True, True
        )
        return tokens, tokens_len

    def configure_batch(self, max_tokens: int, num_samples: int, tokens, context):
        batch = llama_cpp.llama_batch_init(max(max_tokens, num_samples), 0, 1)
        batch.n_tokens = max_tokens
        for i in range(max_tokens):
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

    @task(
        name="process_batch",
        task_run_name="process-batch-{experiment_id}-{submission_id}-{batch_id}",
        cache_key_fn=task_input_hash,
    )
    def process_batch(
        self,
        *,
        experiment_id: str,
        submission_id: str,  # for caching
        batch_id: int,
        context,
        top_k: int,
        top_p: float,
        temp: float,
        max_tokens: int,
        num_samples: int,
        tokens,
    ):
        logger.debug(f"Processing batch {batch_id} for experiment {experiment_id}, submission {submission_id}")
        batch = self.configure_batch(
            max_tokens=max_tokens,
            num_samples=num_samples,
            tokens=tokens,
            context=context,
        )
        streams = [""] * num_samples
        batch_indices = [batch.n_tokens - 1] * num_samples
        n_cur = batch.n_tokens
        n_decode = 0
        while n_cur <= max_tokens:
            batch.n_tokens = 0
            for i in range(num_samples):
                if batch_indices[i] < 0:
                    continue

                n_vocab = llama_cpp.llama_n_vocab(self.model)
                logits = llama_cpp.llama_get_logits_ith(context, batch_indices[i])
                candidates = (llama_cpp.llama_token_data * n_vocab)()

                for token_id in range(n_vocab):
                    candidates[token_id].id = token_id
                    candidates[token_id].logit = logits[token_id]
                    candidates[token_id].p = 0.0

                candidates_p = llama_cpp.llama_token_data_array(
                    candidates, len(candidates), False
                )

                llama_cpp.llama_sample_top_k(
                    context, ctypes.byref(candidates_p), top_k, 1
                )
                llama_cpp.llama_sample_top_p(
                    context, ctypes.byref(candidates_p), top_p, 1
                )
                llama_cpp.llama_sample_temp(context, ctypes.byref(candidates_p), temp)

                new_token_id = llama_cpp.llama_sample_token(
                    context, ctypes.byref(candidates_p)
                )

                if (
                    new_token_id == llama_cpp.llama_token_eos(context)
                    or n_cur >= max_tokens
                ):
                    logger.info(
                        f"Stream {batch_id} completed generating with {n_cur} tokens"
                    )
                    batch_indices[i] = -1
                    continue

                buf = (ctypes.c_char * 32)()
                outlen = llama_cpp.llama_token_to_piece(
                    self.model, new_token_id, buf, len(buf)
                )
                streams[i] += bytes(buf[:outlen]).decode("utf-8")

                batch.token[batch.n_tokens] = new_token_id
                batch.pos[batch.n_tokens] = n_cur
                batch.seq_id[batch.n_tokens][0] = i
                batch.n_seq_id[batch.n_tokens] = 1
                batch.logits[batch.n_tokens] = True

                batch_indices[i] = batch.n_tokens
                batch.n_tokens += 1
                n_decode += 1

            if batch.n_tokens == 0:
                break

            n_cur += 1

            if llama_cpp.llama_decode(context, batch) != 0:
                logger.error("Error decoding")
                break
            # todo add cleanup for memory!
            return streams

    @task(
        name="process",
        task_run_name="process-{experiment_id}-{submission_id}",
        cache_key_fn=task_input_hash,
    )
    def sample_async(
        self,
        *,
        experiment_id: str,
        submission_id: str,
        python_code: str,
        num_threads_batch: int,
        num_samples: int,
        top_p=0.95,
        top_k=15,
        temp=0.5,
        batch_size: int = 10,
        prompt_context_size: int = 1024,
        seed: int = 42,
    ):
        logger.info(f"Processing Experiment {experiment_id}, submission {submission_id}")

        assert (
            num_samples % batch_size != 0
        ), f"num_samples {num_samples} must be multiple of batch_size{batch_size}"
        tokens, num_tokens_used = self.get_tokenized_prompt(
            python_code=python_code, prompt_context_size=prompt_context_size
        )
        max_tokens: int = self.estimate_max_length(num_tokens_used)
        logger.info(
            f"Prompt Size: {num_tokens_used}, estimated max tokens to be {max_tokens}"
        )
        key_value_cache_requested: int = num_tokens_used + (
            (max_tokens - num_tokens_used) * batch_size
        )
        context = self.get_context_with_model(
            seed=seed,
            context_size=key_value_cache_requested,
            num_samples=batch_size,
            num_threads_per_batch=num_threads_batch,
        )

        n_ctx = llama_cpp.llama_n_ctx(context)
        if key_value_cache_requested > n_ctx:
            logger.error(
                f"The required KV cache size ({key_value_cache_requested}) is not big enough ({n_ctx})"
            )

        samples = []

        for batch_id in range(num_samples // batch_size):
            samples.append(
                self.process_batch.submit(
                    batch_id=batch_id,
                    context=context,
                    top_k=top_k,
                    top_p=top_p,
                    temp=temp,
                    n_ctx=n_ctx,
                    max_tokens=max_tokens,
                    num_samples=batch_size,
                    tokens=tokens,
                )
            )

        return samples
