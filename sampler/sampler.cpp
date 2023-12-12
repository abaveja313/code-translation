#include "common.h"
#include "llama.h"
#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>


std::string Sampler::base64_decode(const std::string &in) {
    std::string out;
    std::vector<int> T(256, -1);
    for (int i = 0; i < 64; i++) T["ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[i]] = i;

    int val = 0, valb = -8;
    for (unsigned char c: in) {
        if (T[c] == -1) break;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

std::vector<std::string>
Sampler::runBatch(std::string model_file, std::string b64_program, int batch_size, float temp,
                  int top_k, float top_p, int main_gpu, int max_tokens) {
    gpt_params params;
    int n_parallel = batch_size;


    const int n_gpu_layers = 100; // Added command line argument parsing for n_gpu_layers
    params.model = model_file;

    params.main_gpu = main_gpu;

    std::string program = base64_decode(b64_program);

    params.prompt =
            "[INST] Your task is to convert Python to Java, obeying by the following constraints. The Java code should be your only output, and must be between the [JAVA] and [/JAVA] tags. The Java code should contain all necessary imports and be within a driver class called Solution, with an executable main(string[] args) method. The code should be functionally identical to the Python code.\n\n"
            "[PYTHON]\n" + program + "\n[/PYTHON]"
                                     "[/INST]";

    std::cout << params.prompt << std::endl;
    llama_backend_init(params.numa);

    // initialize the model

    llama_model_params model_params = llama_model_default_params();

    model_params.n_gpu_layers = n_gpu_layers;

    llama_model *model = llama_load_model_from_file(params.model.c_str(), model_params);

    if (model == NULL) {
        throw std::runtime_error("error: unable to load model");
    }

    // tokenize the prompt

    std::vector<llama_token> tokens_list;
    tokens_list = ::llama_tokenize(model, params.prompt, true);
    int n_len = max_tokens == -1 ? std::min(static_cast<int>(tokens_list.size() * 2.5), 1200) : max_tokens;
    std::cout << "\nMax Tokens: " << n_len << std::endl;

    const int n_kv_req = tokens_list.size() + (n_len - tokens_list.size()) * n_parallel;

    // initialize the context

    llama_context_params ctx_params = llama_context_default_params();

    // ctx_params.seed  = 1234;
    ctx_params.n_ctx = n_kv_req;
    ctx_params.n_batch = std::max(n_len, n_parallel);
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;

    llama_context *ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == NULL) {
        throw std::runtime_error("failed to create the llama_context");
    }

    const int n_ctx = llama_n_ctx(ctx);

    LOG_TEE("\n%s: n_len = %d, n_ctx = %d, n_batch = %d, n_parallel = %d, n_kv_req = %d\n", __func__, n_len, n_ctx,
            ctx_params.n_batch, n_parallel, n_kv_req);

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (n_kv_req > n_ctx) {
        LOG_TEE("%s: error: n_kv_req (%d) > n_ctx, the required KV cache size is not big enough\n", __func__,
                n_kv_req);
        LOG_TEE("%s:        either reduce n_parallel or increase n_ctx\n", __func__);
        throw std::runtime_error("check logs error");
    }

    // create a llama_batch
    // we use this object to submit token data for decoding
    llama_batch batch = llama_batch_init(std::max(tokens_list.size(), (size_t) n_parallel), 0, 1);

    // evaluate the initial prompt
    for (size_t i = 0; i < tokens_list.size(); ++i) {
        llama_batch_add(batch, tokens_list[i], i, {0}, false);
    }
    GGML_ASSERT(batch.n_tokens == (int) tokens_list.size());

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        LOG_TEE("%s: llama_decode() failed\n", __func__);
        throw std::runtime_error("llama decode failed");
    }

    // assign the system KV cache to all parallel sequences
    // this way, the parallel sequences will "reuse" the prompt tokens without having to copy them
    for (int32_t i = 1; i < n_parallel; ++i) {
        llama_kv_cache_seq_cp(ctx, 0, i, 0, batch.n_tokens);
    }

    if (n_parallel > 1) {
        LOG_TEE("\n\n%s: generating %d sequences ...\n", __func__, n_parallel);
    }

    // main loop

    // we will store the parallel decoded sequences in this vector
    std::vector<std::string> streams(n_parallel);

    // remember the batch index of the last token for each parallel sequence
    // we need this to determine which logits to sample from
    std::vector<int32_t> i_batch(n_parallel, batch.n_tokens - 1);

    int n_cur = batch.n_tokens;
    int n_decode = 0;

    const auto t_main_start = ggml_time_us();

    while (n_cur <= n_len) {
        // prepare the next batch
        llama_batch_clear(batch);

        // sample the next token for each parallel sequence / stream
        for (int32_t i = 0; i < n_parallel; ++i) {
            if (i_batch[i] < 0) {
                // the stream has already finished
                continue;
            }

            auto n_vocab = llama_n_vocab(model);
            auto *logits = llama_get_logits_ith(ctx, i_batch[i]);

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);

            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
            }

            llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

            llama_sample_top_k(ctx, &candidates_p, top_k, 1);
            llama_sample_top_p(ctx, &candidates_p, top_p, 1);
            llama_sample_temp(ctx, &candidates_p, temp);

            const llama_token new_token_id = llama_sample_token(ctx, &candidates_p);
            // std::cout << llama_token_to_piece(ctx, new_token_id).c_str() << " -> ID: " << new_token_id << std::endl;
            //const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

            // is it an end of stream? -> mark the stream as finished
            if (new_token_id == llama_token_eos(model) || n_cur == n_len) {
                // if (new_token_id == 29879 || n_cur == n_len) {  // 29879 = </s
                i_batch[i] = -1;
                LOG_TEE("\n");
                if (n_parallel > 1) {
                    LOG_TEE("%s: stream %d finished at n_cur = %d", __func__, i, n_cur);
                }

                continue;
            }

            streams[i] += llama_token_to_piece(ctx, new_token_id);
            i_batch[i] = batch.n_tokens;

            // push this new token for next evaluation
            llama_batch_add(batch, new_token_id, n_cur, {i}, true);
            n_decode += 1;
        }

        // all streams are finished
        if (batch.n_tokens == 0) {
            break;
        }

        n_cur += 1;

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            throw std::runtime_error("Error to evaluate");

        }
    }

    LOG_TEE("\n");

    if (n_parallel > 1) {
        LOG_TEE("\n");

        for (int32_t i = 0; i < n_parallel; ++i) {
            LOG_TEE("sequence %d:\n\n%s%s\n\n", i, params.prompt.c_str(), streams[i].c_str());
        }
    }

    const auto t_main_end = ggml_time_us();

    LOG_TEE("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f,
            n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    llama_print_timings(ctx);

    fprintf(stderr, "\n");

    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();
    return streams;
}

std::vector<std::string>
Sampler::run(std::string model_file, std::string b64_program,
             int batch_size, int max_tokens, float temp, int top_k,
             float top_p, int main_gpu, int n_samples) {
    std::vector<std::string> output;

    int num_batches = n_samples / batch_size;
    int remain = n_samples % batch_size;

    for (int i = 0; i < num_batches; ++i) {
        std::vector<std::string> batch = this->runBatch(model_file, b64_program, batch_size, temp, top_k,
                                                        top_p, main_gpu, max_tokens);
        output.insert(output.end(), batch.begin(), batch.end());
        std::cout << "Processed " << (i + 1) * batch_size << "/" << n_samples << " samples" << std::endl;
    }

    if (remain > 0) {
        std::vector<std::string> remaining = this->runBatch(model_file, b64_program, remain, temp, top_k,
                                                            top_p, main_gpu, max_tokens);
        output.insert(output.end(), remaining.begin(), remaining.end());
        std::cout << "Processed " << n_samples << "/" << n_samples << " samples" << std::endl;
    }

    return output;
};

int main(int argc, char **argv) {
    std::unordered_map<std::string, std::string> args;

    // Parse named command-line arguments
    for (int i = 1; i < argc; i += 2) {
        if (argv[i][0] == '-' && i + 1 < argc) {
            args[argv[i]] = argv[i + 1];
        } else {
            std::cerr << "Invalid argument format: " << argv[i] << std::endl;
            return 1;
        }
    }

    std::cout << "Parsed Command-Line Arguments:" << std::endl;
    for (const auto &arg: args) {
        std::cout << arg.first << ": " << arg.second << std::endl;
    }

    // Check for mandatory arguments
    if (args.find("-model") == args.end() || args.find("-outputfile") == args.end()) {
        std::cerr << "Usage: " << argv[0]
                  << " -model MODEL_PATH -outputfile OUTPUT_FILE_PATH [-b64 B64_PROMPT] [-samples PARALLEL] [-max-tokens LEN] [-temp TEMP] [-top-k TOP_K] [-top-p TOP_P] [-batch BATCH_SIZE]\n";
        return 1;
    }

    std::string model_file = args["-model"];
    std::string output_file = args["-outputfile"];
    std::string b64_program = args.count("-b64") ? args["-b64"] : "";
    int n_parallel = args.count("-samples") ? std::stoi(args["-samples"]) : 2;
    int n_len = args.count("-max-tokens") ? std::stoi(args["-max-tokens"]) : -1;
    int batch_size = args.count("-batch") ? std::stoi(args["-batch"]) : 20;
    float temp = args.count("-temp") ? std::stof(args["-temp"]) : 0.4f;
    int top_k = args.count("-top-k") ? std::stoi(args["-top-k"]) : 10;
    float top_p = args.count("-top-p") ? std::stof(args["-top-p"]) : 0.8f;
    int mgpu = args.count("-main-gpu") ? std::stoi(args["-main-gpu"]) : 0;

    Sampler sampler;
    std::vector<std::string> results = sampler.run(model_file, b64_program, batch_size, n_len, temp, top_k, top_p, mgpu, n_parallel);


    // Write the results to the specified output file
    std::ofstream outfile(output_file);
    if (!outfile.is_open()) {
        std::cerr << "Error: Unable to open output file at " << output_file << std::endl;
        return 1;
    }

    for (const auto &result: results) {
        outfile << result << "\n======\n";
    }

    outfile.close();
    std::cout << "Results written to " << output_file << std::endl;


    return 0;
}