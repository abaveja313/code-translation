#include <string>
#include <vector>

class Sampler {
private:
    std::string base64_decode(const std::string &in);
    std::vector<std::string> runBatch(std::string
                                      model_file,
                                      std::string b64_program,
                                      int batch_size,
                                      float temp,
                                      int top_k,
                                      float top_p,
                                      int main_gpu,
                                      int max_tokens

    );

public:
    std::vector<std::string>
    run(std::string
        model_file,
        std::string b64_program,
        int n_samples,
        int batch_size,
        float temp,
        int top_k,
        float top_p,
        int main_gpu,
        int max_tokens
    );
};