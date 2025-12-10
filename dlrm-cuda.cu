#include <torch/torch.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h> 

namespace F = torch::nn::functional;
using torch::indexing::Slice;

torch::nn::Sequential make_bottom_mlp(
    const std::vector<int64_t>& layer_sizes,
    torch::nn::AnyModule output_activation = {}) {
    torch::nn::Sequential seq;
    const auto n = static_cast<int64_t>(layer_sizes.size());
    for (int64_t i = 0; i < n - 1; ++i) {
        const int64_t in_f = layer_sizes[i];
        const int64_t out_f = layer_sizes[i + 1];
        seq->push_back(torch::nn::Linear(in_f, out_f));
        const bool is_last = (i == n - 2);
        if (is_last && output_activation.is_empty() == false) {
            seq->push_back(output_activation);
        } else {
            seq->push_back(torch::nn::ReLU());
        }
    }
    return seq;
}

struct DLRMImpl : torch::nn::Module {
    DLRMImpl(
        int64_t num_dense_features,
        const std::vector<int64_t>& sparse_feature_sizes,
        int64_t embedding_dim,
        const std::vector<int64_t>& bottom_mlp_sizes,
        const std::vector<int64_t>& top_mlp_sizes)
        : num_dense_features_(num_dense_features),
          sparse_feature_sizes_(sparse_feature_sizes),
          embedding_dim_(embedding_dim) {
        
        embeddings_ = register_module("embeddings", torch::nn::ModuleList());
        for (auto n : sparse_feature_sizes_) {
            embeddings_->push_back(torch::nn::Embedding(
                torch::nn::EmbeddingOptions(n, embedding_dim_)));
        }

        std::vector<int64_t> bottom_layers;
        bottom_layers.reserve(bottom_mlp_sizes.size() + 1);
        bottom_layers.push_back(num_dense_features_);
        bottom_layers.insert(
            bottom_layers.end(), bottom_mlp_sizes.begin(), bottom_mlp_sizes.end());

        if (bottom_layers.back() != embedding_dim_) {
            throw std::runtime_error("Last bottom MLP size must equal embedding_dim.");
        }

        bottom_mlp_ = make_bottom_mlp(bottom_layers);
        register_module("bottom_mlp", bottom_mlp_);

        n_sparse_ = static_cast<int64_t>(sparse_feature_sizes_.size());
        const int64_t n_f = n_sparse_ + 1;
        n_int_ = n_f * (n_f - 1) / 2;

        std::vector<int64_t> top_layers;
        top_layers.reserve(top_mlp_sizes.size() + 1);
        top_layers.push_back(embedding_dim_ + n_int_);
        top_layers.insert(top_layers.end(), top_mlp_sizes.begin(), top_mlp_sizes.end());

        top_mlp_ = make_bottom_mlp(top_layers);
        register_module("top_mlp", top_mlp_);
    }

    torch::Tensor forward(const torch::Tensor& dense_x, const torch::Tensor& sparse_x) {
        
        auto z0 = bottom_mlp_->forward(dense_x);

        std::vector<torch::Tensor> emb_list;
        emb_list.reserve(n_sparse_);
        for (int64_t i = 0; i < n_sparse_; ++i) {
            auto idx = sparse_x.index({Slice(), i});
            emb_list.push_back(embeddings_[i]->as<torch::nn::Embedding>()->forward(idx));
        }

        std::vector<torch::Tensor> stack_inputs;
        stack_inputs.reserve(1 + emb_list.size());
        stack_inputs.push_back(z0);
        stack_inputs.insert(stack_inputs.end(), emb_list.begin(), emb_list.end());

        auto z = torch::stack(stack_inputs, 1); // [B, n_f, D]
        auto zz = torch::bmm(z, z.transpose(1, 2)); // [B, n_f, n_f]

        auto idx = torch::triu_indices(z.size(1), z.size(1), 1, 
                                     torch::TensorOptions()
                                         .device(z.device())
                                         .dtype(torch::kLong));
        
        auto interactions = zz.index({Slice(), idx[0], idx[1]});
        auto top_input = torch::cat({z0, interactions}, 1);
        auto logits = top_mlp_->forward(top_input);

        return logits;
    }

    torch::nn::Sequential get_bottom_mlp() { return bottom_mlp_; }
    torch::nn::Sequential get_top_mlp() { return top_mlp_; }
    torch::nn::ModuleList get_embeddings() { return embeddings_; }

private:
    int64_t num_dense_features_;
    std::vector<int64_t> sparse_feature_sizes_;
    int64_t embedding_dim_;
    int64_t n_sparse_;
    int64_t n_int_;

    torch::nn::ModuleList embeddings_{nullptr};
    torch::nn::Sequential bottom_mlp_{nullptr};
    torch::nn::Sequential top_mlp_{nullptr};
};
TORCH_MODULE(DLRM);


void pin_params_to_l2_cache(DLRM& model) {
    if (!torch::cuda::is_available()) return;

    std::vector<torch::Tensor> target_params;
    int64_t pinned_embedding_count = 0;
    
    auto collect_mlp = [&](torch::nn::Sequential seq) {
        for (auto& param : seq->parameters()) target_params.push_back(param);
    };
    collect_mlp(model->get_bottom_mlp());
    collect_mlp(model->get_top_mlp());

    auto embeddings = model->get_embeddings();
    for (int i = 0; i < embeddings->size(); ++i) {
        auto emb_module = embeddings[i]->as<torch::nn::Embedding>();
        if (emb_module->weight.size(0) <= 100000) {
            target_params.push_back(emb_module->weight);
            pinned_embedding_count++;
        }
    }

    if (target_params.empty()) return;

    int64_t total_elements = 0;
    for (const auto& p : target_params) total_elements += p.numel();

    auto options = target_params[0].options(); 
    torch::Tensor big_buffer = torch::empty({total_elements}, options);

    {
        torch::NoGradGuard no_grad;
        int64_t offset = 0;
        for (auto& p : target_params) {
            int64_t numel = p.numel();
            auto flattened = p.view({-1});
            auto buffer_slice = big_buffer.slice(0, offset, offset + numel);
            buffer_slice.copy_(flattened);
            p.set_data(buffer_slice.view(p.sizes()));
            offset += numel;
        }
    }

    size_t size_bytes = big_buffer.nbytes();
    void* d_ptr = big_buffer.data_ptr();

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    size_t l2_limit = prop.persistingL2CacheMaxSize;
    
    // 안전하게 한도의 90%까지만 사용하도록 설정
    size_t set_limit = std::min(size_bytes * 2, l2_limit);
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, set_limit);

    cudaAccessPolicyWindow window;
    window.base_ptr = d_ptr;
    window.num_bytes = size_bytes;
    window.hitRatio = 1.0f; 
    window.hitProp = cudaAccessPropertyPersisting;
    window.missProp = cudaAccessPropertyStreaming;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaStreamAttrValue attr;
    attr.accessPolicyWindow = window;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);

    std::cout << "  - Pinned Size: " << (size_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  - Pinned Embeddings: " << pinned_embedding_count << std::endl;
}


int main() {
    const int64_t num_dense_features = 13;
    const std::vector<int64_t> sparse_feature_sizes = {
        1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 
        5683, 8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4, 
        7046547, 18, 15, 286181, 105, 142572};
    const int64_t embedding_dim = 16;
    const std::vector<int64_t> bottom_mlp_sizes = {512, 256, 64, 16};
    const std::vector<int64_t> top_mlp_sizes = {512, 256, 1};

    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is required for this optimization." << std::endl;
        return -1;
    }
    auto device = torch::kCUDA;
    std::cout << "Using device: CUDA (RTX 5090 Optimized)" << "\n";

    DLRM model(num_dense_features, sparse_feature_sizes, embedding_dim, bottom_mlp_sizes, top_mlp_sizes);
    model->to(device);
    model->eval();

    torch::NoGradGuard no_grad;

    pin_params_to_l2_cache(model);

    const int64_t batch_size = 32; 
    const int64_t num_samples = 1'000'000;
    const int64_t num_batches = num_samples / batch_size;
    
    auto static_dense_x = torch::randn({batch_size, num_dense_features}, 
        torch::TensorOptions().device(device).dtype(torch::kFloat));
    
    std::vector<torch::Tensor> sparse_cols;
    for (auto n : sparse_feature_sizes) {
        sparse_cols.push_back(torch::randint(0, n, {batch_size}, 
            torch::TensorOptions().device(device).dtype(torch::kLong)));
    }
    auto static_sparse_x = torch::stack(torch::TensorList(sparse_cols), 1);
    
    // 결과 저장용 텐서 (Graph 출력용)
    torch::Tensor static_output;

    std::cout << "Capturing CUDA Graph..." << std::endl;
    
    for(int i=0; i<3; ++i) {
        model->forward(static_dense_x, static_sparse_x);
    }
    torch::cuda::synchronize();

    at::cuda::CUDAGraph graph;

    auto stream = at::cuda::getStreamFromPool();
    {
        c10::cuda::CUDAStreamGuard guard(stream);
        
        graph.capture_begin();
        
        static_output = model->forward(static_dense_x, static_sparse_x).squeeze(1);
        
        graph.capture_end();
    }
    torch::cuda::synchronize();
    std::cout << "Graph captured successfully." << std::endl;

    std::cout << "Starting Inference Loop..." << std::endl;


    auto host_dense = torch::randn({batch_size, num_dense_features}, torch::kFloat);
    std::vector<torch::Tensor> host_sparse_vec;
    for (auto n : sparse_feature_sizes) {
        host_sparse_vec.push_back(torch::randint(0, n, {batch_size}, torch::kLong));
    }
    auto host_sparse = torch::stack(torch::TensorList(host_sparse_vec), 1);

    if (torch::cuda::is_available()) torch::cuda::synchronize();
    
    auto start = std::chrono::steady_clock::now();

    for (int64_t b = 0; b < num_batches; ++b) {
        static_dense_x.copy_(host_dense, /*non_blocking=*/true);
        static_sparse_x.copy_(host_sparse, /*non_blocking=*/true);
        
        graph.replay();
    }

    if (torch::cuda::is_available()) torch::cuda::synchronize();
    
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Inference time: " << elapsed << " s" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    return 0;
}
