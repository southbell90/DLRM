#include <ATen/cuda/CUDAGraph.h>
#include <memory>

#include <torch/torch.h>
#include <torch/cuda.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

using torch::indexing::Slice;

torch::nn::Sequential make_mlp(
    const std::vector<int64_t>& layer_sizes,
    bool use_output_activation = false) {
    torch::nn::Sequential seq;
    const int64_t n = static_cast<int64_t>(layer_sizes.size());
    for (int64_t i = 0; i < n - 1; ++i) {
        const int64_t in_f = layer_sizes[i];
        const int64_t out_f = layer_sizes[i + 1];
        seq->push_back(torch::nn::Linear(in_f, out_f));
        const bool is_last = (i == n - 2);
        if (is_last && use_output_activation) {
        seq->push_back(torch::nn::Sigmoid());
        } else {
        seq->push_back(torch::nn::ReLU());
        }
    }
    return seq;
}

template <int D>
__global__ void embed_interact_kernel(
    const float* __restrict__ emb_w,    // [total_rows, D]
    const int64_t* __restrict__ offsets,// [n_sparse]
    const int64_t* __restrict__ sparse, // [B, n_sparse]
    const int64_t* __restrict__ pair_i, // [n_int]
    const int64_t* __restrict__ pair_j, // [n_int]
    const float* __restrict__ z0,       // [B, D]
    float* __restrict__ top_input,      // [B, D + n_int]
    int64_t B,
    int64_t n_sparse,
    int64_t n_int) {

    extern __shared__ float shmem[];
    float* z0_sh  = shmem;                     // D
    float* emb_sh = z0_sh + D;                 // n_sparse * D

    const int b = blockIdx.x;
    if (b >= B) return;

    // z0 로드
    const float* z0_row = z0 + b * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        z0_sh[d] = z0_row[d];
    }

    // embedding gather
    const int64_t* sparse_row = sparse + b * n_sparse;
    for (int f = threadIdx.x; f < n_sparse; f += blockDim.x) {
        int64_t local = sparse_row[f];
        int64_t idx   = offsets[f] + local;
        const float* src = emb_w + idx * D;
        float*       dst = emb_sh + f * D;

    #pragma unroll
        for (int d = 0; d < D; ++d) {
        dst[d] = src[d];
        }
    }

    __syncthreads();

    // z0를 output 앞부분에 써주기
    float* out_row = top_input + b * (D + n_int);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        out_row[d] = z0_sh[d];
    }

    // 모든 (i,j) pair에 대해 dot product
    for (int pair = threadIdx.x; pair < n_int; pair += blockDim.x) {
        int i = static_cast<int>(pair_i[pair]);
        int j = static_cast<int>(pair_j[pair]);

        const float* vi = (i == 0) ? z0_sh : (emb_sh + (i - 1) * D);
        const float* vj = (j == 0) ? z0_sh : (emb_sh + (j - 1) * D);

        float acc = 0.0f;
    #pragma unroll
        for (int d = 0; d < D; ++d) {
        acc += vi[d] * vj[d];
        }

        out_row[D + pair] = acc;
    }
}

torch::Tensor embed_interact_cuda(
    const torch::Tensor& z0,          // [B, D]
    const torch::Tensor& sparse_x,    // [B, n_sparse]
    const torch::Tensor& emb_weight,  // fused_embedding_->weight
    const torch::Tensor& offsets,
    const torch::Tensor& pair_i,
    const torch::Tensor& pair_j) {

    const auto B        = z0.size(0);
    const auto D        = z0.size(1);
    const auto n_sparse = sparse_x.size(1);
    const auto n_int    = pair_i.size(0);

    auto out = torch::empty({B, D + n_int}, z0.options());

    dim3 grid(B);
    int threads = 128;
    size_t shmem_bytes = (D + n_sparse * D) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();

    embed_interact_kernel<16><<<grid, threads, shmem_bytes, stream>>>(
            emb_weight.data_ptr<float>(),
            offsets.data_ptr<int64_t>(),
            sparse_x.data_ptr<int64_t>(),
            pair_i.data_ptr<int64_t>(),
            pair_j.data_ptr<int64_t>(),
            z0.data_ptr<float>(),
            out.data_ptr<float>(),
            B, n_sparse, n_int);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "embed_interact_kernel launch failed");
    return out;
}


struct DLRMOptimizedImpl : torch::nn::Module {
  DLRMOptimizedImpl(
      int64_t num_dense_features,
      const std::vector<int64_t>& sparse_feature_sizes,
      int64_t embedding_dim,
      const std::vector<int64_t>& bottom_mlp_sizes,
      const std::vector<int64_t>& top_mlp_sizes)
      : num_dense_features_(num_dense_features),
        sparse_feature_sizes_(sparse_feature_sizes),
        embedding_dim_(embedding_dim) {
    // --- fused embedding 설정 ---
    n_sparse_ = static_cast<int64_t>(sparse_feature_sizes_.size());

    std::vector<int64_t> offsets_vec(n_sparse_);
    int64_t prefix = 0;
    for (int64_t i = 0; i < n_sparse_; ++i) {
      offsets_vec[i] = prefix;
      prefix += sparse_feature_sizes_[i];
    }
    total_rows_ = prefix;

    offsets_ = torch::from_blob(
                   offsets_vec.data(),
                   {n_sparse_},
                   torch::dtype(torch::kLong))
                   .clone();  
    register_buffer("offsets", offsets_);

    fused_embedding_ = torch::nn::Embedding(
        torch::nn::EmbeddingOptions(total_rows_, embedding_dim_));
    register_module("fused_embedding", fused_embedding_);


    std::vector<int64_t> bottom_layers;
    bottom_layers.reserve(bottom_mlp_sizes.size() + 1);
    bottom_layers.push_back(num_dense_features_);
    bottom_layers.insert(
        bottom_layers.end(), bottom_mlp_sizes.begin(), bottom_mlp_sizes.end());
    if (bottom_layers.back() != embedding_dim_) {
      throw std::runtime_error(
          "Last bottom MLP size must equal embedding_dim for interactions.");
    }
    bottom_mlp_ = make_mlp(bottom_layers, /*use_output_activation=*/false);
    register_module("bottom_mlp", bottom_mlp_);


    const int64_t n_f = n_sparse_ + 1;
    n_int_ = n_f * (n_f - 1) / 2;

    std::vector<int64_t> pair_i_vec;
    std::vector<int64_t> pair_j_vec;
    pair_i_vec.reserve(n_int_);
    pair_j_vec.reserve(n_int_);
    for (int64_t i = 0; i < n_f; ++i) {
      for (int64_t j = i + 1; j < n_f; ++j) {
        pair_i_vec.push_back(i);
        pair_j_vec.push_back(j);
      }
    }

    pair_i_ = torch::from_blob(
                  pair_i_vec.data(),
                  {n_int_},
                  torch::dtype(torch::kLong))
                  .clone();
    pair_j_ = torch::from_blob(
                  pair_j_vec.data(),
                  {n_int_},
                  torch::dtype(torch::kLong))
                  .clone();
    register_buffer("pair_i", pair_i_);
    register_buffer("pair_j", pair_j_);


    std::vector<int64_t> top_layers;
    top_layers.reserve(top_mlp_sizes.size() + 1);
    top_layers.push_back(embedding_dim_ + n_int_);
    top_layers.insert(
        top_layers.end(), top_mlp_sizes.begin(), top_mlp_sizes.end());
    top_mlp_ = make_mlp(top_layers, true);
    register_module("top_mlp", top_mlp_);
  }

  torch::Tensor forward(const torch::Tensor& dense_x,
                        const torch::Tensor& sparse_x) {
    auto z0 = bottom_mlp_->forward(dense_x); // [B, D]

    torch::Tensor top_input;
    auto emb_weight = fused_embedding_->weight;
    top_input = embed_interact_cuda(
        z0,              // bottom MLP output
        sparse_x,        // [B, n_sparse]
        emb_weight,
        offsets_,
        pair_i_, pair_j_);
    

    auto logits = top_mlp_->forward(top_input);


    return logits;
  }

  torch::nn::Sequential& bottom_mlp() { return bottom_mlp_; }
  torch::nn::Sequential& top_mlp() { return top_mlp_; }

 private:
    int64_t num_dense_features_;
    std::vector<int64_t> sparse_feature_sizes_;
    int64_t embedding_dim_;
    int64_t n_sparse_{0};
    int64_t n_int_{0};
    int64_t total_rows_{0};

    // fused embedding
    torch::nn::Embedding fused_embedding_{nullptr};
    torch::Tensor offsets_;  

    // interaction index pairs
    torch::Tensor pair_i_;  
    torch::Tensor pair_j_;  

    torch::nn::Sequential bottom_mlp_{nullptr};
    torch::nn::Sequential top_mlp_{nullptr};
};
TORCH_MODULE(DLRMOptimized);

std::pair<void*, size_t> collect_mlp_address_range(
    torch::nn::Sequential& mlp1,
    torch::nn::Sequential& mlp2) {
    uintptr_t min_addr = std::numeric_limits<uintptr_t>::max();
    uintptr_t max_addr = 0;

    auto visit_mlp = [&](torch::nn::Sequential& mlp) {
        for (const auto& child : mlp->children()) {
        // torch::nn::LinearImpl 로 다운캐스트
        auto linear =
            std::dynamic_pointer_cast<torch::nn::LinearImpl>(child);
        if (!linear) {
            continue;
        }

        // weight
        auto weight = linear->weight;
        if (weight.defined()) {
            auto addr =
                reinterpret_cast<uintptr_t>(weight.data_ptr());
            size_t bytes = weight.numel() * weight.element_size();
            min_addr = std::min(min_addr, addr);
            max_addr = std::max(max_addr, addr + bytes);
        }

        // bias
        auto bias = linear->bias;
        if (bias.defined()) {
            auto addr =
                reinterpret_cast<uintptr_t>(bias.data_ptr());
            size_t bytes = bias.numel() * bias.element_size();
            min_addr = std::min(min_addr, addr);
            max_addr = std::max(max_addr, addr + bytes);
        }
        }
    };

    visit_mlp(mlp1);
    visit_mlp(mlp2);

    if (max_addr <= min_addr) {
        return {nullptr, 0};
    }
    return {reinterpret_cast<void*>(min_addr),
            max_addr - min_addr};
}


void setup_l2_persisting_for_mlp(void* base_ptr, size_t bytes) {
    if (!base_ptr || bytes == 0) return;

    int device_id = 0;
    cudaGetDevice(&device_id);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device_id);

    if (prop.persistingL2CacheMaxSize == 0) {
    return;
    }

    size_t window_bytes =
        std::min(bytes, static_cast<size_t>(prop.accessPolicyMaxWindowSize));
    size_t max_set_aside = static_cast<size_t>(prop.persistingL2CacheMaxSize);
    size_t set_aside_bytes = std::min(window_bytes, max_set_aside);

    // L2 set-aside 영역 크기 설정
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, set_aside_bytes);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.base_ptr = base_ptr;
    attr.accessPolicyWindow.num_bytes = window_bytes;
    attr.accessPolicyWindow.hitRatio =
        static_cast<float>(window_bytes) / static_cast<float>(bytes);
    if (attr.accessPolicyWindow.hitRatio > 1.0f)
    attr.accessPolicyWindow.hitRatio = 1.0f;

    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

    cudaStreamSetAttribute(stream,
                            cudaStreamAttributeAccessPolicyWindow,
                            &attr);
}


int main() {
    const int64_t num_dense_features = 13;
    const std::vector<int64_t> sparse_feature_sizes = {
        1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683,
        8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4, 7046547, 18,
        15, 286181, 105, 142572};
    const int64_t embedding_dim = 16;

    const std::vector<int64_t> bottom_mlp_sizes = {512, 256, 64, 16};
    const std::vector<int64_t> top_mlp_sizes = {512, 256, 1};

    const bool cuda_available = torch::cuda::is_available();
    auto device = cuda_available ? torch::kCUDA : torch::kCPU;
    std::cout << "Using device: " << (cuda_available ? "CUDA" : "CPU") << "\n";

    DLRMOptimized model(
        num_dense_features,
        sparse_feature_sizes,
        embedding_dim,
        bottom_mlp_sizes,
        top_mlp_sizes);
    model->to(device);
    model->eval();

    if (cuda_available) {
        auto range = collect_mlp_address_range(
            model->bottom_mlp(), model->top_mlp());
        setup_l2_persisting_for_mlp(range.first, range.second);
    }

    const int64_t n_sparse = static_cast<int64_t>(sparse_feature_sizes.size());

    const int64_t batch_size = 32768;
    const int64_t num_samples = 6'000'000;
    const int64_t num_batches = num_samples / batch_size;

    auto dense_opts = torch::TensorOptions()
                            .device(device)
                            .dtype(torch::kFloat);
    auto idx_opts = torch::TensorOptions()
                        .device(device)
                        .dtype(torch::kLong);

    torch::Tensor dense_static =
        torch::empty({batch_size, num_dense_features}, dense_opts);
    torch::Tensor sparse_static =
        torch::empty({batch_size, n_sparse}, idx_opts);

    torch::Tensor out_static;

    if (cuda_available) {
        torch::cuda::synchronize();
    }


    std::cout << "Warming up and capturing CUDA graph...\n";

    at::cuda::CUDAStream capture_stream = at::cuda::getStreamFromPool();
    at::cuda::setCurrentCUDAStream(capture_stream);

    // warmup용 입력
    dense_static.normal_(0.0, 1.0);
    for (int64_t i = 0; i < n_sparse; ++i) {
        auto col = sparse_static.index({Slice(), i});
        col.random_(0, sparse_feature_sizes[i]);
    }

    // warmup
    out_static = model->forward(dense_static, sparse_static).squeeze(1);
    torch::cuda::synchronize();

    // Graph capture
    at::cuda::CUDAGraph graph;
    graph.capture_begin();
    out_static = model->forward(dense_static, sparse_static).squeeze(1);
    graph.capture_end();


    auto start = std::chrono::steady_clock::now();

    for (int64_t b = 0; b < num_batches; ++b) {
        dense_static.normal_(0.0, 1.0);
        for (int64_t i = 0; i < n_sparse; ++i) {
            auto col = sparse_static.index({Slice(), i});
            col.random_(0, sparse_feature_sizes[i]);
        }

        graph.replay();

    }

    torch::cuda::synchronize();
    auto end = std::chrono::steady_clock::now();
    double elapsed =
        std::chrono::duration<double>(end - start).count();

    std::cout << "CUDA Graph inference time: " << elapsed
                << " s for " << num_batches
                << " batches (batch_size=" << batch_size
                << ", total_samples=" << (num_batches * batch_size)
                << ").\n";

  return 0;
}

