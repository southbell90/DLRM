// Minimal DLRM inference example in C++ (LibTorch + CUDA).
// Compile example (adjust TORCH path and ABI as needed):
// nvcc -std=c++17 dlrm-cuda.cu -o dlrm-cuda \\
//   -I${TORCH}/include -I${TORCH}/include/torch/csrc/api/include \\
//   -L${TORCH}/lib -ltorch -lc10 -lcudart -lpthread -ldl
// If CUDA is unavailable, it will fall back to CPU.

#include <torch/torch.h>

#include <chrono>
#include <iostream>
#include <vector>

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
      throw std::runtime_error(
          "Last bottom MLP size must equal embedding_dim for interactions.");
    }
    bottom_mlp_ = make_bottom_mlp(bottom_layers);
    register_module("bottom_mlp", bottom_mlp_);

    n_sparse_ = static_cast<int64_t>(sparse_feature_sizes_.size());
    const int64_t n_f = n_sparse_ + 1;
    n_int_ = n_f * (n_f - 1) / 2;

    std::vector<int64_t> top_layers;
    top_layers.reserve(top_mlp_sizes.size() + 1);
    top_layers.push_back(embedding_dim_ + n_int_);
    top_layers.insert(top_layers.end(), top_mlp_sizes.begin(),
                      top_mlp_sizes.end());
    top_mlp_ = make_bottom_mlp(top_layers);
    register_module("top_mlp", top_mlp_);
  }

  torch::Tensor forward(const torch::Tensor& dense_x,
                        const torch::Tensor& sparse_x) {
    const auto B = dense_x.size(0);
    if (sparse_x.size(1) != n_sparse_) {
      throw std::runtime_error("Unexpected number of sparse fields.");
    }

    auto z0 = bottom_mlp_->forward(dense_x);

    std::vector<torch::Tensor> emb_list;
    emb_list.reserve(n_sparse_);
    for (int64_t i = 0; i < n_sparse_; ++i) {
      auto idx = sparse_x.index({Slice(), i});
      // ModuleList stores erased types; recover the Embedding to call forward.
      emb_list.push_back(
          embeddings_[i]->as<torch::nn::Embedding>()->forward(idx));
    }

    std::vector<torch::Tensor> stack_inputs;
    stack_inputs.reserve(1 + emb_list.size());
    stack_inputs.push_back(z0);
    stack_inputs.insert(stack_inputs.end(), emb_list.begin(), emb_list.end());
    auto z = torch::stack(stack_inputs, 1);  // [B, n_f, D]

    auto zz = torch::bmm(z, z.transpose(1, 2));  // [B, n_f, n_f]
    auto idx = torch::triu_indices(z.size(1), z.size(1), /*offset=*/1);
    auto interactions = zz.index({Slice(), idx[0], idx[1]});

    auto top_input = torch::cat({z0, interactions}, 1);
    auto logits = top_mlp_->forward(top_input);
    return logits;
  }

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

  DLRM model(num_dense_features, sparse_feature_sizes, embedding_dim,
             bottom_mlp_sizes, top_mlp_sizes);
  model->to(device);
  model->eval();

  const int64_t batch_size = 32;
  const int64_t num_samples = 1000000;
  const int64_t num_batches = (num_samples + batch_size - 1) / batch_size;

  std::vector<torch::Tensor> preds;
  preds.reserve(num_batches);

  if (cuda_available) {
    torch::cuda::synchronize();
  }
  auto start = std::chrono::steady_clock::now();

  for (int64_t b = 0; b < num_batches; ++b) {
    const int64_t current_bs =
        std::min(batch_size, num_samples - b * batch_size);
    if (current_bs <= 0) {
      break;
    }

    auto dense_x =
        torch::randn({current_bs, num_dense_features},
                     torch::TensorOptions().device(device).dtype(torch::kFloat));

    std::vector<torch::Tensor> sparse_columns;
    sparse_columns.reserve(sparse_feature_sizes.size());
    for (auto n : sparse_feature_sizes) {
      sparse_columns.push_back(
          torch::randint(0, n, {current_bs},
                         torch::TensorOptions()
                             .device(device)
                             .dtype(torch::kLong)));
    }
    auto sparse_x = torch::stack(torch::TensorList(sparse_columns), 1);

    auto logits = model->forward(dense_x, sparse_x).squeeze(1);
    auto probs = logits;
    preds.push_back(probs.cpu());

    if (b < 3) {
      std::cout << "Batch " << (b + 1) << "/" << num_batches
                << " example probs: " << probs.slice(0, 0, 3) << "\n";
    }
  }

  if (cuda_available) {
    torch::cuda::synchronize();
  }
  auto end = std::chrono::steady_clock::now();
  double elapsed =
      std::chrono::duration<double>(end - start).count();

  auto all_preds = torch::cat(preds);
  std::cout << "Inference time: " << elapsed << " s for " << preds.size()
            << " batches (batch_size=" << batch_size << ").\n";
  std::cout << "Finished inference for " << all_preds.numel()
            << " samples.\n";
  return 0;
}
