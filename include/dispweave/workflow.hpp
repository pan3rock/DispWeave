#pragma once
#include <filesystem>
namespace dw {
// Synthetic: model is truth; invert: model is fixed-property reference.
int run_inversion(const std::filesystem::path& config,const std::filesystem::path& output,
                  bool synthetic,const std::filesystem::path& data={},const std::filesystem::path& covariance={},
                  const std::filesystem::path& validation_data={},const std::filesystem::path& validation_covariance={});
}
