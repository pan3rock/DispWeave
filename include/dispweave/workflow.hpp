#pragma once
#include <filesystem>
namespace dw {
int run_predictive(const std::filesystem::path& run,const std::filesystem::path& output,int per_chain,unsigned long long seed);
int run_hmc(const std::filesystem::path& config,const std::filesystem::path& output,
            const std::filesystem::path& data,const std::filesystem::path& covariance={});
// Synthetic: model is truth; invert: model is fixed-property reference.
int run_inversion(const std::filesystem::path& config,const std::filesystem::path& output,
                  bool synthetic,const std::filesystem::path& data={},const std::filesystem::path& covariance={});
}
