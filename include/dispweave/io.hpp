#pragma once
#include "forward.hpp"
#include <filesystem>
namespace dw {
struct ForwardConfig { LayeredModel model; std::vector<DispersionQuery> queries; };
int run_resolution(const std::filesystem::path& config,const std::filesystem::path& output);
ForwardConfig read_forward_config(const std::filesystem::path&);
void write_forward_result(const std::filesystem::path&,const ForwardConfig&,const ForwardResult&);
}
