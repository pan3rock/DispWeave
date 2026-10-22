#include "dispweave/workflow.hpp"
#include "dispweave/receiver.hpp"
#include "version.hpp"
#include <toml.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
namespace dw {
namespace {
Eigen::VectorXd values(const toml::value& table,const char* name) {
    const auto x=toml::find<std::vector<double>>(table,name);
    return Eigen::Map<const Eigen::VectorXd>(x.data(),x.size());
}
std::ofstream file(const std::filesystem::path& path) {
    std::ofstream f(path);if(!f) throw std::runtime_error("Cannot write "+path.string());
    f.exceptions(std::ios::badbit|std::ios::failbit);f<<std::setprecision(17);return f;
}
}
int run_receiver(const std::filesystem::path& config,const std::filesystem::path& output) {
    if(std::filesystem::exists(output) && (!std::filesystem::is_directory(output) || !std::filesystem::is_empty(output)))
        throw std::invalid_argument("Receiver output directory must be new or empty");
    const auto doc=toml::parse(config.string());const auto m=toml::find(doc,"model"),r=toml::find(doc,"receiver");
    if(toml::find_or<std::string>(r,"incident","P")!="P") throw std::invalid_argument("Only incident P receiver functions are supported");
    if(toml::find_or<std::string>(r,"deconvolution","waterlevel")!="waterlevel") throw std::invalid_argument("Only waterlevel receiver deconvolution is supported");
    LayeredModel model{values(m,"depth"),values(m,"vs"),values(m,"vp"),values(m,"rho")};
    ReceiverOptions options;
    options.slowness_s_km=toml::find<double>(r,"slowness_s_km");
    options.samples=toml::find_or<int>(r,"samples",options.samples);
    options.dt=toml::find_or<double>(r,"dt",options.dt);
    options.gaussian_width=toml::find_or<double>(r,"gaussian_width",options.gaussian_width);
    options.water_level=toml::find_or<double>(r,"water_level",options.water_level);
    options.pre_arrival=toml::find_or<double>(r,"pre_arrival",options.pre_arrival);
    options.time_shift=toml::find_or<double>(r,"time_shift",options.time_shift);
    if(r.contains("source")) options.source=values(r,"source");
    const auto result=receiver_function(model,options);
    std::filesystem::create_directories(output);std::filesystem::copy_file(config,output/"config.toml");
    auto wave=file(output/"receiver.txt");wave<<"# time_s radial_receiver_function status\n";
    for(Eigen::Index i=0;i<result.time.size();++i) wave<<result.time[i]<<' '<<result.radial_rf[i]<<' '<<to_string(result.status)<<'\n';
    auto spectrum=file(output/"spectrum.txt");
    spectrum<<"# frequency_hz status R_real R_imag Z_up_real Z_up_imag rcond backward_error source_real source_imag radial_transfer_real radial_transfer_imag vertical_transfer_real vertical_transfer_imag message\n";
    for(size_t k=0;k<result.spectrum.size();++k) {
        const auto& row=result.spectrum[k];
        spectrum<<result.frequency_hz[k]<<' '<<to_string(row.status)<<' '<<row.radial.real()<<' '<<row.radial.imag()<<' '
            <<row.vertical.real()<<' '<<row.vertical.imag()<<' '<<row.reciprocal_condition<<' '<<row.backward_error<<' '
            <<result.source_spectrum[k].real()<<' '<<result.source_spectrum[k].imag()<<' '
            <<result.radial_transfer[k].real()<<' '<<result.radial_transfer[k].imag()<<' '
            <<result.vertical_transfer[k].real()<<' '<<result.vertical_transfer[k].imag()<<' '<<std::quoted(row.message)<<'\n';
    }
    auto metadata=file(output/"metadata.txt");
    metadata<<"version="<<DW_VERSION<<"\ncompiler="<<DW_COMPILER<<"\nbuild="<<DW_BUILD_TYPE
        <<"\nbackend=elastic_isotropic_PSV_boundary_system\nincident=P_upgoing_unit_displacement_at_halfspace_top"
        <<"\ncoordinates=radial_along_horizontal_propagation; vertical_up\nspectral_convention=exp(-i*omega*t)"
        <<"\ndeconvolution=waterlevel_R_conjZ/max(absZ_squared,water_level*max_absZ_squared)"
        <<"\nfilter=exp(-omega_squared/(4*gaussian_width_squared))\nnormalization=vertical_self_deconvolution_at_zero_lag"
        <<"\nnormalization_value="<<result.normalization<<"\nslowness_s_km="<<options.slowness_s_km
        <<"\nsamples="<<options.samples<<"\ndt="<<options.dt<<"\npre_arrival="<<options.pre_arrival<<"\ntime_shift="<<options.time_shift
        <<"\ngaussian_width="<<options.gaussian_width<<"\nwater_level="<<options.water_level
        <<"\nsource="<<(options.source.size()?"explicit_time_samples":"unit_impulse")
        <<"\nwindow=finite_periodic_DFT; convergence_requires_window_and_sampling_checks"
        <<"\nstatus="<<to_string(result.status)<<"\nmessage="<<result.message<<'\n';
    std::cout<<"receiver_samples="<<result.time.size()<<" status="<<to_string(result.status)<<" output="<<output<<'\n';
    return result.status==ReceiverStatus::valid?0:2;
}
}
