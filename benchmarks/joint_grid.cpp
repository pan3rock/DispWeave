// Validation only: physical-Vs grid for independent marginal posterior integration.
// Emits chi-squared for every observation block at every node, including failures.
#include "dispweave/joint.hpp"
#include <toml.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
namespace {
Eigen::VectorXd vector(const toml::value& t,const char* k) {
    const auto a=toml::find<std::vector<double>>(t,k);
    return Eigen::Map<const Eigen::VectorXd>(a.data(),a.size());
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=5) throw std::invalid_argument("joint_grid config output n0 n1");
        const int n0=std::stoi(argv[3]),n1=std::stoi(argv[4]);
        if(n0<3 || n1<3 || std::filesystem::exists(argv[2])) throw std::invalid_argument("Need new output and >=3 grid points per axis");
        const auto doc=toml::parse(argv[1]),m=toml::find(doc,"model"),p=toml::find(doc,"parameters");
        dw::LayeredModel model{vector(m,"depth"),vector(m,"vs"),vector(m,"vp"),vector(m,"rho")};model.validate();
        const auto lower=vector(p,"lower_vs"),upper=vector(p,"upper_vs");
        if(model.vs.size()!=2 || lower.size()!=2 || upper.size()!=2 || toml::find_or<double>(p,"vp_vs_ratio",0)!=0)
            throw std::invalid_argument("This validation grid requires two Vs and fixed Vp");
        dw::InvLMForward backend;std::vector<dw::ObservationBlock> blocks;
        for(const auto& b:toml::find<std::vector<toml::value>>(doc,"observations")) {
            const auto name=toml::find<std::string>(b,"name"),kind=toml::find<std::string>(b,"observable");
            dw::GaussianData data;data.values=vector(b,"values");
            if(b.contains("sigma")) data.covariance=vector(b,"sigma").array().square().matrix().asDiagonal();
            else {
                const auto c=toml::find<std::vector<std::vector<double>>>(b,"covariance");
                data.covariance.resize(c.size(),c.size());
                for(size_t i=0;i<c.size();++i) {if(c[i].size()!=c.size()) throw std::invalid_argument("Ragged covariance");
                    for(size_t j=0;j<c.size();++j) data.covariance(i,j)=c[i][j];}
            }
            if(kind=="p_receiver_function") {
                const auto r=toml::find(b,"processing");dw::ReceiverOptions o;
                o.samples=toml::find<int>(r,"samples");o.dt=toml::find<double>(r,"dt");
                o.pre_arrival=toml::find<double>(r,"pre_arrival");o.time_shift=toml::find<double>(r,"time_shift");
                o.slowness_s_km=toml::find<double>(r,"slowness_s_km");o.gaussian_width=toml::find<double>(r,"gaussian_width");
                o.water_level=toml::find<double>(r,"water_level");
                if(r.contains("source")) o.source=vector(r,"source");
                blocks.push_back(dw::receiver_block(name,o,toml::find<std::vector<int>>(b,"sample_indices"),data));
            } else {
                const auto f=vector(b,"frequency_hz");const auto modes=toml::find<std::vector<int>>(b,"mode");
                if(modes.size()!=static_cast<size_t>(f.size())) throw std::invalid_argument("Invalid queries");
                std::vector<dw::DispersionQuery> q;for(Eigen::Index i=0;i<f.size();++i) q.push_back({f[i],modes[i]});
                if(kind=="rayleigh_phase_velocity") blocks.push_back(dw::dispersion_block(name,backend,{q,data.values,data.covariance}));
                else blocks.push_back(dw::ellipticity_block(name,kind,q,data));
            }
        }
        dw::JointLikelihood likelihood(blocks,dw::BlockDependence::conditional_independence);
        std::ofstream out(argv[2]);out.exceptions(std::ios::badbit|std::ios::failbit);out<<std::setprecision(17)<<"# vs0 vs1 status";
        for(const auto& b:blocks) out<<' '<<b.name;out<<'\n';int failures=0;
        for(int i=0;i<n0;++i) {
            model.vs[0]=lower[0]+(upper[0]-lower[0])*i/(n0-1);
            for(int j=0;j<n1;++j) {
                model.vs[1]=lower[1]+(upper[1]-lower[1])*j/(n1-1);
                const auto e=likelihood.evaluate(model);const bool valid=e.status==dw::EvaluationStatus::valid;
                out<<model.vs.transpose()<<' '<<(valid?"valid":"failed");
                for(const auto& b:e.blocks) out<<' '<<b.chi_squared;out<<'\n';failures+=!valid;
            }
            if(i%20==0) std::cout<<"grid row "<<i<<'/'<<n0<<std::endl;
        }
        std::cout<<"grid_nodes="<<n0*n1<<" failures="<<failures<<'\n';return failures?2:0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
