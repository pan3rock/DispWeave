#include "dispweave/forward.hpp"
#include "independent_reference.hpp"
#include "models.hpp"
#include <iostream>
#include <stdexcept>
void check(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
int main() { try {
    const std::vector<Eigen::VectorXd> profiles{v({1,1.2,1.45,1.8,2.3,3}),v({1,1,1,2.2,2.2,3}),
                                             v({1.1,1.8,1.1,1.8,2.4,3}),v({1,1.8,1.2,1.65,2.2,3})};
    const std::vector<dw::DispersionQuery> queries{{.12,0},{.3,1},{.6,2}};
    dw::InvLMForward backend; double root_error=0, fd_error=0;
    for(const auto& vs:profiles) {
        dw::LayeredModel model{v({0,.5,1,2,3,4}),vs,Eigen::VectorXd::Constant(6,6),Eigen::VectorXd::Constant(6,2.5)};
        const auto base=backend.evaluate(model,queries,dw::SensitivityMethod::energy);
        check(base.all_sensitivities_valid(),"Study kernels unavailable");
        for(size_t i=0;i<queries.size();++i) {
            const double c=base.rows[i].phase_velocity;
            const double reference=reference_root(model,queries[i].frequency_hz,c*(1-1e-5),c*(1+1e-5));
            root_error=std::max(root_error,std::abs(c-reference));
            check(std::abs(c-reference)<1e-7,"Study independent root disagrees");
        }
        for(double step:{1e-4,1e-5}) for(int k=0;k<6;++k) {
            auto plus=model,minus=model; const double h=step*vs[k]; plus.vs[k]+=h; minus.vs[k]-=h;
            auto a=backend.evaluate(plus,queries),b=backend.evaluate(minus,queries);
            check(a.all_valid() && b.all_valid(),"Study finite difference branch unavailable");
            for(size_t i=0;i<queries.size();++i) {
                const double kernel=base.rows[i].sensitivity.dc_dvs[k];
                const double error=std::abs((a.rows[i].phase_velocity-b.rows[i].phase_velocity)/(2*h)-kernel)/(1+std::abs(kernel));
                fd_error=std::max(fd_error,error);
                check(error<1e-4,"Study finite difference disagreement");
            }
        }
    }
    std::cout<<"study independent root max_error="<<root_error<<" Vs FD max_scaled_error="<<fd_error<<'\n';
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; } }
