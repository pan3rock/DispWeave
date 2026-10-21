#include "dispweave/forward.hpp"
#include "models.hpp"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
void check(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
Eigen::VectorXd phases(const dw::ForwardResult& f) {check(f.all_valid(),"FD roots unavailable");Eigen::VectorXd c(f.rows.size());for(size_t i=0;i<f.rows.size();++i)c[i]=f.rows[i].phase_velocity;return c;}
void derivatives() {
    dw::InvLMForward backend;
    for(const auto& item:std::vector<std::pair<std::string,dw::LayeredModel>>{{"normal",normal()},{"low_velocity",low_velocity()},{"dense",dense()}}) {
        const auto& [name,m]=item;
        const double f=name=="normal"?.5:(name=="low_velocity"?19.70981:100.);
        const std::vector<dw::DispersionQuery> queries= name=="dense"?std::vector<dw::DispersionQuery>{{f,0},{f,5},{f,6}}:std::vector<dw::DispersionQuery>{{f,0},{f,1},{f,2}};
        const auto energy=backend.evaluate(m,queries,dw::SensitivityMethod::energy);
        const auto implicit=backend.evaluate(m,queries,dw::SensitivityMethod::implicit);
        check(energy.all_sensitivities_valid()&&implicit.all_sensitivities_valid(),"Base derivatives unavailable");
        for(double relative_step:{1e-3,1e-4,1e-5}) {
            double absolute=0,relative=0,cross=0;
            for(int parameter=0;parameter<3;++parameter)for(int layer=0;layer<m.vs.size();++layer) {
                auto plus=m,minus=m;
                auto& a=parameter==0?plus.vs:(parameter==1?plus.vp:plus.rho);
                auto& b=parameter==0?minus.vs:(parameter==1?minus.vp:minus.rho);
                const double h=relative_step*a[layer];a[layer]+=h;b[layer]-=h;
                const Eigen::VectorXd fd=(phases(backend.evaluate(plus,queries))-phases(backend.evaluate(minus,queries)))/(2*h);
                for(size_t row=0;row<queries.size();++row) {
                    const auto& e=energy.rows[row].sensitivity;const auto& i=implicit.rows[row].sensitivity;
                    const auto& ev=parameter==0?e.dc_dvs:(parameter==1?e.dc_dvp:e.dc_drho);
                    const auto& iv=parameter==0?i.dc_dvs:(parameter==1?i.dc_dvp:i.dc_drho);
                    absolute=std::max(absolute,std::abs(fd[row]-iv[layer]));
                    relative=std::max(relative,std::abs(fd[row]-iv[layer])/(1+std::abs(iv[layer])));
                    cross=std::max(cross,std::abs(ev[layer]-iv[layer]));
                }
            }
            std::cout<<name<<" step="<<relative_step<<" fd_abs="<<absolute<<" scaled="<<relative<<" energy_implicit="<<cross<<'\n';
            check(cross<2e-5,"Energy/implicit disagreement");
            if(relative_step==1e-5)check(relative<3e-4,"Fine central differences disagree");
        }
    }
}
void halfspace() {
    for(double ratio:{1.2,1.73,3.0}) {
        const dw::LayeredModel m{v({0}),v({.6}),v({.6*ratio}),v({2.})};
        const double k=1/(ratio*ratio);
        auto polynomial=[&](double x){return x*x*x-8*x*x+(24-16*k)*x-16*(1-k);};
        double lo=0,hi=1;
        for(int it=0;it<80;++it) {double mid=(lo+hi)/2;if(polynomial(mid)>0)hi=mid;else lo=mid;}
        const double x=(lo+hi)/2,dxdk=-16*(1-x)/(3*x*x-16*x+24-16*k);
        const double dvs=std::sqrt(x)+k*dxdk/std::sqrt(x),dvp=-k*dxdk/(ratio*std::sqrt(x));
        for(auto method:{dw::SensitivityMethod::energy,dw::SensitivityMethod::implicit}) {
            const auto f=dw::InvLMForward().evaluate(m,{{1,0}},method);
            check(f.all_sensitivities_valid(),"Halfspace sensitivity unavailable");const auto& d=f.rows[0].sensitivity;
            check(std::abs(d.dc_dvs[0]-dvs)<1e-8 && std::abs(d.dc_dvp[0]-dvp)<1e-8 && std::abs(d.dc_drho[0])<1e-8,"Independent halfspace derivatives");
        }
    }
}
void diagnostics() {
    dw::InvLMForward backend;
    const auto m=normal();bool missing=false,valid=false;
    for(double f:{.05,.1,.12,.14,.16,.18,.2,.5}) {
        const auto r=backend.evaluate(m,{{f,1}},dw::SensitivityMethod::implicit);const auto& row=r.rows[0];
        std::cout<<"cutoff f="<<f<<" root="<<dw::to_string(row.status)<<" derivative="<<static_cast<int>(row.sensitivity.status)<<" halfspace_gap="<<(m.vs.tail(1)[0]-row.phase_velocity)<<" correction="<<row.sensitivity.root_correction<<'\n';
        if(row.status==dw::RootStatus::valid) {valid=true;check(std::isfinite(row.phase_velocity),"Root erased");}
        else {missing=true;check(row.sensitivity.status==dw::SensitivityStatus::unavailable,"Missing root gradient");check(!row.sensitivity.dc_dvs.allFinite(),"Missing root pseudozero gradient");}
    }
    check(missing&&valid,"Cutoff coverage");
    for(auto method:{dw::SensitivityMethod::energy,dw::SensitivityMethod::implicit}) {
        const auto wrong=dw::rayleigh_sensitivity(m,.5,.4,method);
        check(wrong.status==dw::SensitivityStatus::unavailable,"Off-root kernel should fail");
        check(!wrong.dc_dvs.allFinite()&&!wrong.message.empty(),"Failure diagnosis");
    }
    // Locate the observed root-presence transition, not a certified physical cutoff.
    double lo=.12,hi=.14;
    for(int it=0;it<32;++it) {
        const double mid=(lo+hi)/2;
        if(backend.evaluate(m,{{mid,1}}).all_valid())hi=mid;else lo=mid;
    }
    for(double offset:{1e-6,1e-4}) {
        const double frequency=hi*(1+offset);
        const auto result=backend.evaluate(m,{{frequency,1}},dw::SensitivityMethod::implicit);
        const auto& row=result.rows[0];
        const auto energy=backend.evaluate(m,{{frequency,1}},dw::SensitivityMethod::energy);
        std::cout<<"near_transition f="<<frequency<<" gap="<<m.vs.tail(1)[0]-row.phase_velocity
                 <<" implicit="<<dw::to_string(row.sensitivity.status)<<" energy="<<dw::to_string(energy.rows[0].sensitivity.status)
                 <<" root_correction="<<row.sensitivity.root_correction<<'\n';
        check(result.all_valid(),"Transition root lost");
        for(double step:{1e-4,1e-6,1e-8}) {
            auto plus=m,minus=m;plus.vs[2]+=step;minus.vs[2]-=step;
            const auto a=backend.evaluate(plus,{{frequency,1}}),b=backend.evaluate(minus,{{frequency,1}});
            const bool stable=a.all_valid()&&b.all_valid()&&result.all_sensitivities_valid();
            const double error=stable?std::abs((a.rows[0].phase_velocity-b.rows[0].phase_velocity)/(2*step)-row.sensitivity.dc_dvs[2]):NAN;
            std::cout<<"  step="<<step<<" both_sides="<<stable<<" fd_error="<<error<<'\n';
        }
    }
    const auto root_only=backend.evaluate(m,{{.5,0}});
    check(root_only.rows[0].sensitivity.status==dw::SensitivityStatus::not_requested,"Optional sensitivity");
    const auto dense_result=backend.evaluate(dense(),{{100,5},{100,6}},dw::SensitivityMethod::implicit);
    check(dense_result.all_valid(),"Close branches roots");
    for(const auto& row:dense_result.rows)std::cout<<"close phase="<<row.phase_velocity<<" sensitivity="<<static_cast<int>(row.sensitivity.status)<<" correction="<<row.sensitivity.root_correction<<'\n';
}
int main(int argc,char** argv) {
    try {std::cout<<std::setprecision(9);if(argc!=2)throw std::invalid_argument("Test name");
        if(std::string(argv[1])=="derivatives")derivatives();else if(std::string(argv[1])=="halfspace")halfspace();else if(std::string(argv[1])=="diagnostics")diagnostics();else throw std::invalid_argument("Test name");return 0;}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
