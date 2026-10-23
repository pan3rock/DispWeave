// Standalone invLM physics regression, shared with the invLM repository.
// No external examples, Python packages or optimization libraries are needed.
#include "rayleigh_clustered_fixture.hpp"
#include "dispersion.hpp"
#include "dunkin_derivative.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition,const char* message) {
    if(!condition) throw std::runtime_error(message);
}
void near(double actual,double expected,double tolerance) {
    if(!std::isfinite(actual) || std::abs(actual-expected)>tolerance) {
        std::cerr<<"actual="<<actual<<" expected="<<expected<<" tolerance="<<tolerance<<'\n';
        throw std::runtime_error("Clustered Rayleigh root/derivative mismatch");
    }
}
}

int main(int argc,char** argv) { try {
    std::cout.precision(17);
    const auto original=rayleigh_clustered_fixture();
    if(argc==2 && std::string(argv[1])=="--dump-fixture") {
        auto model=original;model(15,3)*=std::exp(1e-5);
        std::cout<<model<<'\n';return 0;
    }
    constexpr double frequency=.392;
    // Independent P/SV boundary matching (rayleigh_boundary_reference.py).
    // The global search used to skip modes 3/4
    // after a tiny positive perturbation, returning 4.17018 and 4.31777 instead.
    const double expected[]={2.767925769291073,3.592121649112553,3.753187475904078,
                             3.937656639506221,3.937716231795848};
    // FAST_SCALAR has a differently scaled secular function and remains a
    // known failing diagnostic for this fixture. It is not the default backend.
    require(argc==1 || (argc==2 && std::string(argv[1])=="--fast-scalar"),"Usage: rayleigh_clustered_regression [--fast-scalar|--dump-fixture]");
    const auto method=argc==2?fwd::DispMethod::FAST_SCALAR:fwd::DispMethod::DUNKIN;
    double worst_fd=0;
    {
        for(double perturbation:{-1e-4,-1e-5,0.,1e-5,1e-4}) {
            auto a=original;a(15,3)*=std::exp(perturbation);
            fwd::Dispersion dispersion(a,false,method);
            const auto roots=dispersion.search(frequency,5);
            require(roots.size()==5 && !dispersion.search_diagnostics().numerical_issue,"Missing clustered roots");
            require(roots[3]>3.93763 && roots[3]<3.93769,"Lost lower member of pair");
            require(roots[4]>3.93770 && roots[4]<3.93774,"Lost upper member of pair");
            for(int j=0;j<5;++j) {
                if(j) require(roots[j]>roots[j-1],"Unordered roots");
                if(perturbation==1e-5) near(roots[j],expected[j],2e-9);
                bool fallback=false;
                near(dispersion.search_mode(frequency,j,&fallback),roots[j],2e-10);
                require(!fallback,"Substitute phase returned");
            }
            // Requesting more modes must not change the earlier modal labels.
            const auto more=dispersion.search(frequency,7);
            require(more.size()>=5,"Lost requested prefix");
            for(int j=0;j<5;++j) near(more[j],roots[j],2e-10);
        }
        // At the old discontinuity, test derivatives of both near-degenerate
        // branches with Vp/density fixed, including perturbations of each guide.
        auto a=original;a(15,3)*=std::exp(1e-5);
        fwd::Dispersion dispersion(a,false,method);
        const auto roots=dispersion.search(frequency,5);
        VelocityModel model;model.nl=a.rows();model.depth=a.col(1);model.rho=a.col(2);
        model.vs=a.col(3);model.vp=a.col(4);
        for(int mode:{3,4}) {
            const auto kernel=fwd::dunkin_derivatives(model,2*std::acos(-1.)*frequency,roots[mode]);
            // h=1e-5 is already nonlinear for the strongly coupled guides:
            // shrinking it by ten reduces the mode-3 error from 3.25e-5 to
            // 4.64e-7. At 1e-7 the default root tolerance limits FD accuracy.
            for(int layer:{0,15,16,29,43}) for(double h:{2e-6,1e-6}) {
                auto plus=a,minus=a;plus(layer,3)*=std::exp(h);minus(layer,3)*=std::exp(-h);
                fwd::Dispersion p(plus,false,method),m(minus,false,method);
                const double fd=(p.search_mode(frequency,mode)-m.search_mode(frequency,mode))/(2*h);
                const double derivative=kernel.dc_dvs(layer)*a(layer,3);
                if(std::abs(fd-derivative)>2e-6+2e-4*std::abs(derivative))
                    std::cerr<<"method="<<int(method)<<" mode="<<mode<<" layer="<<layer<<" h="<<h<<'\n';
                near(fd,derivative,2e-6+2e-4*std::abs(derivative));
                worst_fd=std::max(worst_fd,std::abs(fd-derivative));
            }
        }
    }
    // The candidate-generation change is shared with Love; retain its existing
    // oscillation-count certification, including a pair below the dedup scale.
    Eigen::ArrayXXd guides(6,5);
    guides<<0,0,2,.55,1.1, 1,.008,2,.12,.24, 2,.012,2,.55,1.1,
            3,.028,2,.12,.24, 4,.032,2,.55,1.1, 5,.040,2,.6,1.2;
    for(auto kind:{fwd::DispMethod::DUNKIN,fwd::DispMethod::FAST_SCALAR}) {
        fwd::Dispersion love(guides,true,kind);
        const auto roots=love.search(21,5);
        require(roots.size()==5 && !love.search_diagnostics().numerical_issue,"Love counted prefix changed");
        near(roots[0],.1663396080577632971,1e-11);
        near(roots[1],.1663396655464466935,1e-11);
        require(roots[1]>roots[0] && roots[1]-roots[0]<1e-7,"Close Love modes merged");
    }
    std::cout<<"PASS clustered Rayleigh roots, requested prefixes, log-Vs derivatives and counted Love pair; max FD error="<<worst_fd<<'\n';
    return 0;
} catch(const std::exception& e) {std::cerr<<"FAIL "<<e.what()<<'\n';return 1;} }
