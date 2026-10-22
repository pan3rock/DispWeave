#include "dispweave/ellipticity.hpp"
#include "models.hpp"
#include "independent_reference.hpp"
#include <iostream>
void require(bool x,const char* msg) {if(!x) throw std::runtime_error(msg);}
// Independent traction-free P/SV null vector at the surface.
double independent_ratio(const dw::LayeredModel& m,double f,double c) {
    const double omega=2*std::acos(-1.)*f,k=omega/c;
    const int last=static_cast<int>(m.vs.size())-1;
    Eigen::Matrix<Complex,4,2> state=basis(m.vp[last],m.vs[last],m.rho[last],omega,k).field.leftCols<2>();
    for(int i=last-1;i>=0;--i) {
        const auto b=basis(m.vp[i],m.vs[i],m.rho[i],omega,k);
        Eigen::Matrix<Complex,4,2> a=b.field.fullPivLu().solve(state);
        for(int j=0;j<4;++j) a.row(j)*=std::exp(-b.vertical[j]*(m.depth[i+1]-m.depth[i]));
        state=b.field*a;
        for(int j=0;j<2;++j) state.col(j)/=state.col(j).norm();
    }
    int t=state.row(2).norm()>state.row(3).norm()?2:3;
    Eigen::Matrix<Complex,2,1> a;a<<-state(t,1),state(t,0);
    const Eigen::Matrix<Complex,4,1> surface=state*a;
    require(surface.bottomRows<2>().norm()/surface.norm()<1e-6,"Reference surface traction");
    return std::abs(surface[0])/std::abs(surface[1]);
}
int main() {try {
    for(double ratio:{1.2,1.73,3.}) {
        dw::LayeredModel m{v({0}),v({2}),v({2*ratio}),v({2.3})};
        const auto rows=dw::rayleigh_ellipticity(m,{{.01,0},{1,0},{100,0},{1,1}});
        for(int i=0;i<3;++i) {
            require(rows[i].status==dw::EllipticityStatus::valid,"Halfspace eigenfunction failed");
            const double x=rows[i].root.phase_velocity/2,p=std::sqrt(1-x*x/(ratio*ratio)),s=std::sqrt(1-x*x);
            const double exact=std::abs((1+s*s-2*p*s)/(p*(1-s*s)));
            require(std::abs(rows[i].ratio-exact)<1e-8,"Analytic halfspace ellipticity mismatch");
        }
        require(rows[3].status==dw::EllipticityStatus::root_unavailable && std::isnan(rows[3].ratio),"Missing mode hidden");
    }
    const auto m=normal();
    // Refinement must preserve mode identity and the existing standard path.
    const std::vector<dw::DispersionQuery> prefix={{.5,0},{.5,1},{.5,2},{.5,3},{.5,4}};
    const auto standard=dw::InvLMForward().evaluate(m,prefix);
    const auto refined=dw::InvLMForward(dw::SecularMethod::dunkin,dw::RootAccuracy::refined).evaluate(m,prefix);
    require(standard.all_valid() && refined.all_valid(),"Refinement lost a mode");
    for(size_t i=0;i<prefix.size();++i)
        require(std::abs(standard.rows[i].phase_velocity-refined.rows[i].phase_velocity)<2e-12,"Refinement changed mode identity");
    const auto rows=dw::rayleigh_ellipticity(m,{{.2,0},{.5,0},{.5,1},{.5,2}});
    for(size_t i=0;i<rows.size();++i) {
        require(rows[i].status==dw::EllipticityStatus::valid,"Layered eigenfunction failed");
        const double expected=independent_ratio(m,i==0?.2:.5,rows[i].root.phase_velocity);
        std::cout<<i<<" invLM="<<rows[i].ratio<<" independent="<<expected<<'\n';
        require(std::abs(rows[i].ratio-expected)<1e-6*std::max(1.,expected),"Layered ellipticity mismatch");
    }
    dw::LayeredModel split{v({0,1,2,3,4}),v({1,1,2,2,3}),v({1.8,1.8,3.6,3.6,5.2}),v({2,2,2.2,2.2,2.5})};
    const auto divided=dw::rayleigh_ellipticity(split,{{.2,0},{.5,0},{.5,1},{.5,2}});
    for(size_t i=0;i<rows.size();++i) {
        require(divided[i].status==dw::EllipticityStatus::valid,"Subdivision field failure");
        require(std::abs(divided[i].ratio-rows[i].ratio)<1e-7,"Layer subdivision changed surface ratio");
    }
    const auto low=low_velocity();
    for(double f:{5.,10.,15.}) {
        const auto values=dw::rayleigh_ellipticity(low,{{f,0},{f,1}});
        for(const auto& row:values) {
            if(row.root.status!=dw::RootStatus::valid) {
                require(row.status==dw::EllipticityStatus::root_unavailable,"Missing LVL root status");continue;
            }
            require(row.status==dw::EllipticityStatus::valid,"LVL eigenfunction failure");
            const double reference=independent_ratio(low,f,row.root.phase_velocity);
            require(std::abs(row.ratio-reference)<1e-6*std::max(1.,reference),"LVL reference mismatch");
        }
    }
    // Strong-contrast cap: vertical surface displacement crosses zero near
    // 1.0175859 Hz. A finite root tolerance must not turn the pole into a
    // seemingly reliable enormous finite ratio.
    dw::LayeredModel pole{v({0,.1}),v({.4,2.5}),v({.8,4.5}),v({2,2.5})};
    const auto at_pole=dw::rayleigh_ellipticity(pole,{{1.0175858928204988,0}});
    require(at_pole[0].root.status==dw::RootStatus::valid,"Pole should preserve the phase root");
    require(at_pole[0].status==dw::EllipticityStatus::near_pole && std::isnan(at_pole[0].ratio),"Pole silently represented by finite clipped ratio");
    for(double f:{1.0174,1.0178}) {
        const auto side=dw::rayleigh_ellipticity(pole,{{f,0}})[0];
        require(side.status==dw::EllipticityStatus::valid,"Resolvable pole side rejected");
        const double reference=independent_ratio(pole,f,side.root.phase_velocity);
        std::cout<<"pole side f="<<f<<" E="<<side.ratio<<" ref="<<reference<<" relative="<<side.ratio/reference-1<<" energy="<<side.energy_residual<<std::endl;
        require(std::abs(side.ratio/reference-1)<1e-6,"Pole-side independent ratio mismatch");
    }
    const auto zero=dw::rayleigh_ellipticity(pole,{{1.8137044387655514,0},{1.817,0},{1.8137044387655514,0}});
    require(zero[0].status==dw::EllipticityStatus::valid && zero[0].ratio<1e-10,"Horizontal node should yield a valid zero limit");
    require(zero[0].ratio==zero[2].ratio && zero[1].ratio>zero[0].ratio,"Repeated query order changed");
    require(std::abs(zero[0].ratio-independent_ratio(pole,1.8137044387655514,zero[0].root.phase_velocity))<1e-10,"Independent horizontal node mismatch");
    const auto fast=dw::rayleigh_ellipticity(pole,{{1.0174,0},{1.0178,0},{1.0175858928204988,0}},dw::SecularMethod::fast_scalar);
    for(int i=0;i<2;++i) {
        require(fast[i].status==dw::EllipticityStatus::valid,"Fast scalar pole-side failure");
        const double expected=independent_ratio(pole,i==0?1.0174:1.0178,fast[i].root.phase_velocity);
        require(std::abs(fast[i].ratio/expected-1)<1e-6,"Fast scalar ellipticity mismatch");
    }
    require(fast[2].status==dw::EllipticityStatus::near_pole,"Fast scalar pole hidden");
    std::cout<<"Ellipticity halfspace, independent layered surface fields and missing-mode checks passed\n";
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
