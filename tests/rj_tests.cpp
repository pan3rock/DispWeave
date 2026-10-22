#include "dispweave/rj_model.hpp"
#include "dispweave/rj_proposal.hpp"
#include <Eigen/LU>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <vector>
using namespace dw;
Eigen::VectorXd v(std::initializer_list<double> x) {return Eigen::Map<const Eigen::VectorXd>(x.begin(),x.size());}
void check(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
int main() {try {
    RJPrior p;p.min_layers=1;p.max_layers=3;p.depth_limit=4;p.min_spacing=.5;
    p.vs_lower=1;p.vs_upper=3;p.layer_weights=v({1,2,3});p.validate();
    RJState one{v({}),v({2})},two{v({2}),v({2.5,1.5})},three{v({1,3}),v({1.5,2.5,2})};
    check(std::abs(std::exp(p.log_density(one))-1./12)<1e-14,"K1 normalized prior");
    check(std::abs(std::exp(p.log_density(two))-1./36)<1e-14,"K2 normalized prior");
    check(std::abs(std::exp(p.log_density(three))-.02)<1e-14,"K3 normalized prior");
    check(p.supports(two),"Low velocity layer rejected");
    // Integrate ordered-interface triangle by midpoint quadrature and analytic Vs volume.
    double mass=0;
    for(int i=0;i<1000;++i) {
        const double z1=.5+(i+.5)*2.5/1000,remaining=3.5-(z1+.5);
        RJState point{v({z1,z1+.5+.5*remaining}),v({2,2,2})};
        mass+=std::exp(p.log_density(point))*remaining*(2.5/1000)*8;
    }
    check(std::abs(mass-.5)<1e-12,"Interface prior normalization");
    p.layer_weights*=1e300;
    check(std::abs(std::exp(p.log_density(three))-.02)<1e-14,"Weight scaling overflow");
    p.layer_weights=v({1,0,3});check(!p.supports(two) && std::isinf(p.log_density(two)),"Zero prior mass");
    for(const auto& invalid:std::vector<RJState>{ {v({.25}),v({2,2})},{v({3.75}),v({2,2})},
            {v({2,1}),v({2,2,2})},{v({1,1.25}),v({2,2,2})},{v({1}),v({2})},
            {v({}),v({NAN})},{v({}),v({.9})} }) check(!p.supports(invalid),"Invalid state supported");
    p.depth_limit=1.5;bool caught=false;
    try {p.validate();} catch(const std::invalid_argument&) {caught=true;}
    check(caught,"Zero-volume support accepted");
    p.depth_limit=4;p.layer_weights=v({1,2,3});
    auto same=[](const RJState& a,const RJState& b) {return a.interfaces==b.interfaces && a.vs==b.vs;};
    for(const auto& source:std::vector<RJState>{one,two}) {
        for(int t=1;t<20;++t) {
            const double depth=rj_insertion_position(p,source,t/20.);
            const auto born=rj_birth(p,source,depth,1.3,.2,.3);
            check(born.status==RJProposalStatus::valid,"Birth rejected valid auxiliaries");
            const auto dead=rj_death(p,born.state,born.index,.2,.3);
            check(dead.status==RJProposalStatus::valid && same(source,dead.state),"Birth/death inverse");
            check(std::abs(born.log_hastings()+dead.log_hastings())<1e-14,"Proposal ratios not inverse");
            check(born.log_jacobian==0 && dead.log_jacobian==0,"Unexpected Jacobian");
            auto target=[&](const RJState& x) {return p.log_density(x)-.7*x.vs.squaredNorm()-.1*x.interfaces.squaredNorm();};
            const double logratio=target(born.state)-target(source)+born.log_hastings();
            const double forward=target(source)+born.log_forward+std::min(0.,logratio);
            const double reverse=target(born.state)+born.log_reverse+std::min(0.,-logratio);
            check(std::abs(forward-reverse)<1e-12,"Detailed balance flow mismatch");
        }
    }
    const auto known=rj_birth(p,one,2,1.5,.2,.3);
    check(std::abs(std::exp(known.log_forward)-.2/6)<1e-14,"Birth density omitted length or Vs width");
    check(std::abs(std::exp(known.log_reverse)-.3)<1e-14,"Reverse death selection probability");
    const auto known2=rj_birth(p,two,1,1.8,.2,.3);
    check(std::abs(std::exp(known2.log_forward)-.2/4)<1e-14,"Multiple insertion intervals density");
    check(std::abs(std::exp(known2.log_reverse)-.3/2)<1e-14,"Death interface multiplicity");
    check(rj_birth(p,three,2,2).status==RJProposalStatus::unavailable,"Maximum layer self transition");
    check(rj_death(p,one,0).status==RJProposalStatus::unavailable,"Minimum layer self transition");
    check(rj_birth(p,two,2.1,2).status==RJProposalStatus::outside_support,"Too thin birth accepted");
    check(rj_birth(p,two,1,NAN).status==RJProposalStatus::outside_support,"Nonfinite auxiliary accepted");
    // Finite difference determinant of dimension-matching map (z,vs0,vs1,t,w).
    for(double t:{1.,3.}) {
        Eigen::VectorXd input=v({2.,2.5,1.5,t,1.8});Eigen::MatrixXd jacobian(5,5);
        auto mapping=[&](const Eigen::VectorXd& x) {
            const auto r=rj_birth(p,{v({x[0]}),v({x[1],x[2]})},x[3],x[4]);
            check(r.status==RJProposalStatus::valid,"Jacobian perturbation invalid");
            Eigen::VectorXd y(5);y<<r.state.interfaces,r.state.vs;return y;
        };
        for(int j=0;j<5;++j) {auto a=input,b=input;a[j]+=1e-5;b[j]-=1e-5;jacobian.col(j)=(mapping(a)-mapping(b))/(2e-5);}
        check(std::abs(std::abs(jacobian.determinant())-1)<1e-9,"Dimension matching Jacobian is not unity");
    }
    // Union sampling must be proportional to available interval lengths, not layer count.
    RJPrior uneven;uneven.max_layers=4;uneven.depth_limit=8;uneven.min_spacing=.5;uneven.layer_weights=Eigen::VectorXd::Ones(4);
    RJState gaps{v({1,3}),v({2,1,3})};
    check(rj_insertion_length(uneven,gaps)==5,"Union length excludes zero-width gap");
    int shallow=0;
    for(int j=0;j<10000;++j) {
        const double z=rj_insertion_position(uneven,gaps,(j+.5)/10000.);
        check((z>1.5 && z<2.5)||(z>3.5 && z<7.5),"Invalid insertion interval selected");
        shallow+=z<2.5;
    }
    check(shallow==2000,"Insertion density not uniform across union");
    check(rj_birth(uneven,gaps,.5,2).status==RJProposalStatus::outside_support,"Isolated zero-measure insertion accepted");
    RJState packed{v({.8}),v({2,2})};RJPrior narrow=p;narrow.depth_limit=1.6;
    check(rj_insertion_length(narrow,packed)==0,"Empty union not detected");
    check(rj_birth(narrow,packed,1,2).status==RJProposalStatus::unavailable,"No-insertion transition");
    RJState exact_boundary{v({.5,1}),v({2,2,2})};
    check(rj_death(p,exact_boundary,0).status==RJProposalStatus::outside_support,"Death with zero-measure reverse insertion accepted");
    int rejected=0;
    try {rj_death(p,two,9);} catch(const std::invalid_argument&) {++rejected;}
    try {rj_birth(p,one,2,2,0,.25);} catch(const std::invalid_argument&) {++rejected;}
    try {rj_insertion_position(p,one,1.);} catch(const std::invalid_argument&) {++rejected;}
    check(rejected==3,"Invalid proposal inputs not rejected");
    p.layer_weights=v({1,0,3});
    check(rj_birth(p,one,2,2).status==RJProposalStatus::outside_support,"Zero target prior weight ignored");
    std::cout<<"RJ prior normalization, support, low velocities and scaled weights, reversible proposals, Jacobians and detailed balance passed\n";
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
