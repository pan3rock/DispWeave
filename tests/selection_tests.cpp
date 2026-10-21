#include "dispweave/selection.hpp"
#include <iostream>
#include <stdexcept>
void check(bool condition,const char* message) { if(!condition) throw std::runtime_error(message); }
int main() { try {
    auto ic=dw::information_criteria(-10,100,3);
    check(std::abs(ic.aic-26)<1e-12 && std::abs(ic.bic-(20+3*std::log(100.)))<1e-12,"IC formulas");
    bool caught=false; try { dw::information_criteria(NAN,10,1); } catch(const std::invalid_argument&) { caught=true; }
    check(caught,"invalid likelihood");
    dw::LMResult fit; fit.q=Eigen::Vector2d::Zero(); fit.converged=true;
    fit.evaluation.status=dw::EvaluationStatus::valid; fit.evaluation.log_likelihood=-10;
    fit.evaluation.residual=Eigen::Vector3d::Ones(); fit.evaluation.jacobian.resize(3,2);
    fit.evaluation.jacobian<<1,0,0,1,1,1;
    const Eigen::Vector2d lo(-1,-1),hi(1,1),scale(.1,.1);
    auto score=dw::assess_candidate(fit,lo,hi,scale,1);
    check(score.regular && score.aic==24,"regular interior fit");
    fit.q[0]=1;
    score=dw::assess_candidate(fit,lo,hi,scale,1);
    check(score.fitted && !score.regular && std::isnan(score.bic) && score.reason=="boundary_solution","boundary excluded");
    fit.q[0]=0; fit.evaluation.jacobian.col(1)=fit.evaluation.jacobian.col(0);
    score=dw::assess_candidate(fit,lo,hi,scale,1);
    check(!score.regular && score.reason=="rank_deficient","rank deficiency excluded");
    fit.converged=false;
    score=dw::assess_candidate(fit,lo,hi,scale,1);
    check(!score.fitted && std::isnan(score.log_likelihood),"nonconverged not scored");
    // Known linear Gaussian prediction: independent held-out likelihood retains normalization.
    dw::DispersionObservations data{{{1,0},{2,0}},Eigen::Vector2d(2,3),4*Eigen::Matrix2d::Identity()};
    const double expected=-.5*(.5+2*std::log(8*std::acos(-1.)));
    check(std::abs(dw::GaussianLikelihood(data).log_likelihood(Eigen::Vector2d(1,4))-expected)<1e-12,"validation Gaussian score");
    std::cout<<"selection checks passed\n";
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; } }
