#include "dispweave/lm.hpp"
#include <iostream>
#include <stdexcept>
using Eigen::VectorXd;
void check(bool condition,const char* message) { if(!condition) throw std::runtime_error(message); }
VectorXd v(std::initializer_list<double> a) { return Eigen::Map<const VectorXd>(a.begin(),a.size()); }
dw::DataEvaluation linear(const VectorXd& x) {
    dw::DataEvaluation e; e.status=dw::EvaluationStatus::valid;
    e.jacobian.resize(3,2); e.jacobian<<1,2,2,-1,1,1;
    e.residual=e.jacobian*(x-v({1,2})); return e;
}
int main(int argc,char** argv) { try {
    const std::string test=argc>1?argv[1]:"";
    if(test=="linear") {
        auto r=dw::bounded_lm(linear,v({0,0}),v({-5,-5}),v({5,5}));
        check(r.converged && (r.q-v({1,2})).norm()<1e-8,"linear optimum");
        auto b=dw::bounded_lm(linear,v({0,0}),v({-5,-5}),v({.5,5}));
        // constrained x=.5; optimal y=2+1/12 from normal equation
        check(b.converged && (b.q-v({.5,2+1./12})).norm()<1e-7,"box KKT optimum");
        auto coupled=[](const VectorXd& x) {
            dw::DataEvaluation e; e.status=dw::EvaluationStatus::valid;
            e.jacobian.resize(2,2); e.jacobian<<1,.99,0,std::sqrt(1-.99*.99);
            e.residual=e.jacobian*x+v({1,(1.5-.99)/std::sqrt(1-.99*.99)}); return e;
        };
        auto c=dw::bounded_lm(coupled,v({0,0}),v({-4,-4}),v({0,4}));
        check(c.converged && (c.q-v({0,-1.5})).norm()<1e-6 && c.cauchy_steps>0,"clipped non-descent Cauchy recovery");
    } else if(test=="nonlinear") {
        auto f=[](const VectorXd& x) { dw::DataEvaluation e; e.status=dw::EvaluationStatus::valid;
            e.residual=v({10*(x[1]-x[0]*x[0]),1-x[0]}); e.jacobian.resize(2,2); e.jacobian<<-20*x[0],10,-1,0; return e; };
        auto r=dw::bounded_lm(f,v({-1.2,1}),v({-3,-3}),v({3,3}));
        check(r.converged && (r.q-v({1,1})).norm()<1e-6,"Rosenbrock minimum");
        check(r.rejected>0,"nonlinear rejection exercised");
    } else if(test=="failures") {
        auto r=dw::bounded_lm(linear,v({6,0}),v({-5,-5}),v({5,5}));
        check(!r.converged && r.evaluations==0,"out of bounds initial");
        dw::LMOptions o; o.max_iterations=1;
        r=dw::bounded_lm(linear,v({0,0}),v({-5,-5}),v({5,5}),o);
        check(!r.converged && r.reason=="max_iterations","iteration limit");
        auto invalid=[](const VectorXd&) { dw::DataEvaluation e; e.status=dw::EvaluationStatus::forward_failure; return e; };
        r=dw::bounded_lm(invalid,v({0}),v({-1}),v({1}));
        check(!r.converged && r.forward_failures==1 && r.reason=="invalid_initial_evaluation","invalid initial physics");
        auto blocked=[](const VectorXd& x) { dw::DataEvaluation e; e.status=x[0]>.1?dw::EvaluationStatus::forward_failure:dw::EvaluationStatus::valid;
            e.residual=v({x[0]-1}); e.jacobian=Eigen::MatrixXd::Ones(1,1); return e; };
        r=dw::bounded_lm(blocked,v({0}),v({-2}),v({2}));
        check(!r.converged && r.forward_failures>0 && r.rejected>0,"missing trial cannot converge by shrinking");
        auto bad_gradient=[](const VectorXd&) { dw::DataEvaluation e; e.status=dw::EvaluationStatus::gradient_failure; return e; };
        r=dw::bounded_lm(bad_gradient,v({0}),v({-1}),v({1}));
        check(!r.converged && r.gradient_failures==1,"gradient failure");
    } else throw std::runtime_error("unknown test");
    std::cout<<test<<" passed\n"; return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; } }
