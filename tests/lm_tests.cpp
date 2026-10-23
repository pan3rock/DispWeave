#include "dispweave/lm.hpp"
#include <iostream>
#include <Eigen/Cholesky>
#include <random>
#include <stdexcept>
using Eigen::VectorXd;
void check(bool condition,const char* message) { if(!condition) throw std::runtime_error(message); }
VectorXd v(std::initializer_list<double> a) { return Eigen::Map<const VectorXd>(a.begin(),a.size()); }
dw::DataEvaluation linear(const VectorXd& x) {
    dw::DataEvaluation e; e.status=dw::EvaluationStatus::valid;
    e.jacobian.resize(3,2); e.jacobian<<1,2,2,-1,1,1;
    e.residual=e.jacobian*(x-v({1,2})); return e;
}
// Independent analytic observation model to check conditional optimization:
// phase = 10 + A log(Vs), with a nonzero residual in the fixed coordinates.
class LogLinearForward final : public dw::Forward {
public:
    Eigen::Matrix3d a;
    LogLinearForward() {a<<1,2,3,0,3,-1,2,1,2;}
    dw::ForwardResult evaluate(const dw::LayeredModel& m,const std::vector<dw::DispersionQuery>& queries,
                              dw::SensitivityMethod method) const override {
        check(queries.size()==3,"analytic query count");
        dw::ForwardResult out;out.rows.resize(3);out.sensitivity_method=method;
        VectorXd c=VectorXd::Constant(3,10)+a*m.vs.array().log().matrix();
        for(int i=0;i<3;++i) {
            auto& r=out.rows[i];r.phase_velocity=c[i];r.status=dw::RootStatus::valid;
            r.sensitivity.status=dw::SensitivityStatus::valid;
            r.sensitivity.dc_dvs=a.row(i).transpose().array()/m.vs.array();
            r.sensitivity.dc_dvp=VectorXd::Zero(3);r.sensitivity.dc_drho=VectorXd::Zero(3);
        }
        return out;
    }
};
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
        check(c.converged && (c.q-v({0,-1.5})).norm()<1e-6 && c.evaluations<12 && c.box_iterations>0,"bounded coupled quadratic");
    } else if(test=="nonlinear") {
        auto f=[](const VectorXd& x) { dw::DataEvaluation e; e.status=dw::EvaluationStatus::valid;
            e.residual=v({10*(x[1]-x[0]*x[0]),1-x[0]}); e.jacobian.resize(2,2); e.jacobian<<-20*x[0],10,-1,0; return e; };
        auto r=dw::bounded_lm(f,v({-1.2,1}),v({-3,-3}),v({3,3}));
        check(r.converged && (r.q-v({1,1})).norm()<1e-6,"Rosenbrock minimum");
        check(r.rejected>0,"nonlinear rejection exercised");
    } else if(test=="boxes") {
        // Independent reference: enumerate every active face and solve its normal equations.
        std::mt19937 rng(17); std::normal_distribution<double> normal;
        for(int sample=0;sample<40;++sample) {
            Eigen::MatrixXd a(5,3); VectorXd b(5);
            for(int i=0;i<5;++i) { b[i]=3*normal(rng); for(int j=0;j<3;++j) a(i,j)=normal(rng); }
            const VectorXd lo=v({-1,-2,-.5}),hi=v({.8,.5,2});
            VectorXd reference; double best=INFINITY;
            for(int face=0;face<27;++face) {
                int code=face; VectorXd q=VectorXd::Zero(3); std::vector<int> free;
                for(int k=0;k<3;++k) { int state=code%3; code/=3;
                    if(state==0) free.push_back(k); else q[k]=state==1?lo[k]:hi[k]; }
                if(!free.empty()) {
                    Eigen::MatrixXd reduced(5,free.size());
                    for(size_t k=0;k<free.size();++k) reduced.col(k)=a.col(free[k]);
                    const VectorXd sol=(reduced.transpose()*reduced).ldlt().solve(reduced.transpose()*(b-a*q));
                    for(size_t k=0;k<free.size();++k) q[free[k]]=sol[k];
                }
                if((q.array()<lo.array()-1e-12).any() || (q.array()>hi.array()+1e-12).any()) continue;
                const double cost=(a*q-b).squaredNorm(); if(cost<best) {best=cost; reference=q;}
            }
            auto f=[&](const VectorXd& q) { dw::DataEvaluation e; e.status=dw::EvaluationStatus::valid;
                e.residual=a*q-b; e.jacobian=a; return e; };
            for(const auto& start:std::vector<VectorXd>{lo,hi,VectorXd::Zero(3)}) {
                auto r=dw::bounded_lm(f,start,lo,hi);
                check(r.converged && (r.q-reference).norm()<1e-6,"enumerated box optimum");
                check((r.q.array()>=lo.array()).all() && (r.q.array()<=hi.array()).all(),"box feasibility");
            }
        }
    } else if(test=="precision") {
        auto offset=[](const VectorXd& x) { dw::DataEvaluation e; e.status=dw::EvaluationStatus::valid;
            e.residual=v({1e12,x[0]-1}); e.jacobian.resize(2,1); e.jacobian<<0,1; return e; };
        auto r=dw::bounded_lm(offset,v({0}),v({-2}),v({2}));
        check(r.converged && std::abs(r.q[0]-1)<1e-7,"large constant cost must not erase actual decrease");
        check(r.trace[0].cost==r.trace[0].trial_cost && r.trace[0].actual_reduction>0 && r.trace[0].accepted,
              "factored residual decrease survives equal rounded costs");
        // Simulated evaluation plateau with a nonzero reported gradient; do not claim convergence.
        auto plateau=[](const VectorXd&) { dw::DataEvaluation e; e.status=dw::EvaluationStatus::valid;
            e.residual=v({1}); e.jacobian=Eigen::MatrixXd::Ones(1,1); return e; };
        r=dw::bounded_lm(plateau,v({0}),v({-2}),v({2}));
        check(!r.converged && r.termination==dw::LMTermination::stalled && r.reason=="step_stagnation",
              "finite evaluation plateau classified as stagnation");
        check(r.projected_gradient>1e-7 && r.q[0]==0 && r.rejected>0,"stagnation is not success");
    } else if(test=="fixed") {
        dw::LayeredModel m{v({0,1,2}),v({1,2,3}),v({10,10,10}),v({2,2,2})};
        dw::FixedVs p(m,.5*m.vs,1.5*m.vs);LogLinearForward forward;
        const VectorXd target=v({1.3,2.2,2.8}).array().log();
        dw::DispersionObservations data{{{1,0},{2,0},{3,0}},
            VectorXd::Constant(3,10)+forward.a*target,Eigen::Matrix3d::Identity()};
        dw::DispersionProblem problem(forward,p,data);
        const auto initial=p.encode(m);
        const double expected=initial[1]+forward.a.col(1).dot(forward.a*(target-initial))/forward.a.col(1).squaredNorm();
        auto r=dw::invert_lm(problem,initial,{}, {2,0});
        check(r.converged && std::abs(r.q[1]-expected)<1e-8,"conditional analytic optimum");
        check(r.q[0]==initial[0] && r.q[2]==initial[2],"fixed coordinates moved");
        check(r.evaluation.jacobian.cols()==3 && r.evaluation.jacobian.isApprox(forward.a,1e-12),"full Jacobian contract");
        check((r.evaluation.jacobian.transpose()*r.evaluation.residual).norm()>.1 && r.projected_gradient<1e-7,
              "fixed gradient must not prevent conditional convergence");
        for(const auto& mask:std::vector<std::vector<int>>{{-1},{3},{0,0},{0,1,2}}) {
            bool threw=false;try {dw::invert_lm(problem,initial,{},mask);} catch(const std::invalid_argument&) {threw=true;}
            check(threw,"invalid fixed layer list accepted");
        }
        auto invalid=initial;invalid[0]=p.lower()[0]-1;
        r=dw::invert_lm(problem,invalid,{}, {0});
        check(r.reason=="invalid_initial_bounds" && r.evaluations==0,"fixed initial outside bounds accepted");
        dw::LMOptions options;options.max_iterations=1;
        r=dw::invert_lm(problem,initial,options,{2,0});
        check(!r.converged && r.reason=="max_iterations" && r.q.size()==3 && r.q[2]==initial[2],"fixed failed result contract");
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
        check(!r.converged && r.termination==dw::LMTermination::failed && r.forward_failures>0 && r.rejected>0,"missing trial cannot converge by shrinking");
        auto bad_gradient=[](const VectorXd&) { dw::DataEvaluation e; e.status=dw::EvaluationStatus::gradient_failure; return e; };
        r=dw::bounded_lm(bad_gradient,v({0}),v({-1}),v({1}));
        check(!r.converged && r.gradient_failures==1,"gradient failure");
    } else throw std::runtime_error("unknown test");
    std::cout<<test<<" passed\n"; return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; } }
