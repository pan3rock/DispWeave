#include "dispweave/lm.hpp"
#include <iostream>
#include <stdexcept>
using Eigen::VectorXd;
VectorXd v(std::initializer_list<double> x) {return Eigen::Map<const VectorXd>(x.begin(),x.size());}
void check(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
void near(double a,double b,double tol) {check(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=tol,"Numerical mismatch");}
void rejects(const std::function<void()>& fn) {bool caught=false;try{fn();}catch(const std::invalid_argument&){caught=true;}check(caught,"Invalid input accepted");}
VectorXd phases(const dw::ForwardResult& f) {check(f.all_valid(),"Unresolved reference roots");VectorXd c(f.rows.size());for(size_t i=0;i<f.rows.size();++i)c[i]=f.rows[i].phase_velocity;return c;}
void materials() {
    dw::LayeredModel input{v({0,2,4}),v({1,2,3}),v({6,7,8}),v({2,2.5,3})};
    dw::InvLMForward backend(dw::SecularMethod::dunkin,dw::RootAccuracy::refined);
    const std::vector<dw::DispersionQuery> query{{.5,0},{.5,1},{.2,0}};
    for(const std::string kind:{"fixvprho","constant_ratio","brocher05","gardner","nearsurface","constant_rho"}) {
        auto ref=input;dw::ElasticRelation relation;relation.kind=kind;relation.vp_vs_ratio=1.8;
        if(kind=="constant_ratio") ref.vp=1.8*ref.vs;
        dw::FixedVs p(ref,.8*ref.vs,1.2*ref.vs,relation);
        const VectorXd q=ref.vs.array().log();const auto m=p.decode(q);
        check(p.encode(m).isApprox(q,1e-12),"Material round trip");
        const auto f=backend.evaluate(m,query,dw::SensitivityMethod::implicit);const auto j=p.phase_jacobian(q,f);
        const auto zero=Eigen::Matrix3d::Zero().eval(),identity=Eigen::Matrix3d::Identity().eval();
        const auto jvp=p.model_jacobian(q,zero,identity,zero),jrho=p.model_jacobian(q,zero,zero,identity);
        for(int col=0;col<3;++col) for(double h:{1e-4,1e-5}) {
            auto a=q,b=q;a[col]+=h;b[col]-=h;const auto ma=p.decode(a),mb=p.decode(b);
            const VectorXd fd=(phases(backend.evaluate(ma,query))-phases(backend.evaluate(mb,query)))/(2*h);
            check((fd-j.col(col)).cwiseAbs().maxCoeff()<5e-6,"Empirical phase chain rule");
            near((ma.vp[col]-mb.vp[col])/(2*h),jvp(col,col),1e-6);
            near((ma.rho[col]-mb.rho[col])/(2*h),jrho(col,col),1e-6);
        }
        if(kind=="brocher05" || kind=="gardner" || kind=="nearsurface")
            rejects([&]{p.model_jacobian(q,zero,identity);});
        auto wrong=m;wrong.rho[0]+=.1;rejects([&]{p.encode(wrong);});
        dw::DispersionObservations data{query,phases(f)+v({.001,-.002,.003}),Eigen::Matrix3d::Identity()*1e-4};
        dw::DispersionProblem problem(backend,p,data,dw::SensitivityMethod::implicit);
        const auto prior=dw::uniform_prior(.8*ref.vs,1.2*ref.vs,dw::PriorMeasure::physical_vs);
        const auto posterior=dw::log_posterior(problem,q,prior);
        check(posterior.status==dw::EvaluationStatus::valid,"Empirical posterior status");
        for(int col=0;col<3;++col) {auto a=q,b=q;a[col]+=1e-6;b[col]-=1e-6;
            near((dw::log_posterior(problem,a,prior,false).value-dw::log_posterior(problem,b,prior,false).value)/2e-6,
                 posterior.gradient[col],1e-4);}
        const auto reference_prior=dw::bounded_gaussian_prior(ref.vs,VectorXd::Constant(3,.1),
                                  .8*ref.vs,1.2*ref.vs,dw::PriorMeasure::physical_vs);
        const VectorXd shifted=q+v({.01,-.01,.01});
        const auto regularized=dw::log_posterior(problem,shifted,reference_prior);
        check(regularized.status==dw::EvaluationStatus::valid,"Reference posterior status");
        for(int col=0;col<3;++col) {auto a=shifted,b=shifted;a[col]+=1e-6;b[col]-=1e-6;
            near((dw::log_posterior(problem,a,reference_prior,false).value-dw::log_posterior(problem,b,reference_prior,false).value)/2e-6,
                 regularized.gradient[col],1e-4);}
    }
    dw::ElasticRelation near_surface;near_surface.kind="nearsurface";
    dw::FixedVs p(input,.8*input.vs,VectorXd::Constant(3,10),near_surface);
    check(!p.contains(v({10,10,10}).array().log()),"Unphysical extrapolated density accepted");
    rejects([&]{p.decode(v({10,10,10}).array().log());});
}
class IdentityForward final:public dw::Forward {
public:
    dw::ForwardResult evaluate(const dw::LayeredModel& m,const std::vector<dw::DispersionQuery>& query,dw::SensitivityMethod) const override {
        check(query.size()==3,"Identity observation count");dw::ForwardResult f;f.rows.resize(3);
        for(int i=0;i<3;++i) {auto& r=f.rows[i];r.status=dw::RootStatus::valid;r.phase_velocity=m.vs[i];
            r.sensitivity.status=dw::SensitivityStatus::valid;r.sensitivity.dc_dvs=VectorXd::Unit(3,i);
            r.sensitivity.dc_dvp=VectorXd::Zero(3);r.sensitivity.dc_drho=VectorXd::Zero(3);}
        return f;
    }
};
void penalties() {
    const VectorXd depth=v({0,2,5}),reference=v({1,2,3}),trial=v({1.1,1.8,3.3}),q=trial.array().log();
    const std::vector<std::pair<std::string,double>> cases{{"reference_damping",.63},{"perturbation_smooth",.5775},{"vs_smooth",4.4775}};
    for(const auto& [type,expected]:cases) {
        dw::VsRegularization r(depth,reference,type,9);const auto e=r.evaluate(q);
        near(.5*e.residual.squaredNorm(),expected,1e-13);
        for(int j=0;j<3;++j) {auto a=q,b=q;a[j]+=1e-6;b[j]-=1e-6;
            check(((r.evaluate(a).residual-r.evaluate(b).residual)/(2e-6)-e.jacobian.col(j)).norm()<2e-9,"Penalty chain rule");}
        if(type=="perturbation_smooth") near(r.evaluate(reference.array().log()).residual.norm(),0,1e-14);
    }
    rejects([&]{dw::VsRegularization r(depth,reference,"none",1);});
    rejects([&]{dw::VsRegularization r(depth,reference,"invalid",1);});
    rejects([&]{dw::VsRegularization r(depth,reference,"reference_damping",-1);});
    dw::LayeredModel m{depth,reference,v({5,6,7}),v({2,2,2})};dw::FixedVs p(m,.7*reference,1.3*reference);
    IdentityForward backend;dw::DispersionObservations data{{{1,0},{2,0},{3,0}},v({1.2,2.3,3}),Eigen::Matrix3d::Identity()};
    dw::DispersionProblem problem(backend,p,data);dw::VsRegularization reg(depth,reference,"reference_damping",4);
    const auto initial=p.encode(m);const auto fit=dw::invert_lm(problem,initial,{}, {2},&reg);
    check(fit.converged,"Penalized conditional solution did not converge");
    check((fit.q.array().exp().matrix()-v({1.04,2.06,3})).norm()<1e-7,"Analytic regularized optimum");
    near(fit.evaluation.data_cost+fit.evaluation.regularization_cost,fit.evaluation.least_squares_cost,1e-13);
    near(fit.evaluation.log_likelihood,problem.evaluate(fit.q).log_likelihood,1e-13);
    check(fit.evaluation.jacobian.rows()==6 && fit.evaluation.jacobian.cols()==3,"Restored penalized Jacobian");
    dw::VsRegularization none(depth,reference,"none",0);
    const auto a=dw::invert_lm(problem,initial),b=dw::invert_lm(problem,initial,{}, {},&none);
    check(a.q==b.q && a.evaluations==b.evaluations && a.evaluation.least_squares_cost==b.evaluation.least_squares_cost,"Disabled penalty changes optimization");
}
int main() {try{materials();penalties();std::cout<<"PASS empirical derivatives/posteriors, penalties and conditional optimum\n";return 0;}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
