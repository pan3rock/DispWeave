#include "dispweave/forward.hpp"
#include "models.hpp"
#include "independent_reference.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <functional>
#include <limits>
#include <map>
void check(bool ok,const std::string& why) { if(!ok) throw std::runtime_error(why); }
void near(double a,double b,double tol) { std::ostringstream s; s<<std::setprecision(17)<<"Mismatch: "<<a<<" vs "<<b<<" tolerance "<<tol; check(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=tol,s.str()); }
void rejects(const std::function<void()>& call) { bool caught=false; try { call(); } catch(const std::invalid_argument&) { caught=true; } check(caught,"Expected invalid_argument"); }
std::vector<dw::DispersionQuery> modes(double f,int n) { std::vector<dw::DispersionQuery> q; for(int i=0;i<n;++i) q.push_back({f,i}); return q; }
void validation() {
    normal().validate(); low_velocity().validate();
    for(int kind=0;kind<8;++kind) { auto m=normal();
        if(kind==0) m.vs.resize(0); if(kind==1) m.depth[0]=1; if(kind==2) m.depth[1]=0;
        if(kind==3) m.vs[0]=0; if(kind==4) m.vp[0]=-1; if(kind==5) m.rho[0]=-1;
        if(kind==6) m.vs[0]=NAN; if(kind==7) m.vp[0]=1.1*m.vs[0];
        rejects([&]{m.validate();});
    }
    dw::InvLMForward f;
    for(const auto& q:std::vector<std::vector<dw::DispersionQuery>>{{},{{0,0}},{{NAN,0}},{{INFINITY,0}},{{1,-1}},{{1,1024}}})
        rejects([&]{f.evaluate(normal(),q);});
}
void halfspace() {
    dw::InvLMForward backend;
    for(double ratio:{1.155,1.2,1.41,1.73,3.0,10.0}) {
        dw::LayeredModel m{v({0}),v({.6}),v({ratio*.6}),v({2})};
        // Independent Rayleigh cubic in x=(c/Vs)^2; physical root in (0,1).
        const double k=1/(ratio*ratio);
        auto polynomial=[&](double x){return x*x*x-8*x*x+(24-16*k)*x-16*(1-k);};
        double lo=0,hi=1;
        for(int j=0;j<80;++j) {double mid=(lo+hi)/2; if(polynomial(mid)>0) hi=mid; else lo=mid;}
        const double expected=.6*std::sqrt((lo+hi)/2);
        auto r=backend.evaluate(m,{{.001,0},{1,0},{1000,0},{1,1}});
        for(int i=0;i<3;++i) { std::cout<<"ratio="<<ratio<<" row="<<i<<" status="<<dw::to_string(r.rows[i].status)<<" message="<<r.rows[i].message<<"\n"; near(r.rows[i].phase_velocity,expected,2e-9); }
        check(r.rows[3].status==dw::RootStatus::missing_or_unresolved,"No halfspace overtone");
        check(std::isnan(r.rows[3].phase_velocity),"No fallback value");
    }
}
void golden() {
    const auto r=dw::InvLMForward().evaluate(normal(),modes(.5,5)); check(r.all_valid(),"Golden valid");
    const double expected[]={.929396084266,1.49316010302,1.91942613348,2.84123908387,2.99610402191};
    for(int i=0;i<5;++i) near(r.rows[i].phase_velocity,expected[i],2e-9);
    const auto l=dw::InvLMForward().evaluate(low_velocity(),modes(19.70981,8));
    near(l.rows[0].phase_velocity,.173075476118,2e-9); near(l.rows[1].phase_velocity,.265926133202,1e-8);
    near(l.rows[2].phase_velocity,.267443878583,1e-8);
    check(l.rows[7].status==dw::RootStatus::missing_or_unresolved,"LVL eighth unresolved");
}
void independent() {
    const auto m=normal(); const auto r=dw::InvLMForward().evaluate(m,modes(.5,5));
    // Fixed brackets specified before evaluating the production backend.
    const double brackets[][2]={{.92,.94},{1.48,1.51},{1.91,1.93},{2.83,2.85},{2.995,2.997}};
    double worst=0;
    for(int i=0;i<5;++i) {
        const double root=reference_root(m,.5,brackets[i][0],brackets[i][1]);
        near(root,r.rows[i].phase_velocity,2e-8); worst=std::max(worst,std::abs(root-r.rows[i].phase_velocity));
    }
    std::cout<<"Independent five-root max absolute error = "<<worst<<" km/s\n";
    const auto lvl=low_velocity(); const auto l=dw::InvLMForward().evaluate(lvl,modes(19.70981,3));
    const double lb[][2]={{.172,.174},{.2658,.2661},{.2673,.2676}};
    for(int i=0;i<3;++i) near(reference_root(lvl,19.70981,lb[i][0],lb[i][1]),l.rows[i].phase_velocity,2e-8);
}
void scenarios() {
    auto high=normal(); high.vs=v({1,3.5,3}); high.vp=v({1.8,6.2,5.2});
    auto contrast=normal(); contrast.vs=v({.2,1.5,3}); contrast.vp=v({.4,2.7,5.2}); contrast.depth=v({0,.02,.1});
    for(const auto& item:std::vector<std::pair<std::string,dw::LayeredModel>>{{"normal",normal()},{"low_velocity",low_velocity()},{"high_velocity",high},{"strong_contrast",contrast},{"dense",dense()}}) {
        const auto& [name,m]=item;
        int valid=0,unresolved=0,numerical=0;
        for(double freq:{.01,.5,5.,20.,100.}) {
            auto q=modes(freq,4); auto a=dw::InvLMForward().evaluate(m,q);
            auto b=dw::InvLMForward(dw::SecularMethod::fast_scalar).evaluate(m,q);
            double previous=0;
            for(size_t i=0;i<a.rows.size();++i) {
                const auto& row=a.rows[i];
                if(row.status==dw::RootStatus::valid) {
                    ++valid; check(row.phase_velocity>previous && row.phase_velocity<m.vs.tail(1)[0],"Trapped ordered roots"); previous=row.phase_velocity;
                    if(b.rows[i].status==dw::RootStatus::valid) near(row.phase_velocity,b.rows[i].phase_velocity,2e-7);
                } else { check(std::isnan(row.phase_velocity),"Invalid value must be NaN");
                    if(row.status==dw::RootStatus::numerical_failure) ++numerical; else ++unresolved; }
            }
        }
        check(valid>0,"Scenario has no resolved roots");
        std::cout<<name<<" valid="<<valid<<" unresolved="<<unresolved<<" numerical="<<numerical<<'\n';
    }
}
void ordering() {
    dw::InvLMForward f; const auto m=normal();
    const std::vector<dw::DispersionQuery> q={{.5,4},{.1,0},{.5,0},{.5,4},{.5,1},{.1,3}};
    const auto a=f.evaluate(m,q); check(a.rows.size()==q.size(),"Row preservation");
    f.evaluate(low_velocity(),modes(20,5)); const auto b=f.evaluate(m,q);
    for(size_t i=0;i<q.size();++i) {
        const auto alone=f.evaluate(m,{q[i]}).rows[0]; check(a.rows[i].status==alone.status,"Batch status differs");
        if(alone.status==dw::RootStatus::valid) { near(a.rows[i].phase_velocity,alone.phase_velocity,1e-10); near(a.rows[i].phase_velocity,b.rows[i].phase_velocity,0); }
    }
}
void subdivision() {
    auto m=normal(); dw::LayeredModel split{v({0,1,2,3,4}),v({1,1,2,2,3}),v({1.8,1.8,3.6,3.6,5.2}),v({2,2,2.2,2.2,2.5})};
    const auto a=dw::InvLMForward().evaluate(m,modes(.5,5)),b=dw::InvLMForward().evaluate(split,modes(.5,5));
    check(a.all_valid()&&b.all_valid(),"Subdivision valid");
    for(size_t i=0;i<a.rows.size();++i) near(a.rows[i].phase_velocity,b.rows[i].phase_velocity,2e-9);
}
void cutoff() {
    auto m=normal(); bool missing=false,found=false;
    for(double f:{.001,.01,.05,.1,.2,.5,1.}) {
        const auto r=dw::InvLMForward().evaluate(m,{{f,1},{f,20}});
        check(r.rows.size()==2,"Cutoff row loss");
        if(r.rows[0].status==dw::RootStatus::valid) found=true;
        if(r.rows[0].status==dw::RootStatus::missing_or_unresolved) missing=true;
        for(const auto& row:r.rows) if(row.status!=dw::RootStatus::valid) check(std::isnan(row.phase_velocity),"Cutoff fallback");
        std::cout<<"f="<<f<<" mode1="<<dw::to_string(r.rows[0].status)<<'\n';
    }
    check(missing&&found,"Need both sides of unresolved transition");
}
void numerical() {
    const auto r=dw::InvLMForward().evaluate(normal(),{{1e300,0},{.5,0},{1e6,0}});
    check(r.rows[0].status==dw::RootStatus::numerical_failure,"Overflow guard diagnostic");
    check(r.rows[1].status==dw::RootStatus::valid,"Failures must not poison other frequencies");
    check(r.rows[2].status==dw::RootStatus::numerical_failure,"Upstream sampling-cap diagnostic");
    check(std::isnan(r.rows[0].phase_velocity)&&std::isnan(r.rows[2].phase_velocity),"Failure cannot return a phase");
}
// Saved RJ failures: a fast cap above a slow halfspace, and a near-cutoff
// trapped root. Absence is deliberately not certified by this regression.
void trapped_domain() {
    dw::LayeredModel cap{v({0,3.5846558879346531}),v({1.8,1.0219876911644241}),
        v({3.24,1.8*1.0219876911644241}),v({2.3,2.3})};
    for(auto method:{dw::SecularMethod::dunkin,dw::SecularMethod::fast_scalar}) {
        dw::InvLMForward backend(method);
        for(double f:{.15,.2,.3,.4,.6,.9}) {
            auto r=backend.evaluate(cap,modes(f,3));
            for(const auto& row:r.rows) {
                check(row.status==dw::RootStatus::missing_or_unresolved,"Out-of-domain candidate contaminated search");
                check(row.roots_found==0 && std::isnan(row.phase_velocity),"Spurious trapped root");
            }
        }
        dw::LayeredModel near_cutoff{v({0,.48868227440428891,1.6165500687171335}),
            v({1.7715572106514081,1.4904606532961371,1.1436030468024461}),v({1,1,1}),v({2.3,2.3,2.3})};
        near_cutoff.vp=1.8*near_cutoff.vs;
        const double independent=reference_root(near_cutoff,.15,1.14343,1.14344);
        const auto one=backend.evaluate(near_cutoff,modes(.15,1));
        const auto many=backend.evaluate(near_cutoff,modes(.15,4));
        near(one.rows[0].phase_velocity,independent,2e-9);
        near(many.rows[0].phase_velocity,independent,2e-9);
        for(size_t i=1;i<many.rows.size();++i)
            check(many.rows[i].status==dw::RootStatus::missing_or_unresolved,"Unresolved overtone fabricated");
    }
}
int main(int argc,char** argv) {
    const std::map<std::string,std::function<void()>> tests={{"validation",validation},{"halfspace",halfspace},{"golden",golden},{"independent",independent},{"scenarios",scenarios},{"ordering",ordering},{"subdivision",subdivision},{"cutoff",cutoff},{"numerical",numerical},{"trapped_domain",trapped_domain}};
    try { if(argc!=2 || !tests.count(argv[1])) throw std::invalid_argument("Choose test name"); tests.at(argv[1])(); std::cout<<"PASS "<<argv[1]<<'\n'; return 0; }
    catch(const std::exception& e) {std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1;}
}
