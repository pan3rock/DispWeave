#include "dispweave/mode_support.hpp"
#include "dispweave/objective.hpp"
#include "models.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
using namespace dw;
void require(bool b,const char* s) {if(!b) throw std::runtime_error(s);}
LayeredModel load(const std::string& file) {
    std::ifstream f(file);std::string line;std::vector<double> z,r,v,p;
    while(std::getline(f,line)) {if(line.empty() || line[0]=='#') continue;std::istringstream in(line);double i,a,b,c,d;
        if(in>>i>>a>>b>>c>>d) {z.push_back(a);r.push_back(b);v.push_back(c);p.push_back(d);}}
    auto vec=[](const auto& a) {return Eigen::Map<const Eigen::VectorXd>(a.data(),a.size()).eval();};
    LayeredModel m{vec(z),vec(v),vec(p),vec(r)};m.validate();return m;
}
int main(int argc,char** argv) {
    try {
        if(argc!=2) throw std::runtime_error("Need fixture directory");
        auto start=std::chrono::steady_clock::now();
        for(double ratio:{1.16,1.4142135623730951,1.8,3.}) {
            LayeredModel m{v({0}),v({3}),v({3*ratio}),v({2.7})};
            const auto c=trapped_mode_count(m,.1);
            require(c.resolved && c.count==1,"Homogeneous analytic count");
            LayeredModel split{v({0,1,20}),v({3,3,3}),v({3*ratio,3*ratio,3*ratio}),v({2.7,2.7,2.7})};
            const auto divided=trapped_mode_count(split,.1);
            require(divided.resolved && divided.count==1,"Subdivision invariance");
        }
        for(const auto& name:{"formal","warmup"}) {
            auto m=load(std::string(argv[1])+"/crust-cutoff-"+name+".txt");
            auto c=trapped_mode_count(m,.1612);
            std::cout<<name<<" count="<<c.count<<" calls="<<c.evaluations<<" "<<c.message<<'\n';
            require(c.resolved && c.count==2,"Absent third mode at saved failure");
            require(!trapped_mode_count(m,.1612,64).resolved,"Budget exhaustion became absence");
            InvLMForward backend;FixedVs parameters(m,.9*m.vs,1.1*m.vs);
            DispersionObservations data{{{.1612,2}},v({4.45}),Eigen::MatrixXd::Identity(1,1)*.0004};
            DispersionProblem strict(backend,parameters,data,SensitivityMethod::implicit);
            DispersionProblem support(backend,parameters,data,SensitivityMethod::implicit,ModeSupportPolicy::trapped_modes);
            const auto q=parameters.encode(m);
            require(strict.evaluate(q).status==EvaluationStatus::forward_failure,"Default policy changed");
            const auto prior=uniform_prior(parameters.lower(),parameters.upper(),PriorMeasure::log_vs);
            const auto post=log_posterior(support,q,prior);
            require(post.status==EvaluationStatus::outside_support && post.value==-INFINITY,"Absence likelihood contract");
            m.vs[m.vs.size()-1]+=.025;
            const auto present=trapped_mode_count(m,.1612);
            require(present.resolved && present.count==3,"Cutoff continuation count");
            require(backend.evaluate(m,{{.1612,2}}).all_valid(),"Cutoff continuation root");
            if(std::string(name)=="formal") for(double offset:{-1e-8,0.,1e-8}) {
                const double v=4.439194324296876+offset;
                const double vp=.9409+2.0947*v-.8206*v*v+.2683*std::pow(v,3)-.0251*std::pow(v,4);
                const double rho=1.6612*vp-.4721*vp*vp+.0671*std::pow(vp,3)-.0043*std::pow(vp,4)+.000106*std::pow(vp,5);
                m.vs.tail(1)[0]=v;m.vp.tail(1)[0]=vp;m.rho.tail(1)[0]=rho;
                const auto endpoint=trapped_mode_count(m,.1612);
                if(offset==0) require(!endpoint.resolved,"Numerically singular threshold misclassified as absence");
                else require(endpoint.resolved && endpoint.count==(offset<0?2:3),"Near-threshold continuation");
            }
        }
        struct Broken : Forward {
            ForwardResult evaluate(const LayeredModel&,const std::vector<DispersionQuery>& q,SensitivityMethod) const override {
                ForwardResult r;r.rows.resize(q.size());return r;
            }
        } broken;
        LayeredModel m{v({0}),v({3}),v({5.4}),v({2.7})};FixedVs p(m,.9*m.vs,1.1*m.vs);
        DispersionProblem missing(broken,p,{{{.1,0}},v({2.7}),Eigen::MatrixXd::Identity(1,1)},SensitivityMethod::implicit,ModeSupportPolicy::trapped_modes);
        require(missing.evaluate(p.encode(m)).status==EvaluationStatus::forward_failure,"Missed existing root hidden as absence");
        std::cout<<"seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<'\n';
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
