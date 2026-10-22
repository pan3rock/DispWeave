// Investigation tool: bounded independent P/SV scan, not an absence certificate.
#include "dispweave/forward.hpp"
#include "../tests/independent_reference.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
int main(int argc,char** argv) {try {
    if(argc<5 || argc>7) throw std::invalid_argument("Usage: rj_forward_audit FAILED_MODELS VpVs RHO FREQUENCY [MAX_MODELS] [SCAN_SEGMENTS]");
    std::ifstream input(argv[1]);if(!input) throw std::invalid_argument("Cannot open model file");
    const double ratio=std::stod(argv[2]),rho=std::stod(argv[3]),frequency=std::stod(argv[4]);
    const int limit=argc>5?std::stoi(argv[5]):20;
    const int segments=argc>6?std::stoi(argv[6]):5000;
    if(limit<1 || segments<2) throw std::invalid_argument("Positive model limit and >=2 scan segments required");
    std::string line;int checked=0;dw::InvLMForward backend;
    std::cout<<std::setprecision(17)<<"# evaluation K backend_status phase independent_roots scan_failures roots...\n";
    while(checked<limit && std::getline(input,line)) {
        if(line.empty() || line[0]=='#') continue;
        std::istringstream row(line);int id=0,k=0;std::string status;
        if(!(row>>id>>status>>k)) throw std::invalid_argument("Invalid model header");
        if(k<1 || k>10000) throw std::invalid_argument("Invalid dimension");
        dw::LayeredModel m;m.depth.resize(k);m.vs.resize(k);m.depth[0]=0;
        for(int j=1;j<k;++j) row>>m.depth[j];for(int j=0;j<k;++j) row>>m.vs[j];
        if(!row) throw std::invalid_argument("Invalid model row");
        m.vp=ratio*m.vs;m.rho=Eigen::VectorXd::Constant(k,rho);m.validate();
        const auto result=backend.evaluate(m,{{frequency,0}});
        const double lo=.5*m.vs.minCoeff(),hi=m.vs[k-1]*(1-1e-8);
        double previous=NAN,left=lo;int failures=0;std::vector<double> roots;
        for(int j=0;j<=segments;++j) {
            const double c=lo+(hi-lo)*j/static_cast<double>(segments);double value=NAN;
            try {value=independent_secular(m,frequency,c);} catch(const std::runtime_error&) {++failures;}
            if(std::isfinite(value) && std::isfinite(previous) && value*previous<0) {
                try {roots.push_back(reference_root(m,frequency,left,c));} catch(const std::runtime_error&) {++failures;}
            }
            previous=value;left=c;
        }
        std::cout<<id<<' '<<k<<' '<<dw::to_string(result.rows[0].status)<<' '<<result.rows[0].phase_velocity<<' '<<roots.size()<<' '<<failures;
        for(double c:roots) std::cout<<' '<<c;std::cout<<'\n';++checked;
    }
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
