#include "dispweave/forward.hpp"
#include "../tests/models.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
int main(int argc,char** argv) {
    if(argc>2 || (argc==2 && std::string(argv[1])!="normal" && std::string(argv[1])!="dense_low_velocity")) {
        std::cerr<<"Usage: forward_benchmark [normal|dense_low_velocity]\n"; return 1;
    }
    using Clock=std::chrono::steady_clock;
    std::cout<<"case,queries,first_ms,repeat_mean_ms,valid,unresolved,numerical\n"<<std::setprecision(8);
    for(const auto& item:std::vector<std::pair<std::string,dw::LayeredModel>>{{"normal",normal()},{"dense_low_velocity",dense()}}) {
        const auto& [name,m]=item;
        if(argc==2 && name!=argv[1]) continue;
        dw::InvLMForward f; std::vector<dw::DispersionQuery> q;
        for(double frequency: name=="normal"?std::vector<double>{.1,.3,.5,1,2}:std::vector<double>{10,30,50,80,100})
            for(int mode=0;mode<8;++mode) q.push_back({frequency,mode});
        const auto start=Clock::now(); const auto first=f.evaluate(m,q); const auto warm=Clock::now();
        for(int i=0;i<100;++i) f.evaluate(m,q);
        const auto finish=Clock::now(); int valid=0,unresolved=0,numerical=0;
        for(const auto& row:first.rows) {
            if(row.status==dw::RootStatus::valid) ++valid;
            else if(row.status==dw::RootStatus::missing_or_unresolved) ++unresolved;
            else ++numerical;
        }
        std::cout<<name<<','<<q.size()<<','<<std::chrono::duration<double,std::milli>(warm-start).count()<<','
         <<std::chrono::duration<double,std::milli>(finish-warm).count()/100<<','<<valid<<','<<unresolved<<','<<numerical<<'\n';
    }
}
