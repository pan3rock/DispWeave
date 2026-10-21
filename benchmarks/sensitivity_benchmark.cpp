#include "dispweave/forward.hpp"
#include "../tests/models.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
int main() {
    using Clock=std::chrono::steady_clock;
    std::cout<<"case,method,first_ms,repeat_mean_ms\n"<<std::setprecision(9);
    for(const auto& item:std::vector<std::pair<std::string,dw::LayeredModel>>{{"normal",normal()},{"dense",dense()}}) {
        const auto& [name,m]=item;
        const std::vector<dw::DispersionQuery> q=name=="normal"?std::vector<dw::DispersionQuery>{{.5,0},{.5,1},{.5,2}}:std::vector<dw::DispersionQuery>{{100,0},{100,5},{100,6}};
        for(auto method:{dw::SensitivityMethod::energy,dw::SensitivityMethod::implicit}) {
            dw::InvLMForward backend;
            const auto start=Clock::now();const auto first=backend.evaluate(m,q,method);const auto warm=Clock::now();
            if(!first.all_sensitivities_valid()) return 1;
            for(int i=0;i<100;++i)if(!backend.evaluate(m,q,method).all_sensitivities_valid())return 1;
            const auto finish=Clock::now();
            std::cout<<name<<','<<dw::to_string(method)<<','<<std::chrono::duration<double,std::milli>(warm-start).count()<<','
                     <<std::chrono::duration<double,std::milli>(finish-warm).count()/100<<'\n';
        }
    }
}
