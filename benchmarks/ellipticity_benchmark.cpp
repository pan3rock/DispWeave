#include "dispweave/ellipticity.hpp"
#include <chrono>
#include <iostream>
int main() {
    dw::LayeredModel m;
    m.depth=Eigen::Vector3d(0,2,4);m.vs=Eigen::Vector3d(1,2,3);
    m.vp=Eigen::Vector3d(1.8,3.6,5.2);m.rho=Eigen::Vector3d(2,2.2,2.5);
    const std::vector<dw::DispersionQuery> q={{.15,0},{.2,0},{.3,0},{.5,0},{.8,0},
                                            {.3,1},{.5,1},{.8,1},{.5,2},{.8,2}};
    double checksum=0;
    const auto begin=std::chrono::steady_clock::now();
    auto r=dw::rayleigh_ellipticity(m,q);
    const auto first=std::chrono::steady_clock::now();
    for(int i=0;i<1000;++i) {
        r=dw::rayleigh_ellipticity(m,q);
        for(const auto& row:r) {
            if(row.status!=dw::EllipticityStatus::valid) return 1;
            checksum+=row.ratio;
        }
    }
    const auto end=std::chrono::steady_clock::now();
    std::cout<<"first_10_queries_ms="<<std::chrono::duration<double,std::milli>(first-begin).count()
             <<"\nrepeat_10_queries_ms="<<std::chrono::duration<double,std::milli>(end-first).count()/1000
             <<"\nchecksum="<<checksum<<'\n';
}
