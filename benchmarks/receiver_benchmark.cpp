#include "dispweave/receiver.hpp"
#include <chrono>
#include <iostream>
#include <string>
int main(int argc,char** argv) {
    dw::LayeredModel model;
    const std::string name=argc>1?argv[1]:"layered";
    dw::ReceiverOptions options;
    if(name=="layered") {
        model.depth=Eigen::Vector2d(0,20);model.vs=Eigen::Vector2d(3.5,4.5);
        model.vp=Eigen::Vector2d(6.3,8.1);model.rho=Eigen::Vector2d(2.7,3.3);
    } else if(name=="low_velocity") {
        model.depth=Eigen::Vector3d(0,3,10);model.vs=Eigen::Vector3d(3.4,2.8,4.3);
        model.vp=Eigen::Vector3d(6,5,7.6);model.rho=Eigen::Vector3d(2.6,2.4,3.2);
        options.slowness_s_km=.04;
    } else if(name=="evanescent") {
        model.depth=Eigen::Vector2d(0,.1);model.vs=Eigen::Vector2d(12,4.5);
        model.vp=Eigen::Vector2d(22,8.1);model.rho=Eigen::Vector2d(3.5,3.3);
        options.slowness_s_km=.1;
    } else {std::cerr<<"Expected layered, low_velocity or evanescent\n";return 1;}
    const auto begin=std::chrono::steady_clock::now();
    auto result=dw::receiver_function(model,options);
    const auto first=std::chrono::steady_clock::now();
    if(result.status!=dw::ReceiverStatus::valid) return 2;
    double checksum=result.radial_rf.sum();
    for(int i=0;i<100;++i) {
        result=dw::receiver_function(model,options);
        if(result.status!=dw::ReceiverStatus::valid) return 2;
        checksum+=result.radial_rf.sum();
    }
    const auto end=std::chrono::steady_clock::now();
    std::cout<<"model="<<name<<" samples="<<options.samples
        <<" first_ms="<<std::chrono::duration<double,std::milli>(first-begin).count()
        <<" repeat_ms="<<std::chrono::duration<double,std::milli>(end-first).count()/100
        <<" checksum="<<checksum<<'\n';
}
