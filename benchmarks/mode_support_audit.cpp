#include "dispweave/mode_support.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <vector>
int main(int argc,char** argv) {
    try {
        if(argc!=3) throw std::runtime_error("Usage: mode_support_audit model.txt frequency_hz");
        std::ifstream file(argv[1]);std::string line;std::vector<double> z,r,v,p;
        while(std::getline(file,line)) {
            if(line.empty() || line[0]=='#') continue;
            std::istringstream in(line);double i,a,b,c,d;
            if(!(in>>i>>a>>b>>c>>d)) throw std::runtime_error("Malformed model table");
            z.push_back(a);r.push_back(b);v.push_back(c);p.push_back(d);
        }
        auto vector=[](const auto& a) {return Eigen::Map<const Eigen::VectorXd>(a.data(),a.size()).eval();};
        dw::LayeredModel m{vector(z),vector(v),vector(p),vector(r)};
        const auto result=dw::trapped_mode_count(m,std::stod(argv[2]));
        std::cout<<std::setprecision(17)<<"{\"resolved\":"<<(result.resolved?"true":"false")
            <<",\"count\":"<<result.count<<",\"evaluations\":"<<result.evaluations
            <<",\"lower_speed\":"<<result.lower_speed<<",\"maximum_phase_increment\":"<<result.maximum_phase_increment
            <<",\"message\":\""<<result.message<<"\"}\n";
        return result.resolved?0:5;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
