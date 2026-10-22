#include "dispweave/receiver.hpp"
#include "models.hpp"
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <functional>
void check(bool v,const char* msg){if(!v) throw std::runtime_error(msg);}
void rejects(const std::function<void()>& f) {bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}check(caught,"Invalid receiver input accepted");}
int main(int argc,char** argv){try {
    dw::LayeredModel m{v({0}),v({3.5}),v({6.3}),v({2.7})};
    for(double p:{0.,.03,.06,.12}) {
        const double q=std::sqrt(1/(3.5*3.5)-p*p);
        const double expected=2*p*q/(1/(3.5*3.5)-2*p*p);
        const auto rows=dw::receiver_spectrum(m,{0.,.01,1.,10.},p);
        for(const auto& row:rows) {
            check(row.status==dw::ReceiverStatus::valid,"Halfspace response failed");
            check(std::abs(row.radial/row.vertical-expected)<1e-12,"Independent halfspace R/Z mismatch");
            check(row.backward_error<1e-12,"Boundary residual");
        }
    }
    const dw::LayeredModel layered{v({0,20}),v({3.5,4.5}),v({6.3,8.1}),v({2.7,3.3})};
    const dw::LayeredModel split{v({0,7,20}),v({3.5,3.5,4.5}),v({6.3,6.3,8.1}),v({2.7,2.7,3.3})};
    const auto a=dw::receiver_spectrum(layered,{0.,.1,.2,.5,1.,2.},.06);
    const auto b=dw::receiver_spectrum(split,{0.,.1,.2,.5,1.,2.},.06);
    for(size_t i=0;i<a.size();++i) {
        check(a[i].status==dw::ReceiverStatus::valid && b[i].status==dw::ReceiverStatus::valid,"Layered response failure");
        check(std::abs(a[i].radial-b[i].radial)<1e-11 && std::abs(a[i].vertical-b[i].vertical)<1e-11,"Subdivision invariance");
        std::cout<<i<<" R/Z="<<a[i].radial/a[i].vertical<<" rcond="<<a[i].reciprocal_condition<<'\n';
    }
    // Independent frozen state-space spectra: compare full complex R and Z,
    // not just their ratio (no fitted sign, scale or phase correction).
    check(argc==2,"Need reference fixture directory");
    const std::filesystem::path fixtures=argv[1];
    double max_error=0;
    for(const char* name:{"homogeneous_cap","layered","low_velocity","evanescent"}) {
        std::ifstream input(fixtures/(std::string(name)+"-input.txt"));
        int layers=0,count=0;double slow=0;input>>layers>>count>>slow;
        check(input.good() && layers>0 && count>0,"Missing reference input");
        dw::LayeredModel ref;ref.depth.resize(layers);ref.vs.resize(layers);ref.vp.resize(layers);ref.rho.resize(layers);
        double depth=0;
        for(int j=0;j<layers;++j) {double h=0;input>>h>>ref.vs[j]>>ref.vp[j]>>ref.rho[j];ref.depth[j]=depth;depth+=h;}
        std::vector<double> frequencies(count);for(double& f:frequencies) input>>f;
        check(!input.fail(),"Malformed reference model");
        const auto actual=dw::receiver_spectrum(ref,frequencies,slow);
        std::ifstream expected(fixtures/(std::string(name)+".txt"));std::string line;int index=0;
        while(std::getline(expected,line)) {
            if(line.empty() || line[0]=='#') continue;
            double f,rr,ri,zr,zi;std::istringstream row(line);row>>f>>rr>>ri>>zr>>zi;
            check(!row.fail() && index<count,"Malformed reference spectrum");
            check(actual[index].status==dw::ReceiverStatus::valid,"Reference case spectrum failed");
            const double error=std::max(std::abs(actual[index].radial-std::complex<double>(rr,ri)),std::abs(actual[index].vertical-std::complex<double>(zr,zi)));
            max_error=std::max(max_error,error);if(error>=2e-11) std::cerr<<name<<" f="<<f<<" actual="<<actual[index].radial<<","<<actual[index].vertical<<" reference="<<rr<<","<<ri<<","<<zr<<","<<zi<<" error="<<error<<std::endl;check(error<2e-11,"Independent state-space spectrum mismatch");++index;
        }
        check(index==count,"Incomplete independent reference");
    }
    std::cout<<"Independent state-space maximum complex response error="<<max_error<<'\n';
    dw::ReceiverOptions options;options.samples=512;options.dt=.05;options.pre_arrival=5;
    const auto pulse=dw::receiver_function(m,options);check(pulse.status==dw::ReceiverStatus::valid,"Halfspace RF failed");
    const double p=options.slowness_s_km,q=std::sqrt(1/(3.5*3.5)-p*p);
    const double ratio=2*p*q/(1/(3.5*3.5)-2*p*p);
    for(int j=0;j<options.samples;++j)
        check(std::abs(pulse.radial_rf[j]-ratio*std::exp(-std::pow(options.gaussian_width*pulse.time[j],2)))<1e-11,"Independent Gaussian halfspace RF mismatch");
    options.time_shift=.35;
    const auto shifted=dw::receiver_function(m,options);check(shifted.status==dw::ReceiverStatus::valid,"Shift failed");
    for(int j=0;j<options.samples;++j)
        check(std::abs(shifted.radial_rf[j]-ratio*std::exp(-std::pow(options.gaussian_width*(shifted.time[j]-.35),2)))<1e-11,"RF time shift sign/normalization");
    options.time_shift=0;options.source=Eigen::VectorXd::Zero(options.samples);options.source[0]=1;options.source[2]=.3;
    const auto source=dw::receiver_function(layered,options);options.source*=-7;
    const auto scaled=dw::receiver_function(layered,options);
    check(source.status==dw::ReceiverStatus::valid && scaled.status==dw::ReceiverStatus::valid,"Source RF failure");
    check((source.radial_rf-scaled.radial_rf).cwiseAbs().maxCoeff()<1e-11,"Common source amplitude should cancel");
    // Independent ray-theory arrival times, including free-surface multiples.
    options.source.resize(0);options.samples=512;options.dt=.1;
    const auto reference=dw::receiver_function(layered,options);
    check(reference.status==dw::ReceiverStatus::valid,"Layered waveform failed");
    const double qp=std::sqrt(1/(6.3*6.3)-p*p),qs=std::sqrt(1/(3.5*3.5)-p*p);
    const double times[]={20*(qs-qp),20*(qs+qp),40*qs};
    for(int phase=0;phase<3;++phase) {
        int best=-1;const double sign=phase==2?-1.:1.;
        for(int j=0;j<options.samples;++j) if(std::abs(reference.time[j]-times[phase])<.5)
            if(best<0 || sign*reference.radial_rf[j]>sign*reference.radial_rf[best]) best=j;
        check(best>=0 && sign*reference.radial_rf[best]>.01,"Ps/PpPs/PsPs multiple absent or wrong polarity");
        check(std::abs(reference.time[best]-times[phase])<.15,"Ps/PpPs/PsPs arrival time mismatch");
        std::cout<<"phase="<<phase<<" analytic="<<times[phase]<<" peak="<<reference.time[best]<<'\n';
    }
    options.samples=1024;options.dt=.05;
    const auto fine=dw::receiver_function(layered,options);
    check(fine.status==dw::ReceiverStatus::valid,"Fine sampling failed");
    double sampling_error=0;
    for(int j=0;j<512;++j) sampling_error=std::max(sampling_error,std::abs(reference.radial_rf[j]-fine.radial_rf[2*j]));
    check(sampling_error<1e-8,"RF does not converge under dt refinement");
    options.dt=.1;
    const auto longer=dw::receiver_function(layered,options);
    check(longer.status==dw::ReceiverStatus::valid,"Long-window response failed");
    // Compare the declared interpretation window, -5 to 20 s, not the wrapped tail.
    const double short_error=(reference.radial_rf.head(251)-longer.radial_rf.head(251)).cwiseAbs().maxCoeff();
    options.samples=2048;
    const auto longest=dw::receiver_function(layered,options);
    check(longest.status==dw::ReceiverStatus::valid,"Extended-window response failed");
    const double window_error=(longer.radial_rf.head(251)-longest.radial_rf.head(251)).cwiseAbs().maxCoeff();
    check(window_error<1e-5,"RF does not converge under window extension");
    std::cout<<"dt_refinement_error="<<sampling_error<<" short_window_error="<<short_error<<" window_extension_error="<<window_error<<'\n';
    // At f=0, every finite-layer propagator is identity. Surface displacement
    // must equal that of the underlying halfspace regardless of the cap.
    const dw::LayeredModel low{v({0,3,10}),v({3.4,2.8,4.3}),v({6.,5.,7.6}),v({2.6,2.4,3.2})};
    const dw::LayeredModel underlying{v({0}),v({4.3}),v({7.6}),v({3.2})};
    const auto zero=dw::receiver_spectrum(low,{0.},.04)[0],half=dw::receiver_spectrum(underlying,{0.},.04)[0];
    check(zero.status==dw::ReceiverStatus::valid && half.status==dw::ReceiverStatus::valid,"Zero-frequency limit failed");
    check(std::abs(zero.radial-half.radial)<1e-12 && std::abs(zero.vertical-half.vertical)<1e-12,"Zero-frequency halfspace limit");
    auto critical=layered;critical.vp[0]=10;critical.vs[0]=5;
    const auto failed=dw::receiver_spectrum(critical,{0,.1,1.},.1);
    for(const auto& row:failed) check(row.status==dw::ReceiverStatus::numerical_failure && !row.message.empty(),"Critical basis silently regularized");
    for(double slow:{std::nextafter(.1,0.),std::nextafter(.1,1.)})
        check(dw::receiver_spectrum(critical,{1.},slow)[0].status==dw::ReceiverStatus::numerical_failure,"Critical floating-point neighbor accepted");
    options.source.resize(0);options.slowness_s_km=.1;
    const auto bad_rf=dw::receiver_function(critical,options);
    check(bad_rf.status==dw::ReceiverStatus::numerical_failure && bad_rf.radial_rf.size()==options.samples && !bad_rf.radial_rf.allFinite(),"Failed frequency bins deleted");
    // Deliberate conditioning fault, not a geophysical density claim: the
    // zero-frequency bin fails while other bins retain their valid response.
    auto ill_conditioned=layered;ill_conditioned.rho[0]=1e14;
    const auto mixed=dw::receiver_spectrum(ill_conditioned,{0.,.15625},.06);
    check(mixed[0].status==dw::ReceiverStatus::numerical_failure && mixed[0].reciprocal_condition<1e-12,"Ill-conditioned solve accepted");
    check(mixed[1].status==dw::ReceiverStatus::valid,"Valid frequency lost after another bin failed");
    dw::ReceiverOptions mixed_options;mixed_options.samples=64;mixed_options.pre_arrival=1.;
    mixed_options.source=Eigen::VectorXd::Zero(64);mixed_options.source[2]=1.;
    const auto mixed_rf=dw::receiver_function(ill_conditioned,mixed_options);
    check(mixed_rf.status==dw::ReceiverStatus::numerical_failure && !mixed_rf.radial_rf.allFinite(),"RF silently dropped failed bin");
    check(std::abs(mixed_rf.source_spectrum[1]-std::polar(1.,4*std::acos(-1.)/64))<1e-12,"Source spectrum was lost on physical failure");
    rejects([&]{dw::receiver_spectrum(m,{},.06);});
    rejects([&]{dw::receiver_spectrum(m,{NAN},.06);});
    rejects([&]{dw::receiver_spectrum(m,{1.},1./6.3);});
    options.samples=7;rejects([&]{dw::receiver_function(m,options);});
    options.samples=512;options.water_level=0;rejects([&]{dw::receiver_function(m,options);});
    options.water_level=.01;options.source=Eigen::VectorXd::Zero(512);rejects([&]{dw::receiver_function(m,options);});
    std::cout<<"Receiver halfspace and subdivision passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
