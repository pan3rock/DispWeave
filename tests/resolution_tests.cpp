#include "dispweave/resolution.hpp"
#include "dispweave/objective.hpp"
#include <Eigen/Cholesky>
#include <iostream>
#include <stdexcept>
void check(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
int main(int argc,char** argv) { try {
    std::string name=argc>1?argv[1]:"";
    Eigen::MatrixXd j=Eigen::MatrixXd::Zero(2,3); j(0,0)=3; j(1,1)=4;
    const auto scale=Eigen::VectorXd::Ones(3);
    if(name=="analytic") {
        auto r=dw::local_resolution(j,scale,2);
        check(r.numerical_rank==2 && r.singular_values.isApprox(Eigen::Vector3d(4,3,0)),"underdetermined spectrum");
        Eigen::Matrix3d expected=Eigen::Vector3d(9./13,16./20,0).asDiagonal();
        check(r.resolution.isApprox(expected,1e-13),"analytic ridge matrix and nullspace");
        check(std::abs(r.effective_dimension-expected.trace())<1e-13,"trace dimension");
        auto zero=dw::local_resolution(Eigen::MatrixXd::Zero(2,3),scale,1);
        check(zero.numerical_rank==0 && zero.effective_dimension==0 && zero.resolution.norm()==0,"zero sensitivity");
        j(0,1)=1; j(1,2)=2;
        r=dw::local_resolution(j,scale,.7);
        const Eigen::MatrixXd reference=(j.transpose()*j+.49*Eigen::Matrix3d::Identity()).ldlt().solve(j.transpose()*j);
        check(r.resolution.isApprox(reference,1e-12),"independent linear system reference");
    } else if(name=="invariance") {
        j(0,1)=1; j(1,2)=2;
        auto r=dw::local_resolution(j,scale,1);
        Eigen::Vector3d units(2,3,4);
        auto transformed=dw::local_resolution(j*units.cwiseInverse().asDiagonal(),units,1);
        check(r.resolution.isApprox(transformed.resolution,1e-12),"coordinate units with corresponding metric");
        Eigen::Matrix2d rotation; rotation<<0,-1,1,0;
        auto rotated=dw::local_resolution(rotation*j,scale,1);
        check(r.resolution.isApprox(rotated.resolution,1e-12),"whitening rotation invariance");
        auto noisy=dw::local_resolution(j/4,scale,1);
        auto stronger=dw::local_resolution(j,scale,4);
        check(noisy.resolution.isApprox(stronger.resolution,1e-12) && noisy.effective_dimension<r.effective_dimension,"noise and ridge distinction");
        Eigen::MatrixXd extra(3,3); extra.topRows(2)=j; extra.row(2)<<0,0,2;
        auto added=dw::local_resolution(extra,scale,1);
        check(added.effective_dimension>r.effective_dimension,"independent extra data improve ridge information");
        // Correlated covariance: row permutation must leave the result invariant.
        dw::DispersionObservations data{{{1,0},{2,0}},Eigen::Vector2d(1,1),Eigen::Matrix2d()};
        data.covariance<<4,1,1,9;
        auto correlated=dw::local_resolution(dw::GaussianLikelihood(data).whiten_jacobian(j),scale,1);
        Eigen::Matrix2d swap; swap<<0,1,1,0;
        data.covariance=(swap*data.covariance*swap).eval();
        auto permuted=dw::local_resolution(dw::GaussianLikelihood(data).whiten_jacobian(swap*j),scale,1);
        check(correlated.resolution.isApprox(permuted.resolution,1e-12),"correlated row permutation");
    } else if(name=="validation") {
        for(int k=0;k<5;++k) {
            auto a=j; Eigen::VectorXd s=scale; double ridge=1,tol=1e-12;
            if(k==0) s[0]=0; if(k==1) ridge=0; if(k==2) tol=1;
            if(k==3) a(0,0)=NAN; if(k==4) s.resize(2);
            bool caught=false;
            try { dw::local_resolution(a,s,ridge,tol); } catch(const std::invalid_argument&) { caught=true; }
            check(caught,"invalid input accepted");
        }
        j(0,0)=1e200; j(1,1)=1e-200;
        auto r=dw::local_resolution(j,scale,1);
        check(r.filter_factors.allFinite() && r.filter_factors[0]==1,"large singular values");
    } else throw std::runtime_error("unknown test");
    std::cout<<name<<" passed\n";
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; } }
