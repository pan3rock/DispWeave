#include "dispweave/nuts.hpp"
#include "../src/nuts_tree.hpp"
#include <Eigen/Cholesky>
#include <iostream>
#include <stdexcept>
using namespace dw;
void require(bool b,const char* message) {if(!b) throw std::runtime_error(message);}
// Same predeclared gates as HMC: 60 batch means, max(.04,6 MCSE), MCSE < .1.
void moment(const Eigen::VectorXd& values,double truth) {
    const int batches=60,n=values.size()/batches;Eigen::VectorXd means(batches);
    for(int i=0;i<batches;++i) means[i]=values.segment(i*n,n).mean();
    const double se=std::sqrt((means.array()-means.mean()).square().sum()/(batches*(batches-1)));
    std::cout<<" moment "<<values.mean()<<" truth "<<truth<<" MCSE "<<se<<'\n';
    require(se<.1,"Moment MCSE too large");
    require(std::abs(values.mean()-truth)<std::max(.04,6*se),"Moment outside statistical tolerance");
}
int main(int argc,char** argv) {
    try {
        const auto kernel=argc==2 && std::string(argv[1])=="multinomial"?NUTSKernel::multinomial:NUTSKernel::slice;
        LogDensity normal=[](const Eigen::VectorXd& q) {return DensityEvaluation{EvaluationStatus::valid,-.5*q.squaredNorm(),-q};};
        EuclideanMetric identity(Eigen::MatrixXd::Identity(2,2));
        for(auto seed:{17,2026,99173}) {
            NUTSOptions o;o.kernel=kernel;o.seed=seed;o.warmup=1000;o.draws=20000;
            auto r=sample_nuts(normal,Eigen::VectorXd::Zero(2),identity,o);
            const Eigen::VectorXd mean=r.draws.colwise().mean();
            const Eigen::MatrixXd covariance=r.draws.transpose()*r.draws/o.draws;
            std::cout<<seed<<" mean "<<mean.transpose()<<" second moment\n"<<covariance<<'\n';
            for(int j=0;j<2;++j) {moment(r.draws.col(j),0);moment(r.draws.col(j).array().square().matrix(),1);}
            moment((r.draws.col(0).array()*r.draws.col(1).array()).matrix(),0);
            for(int i=o.warmup;i<o.warmup+o.draws;++i) {
                const auto& t=r.transitions[i];
                require(!t.divergent && t.status==EvaluationStatus::valid,"Normal divergence");
                require(t.evaluations<=(1<<t.depth)-1 && t.evaluations>=1,"Tree evaluation budget");
                require(t.step_size==r.step_size,"Sampling adaptation not frozen");
            }
        }
        for(int kind=0;kind<3;++kind) for(auto seed:{17,2026,99173}) {
            Eigen::MatrixXd cov=Eigen::MatrixXd::Identity(2,2);
            if(kind<2) cov<<1,.995,.995,1;
            const Eigen::MatrixXd precision=cov.llt().solve(Eigen::MatrixXd::Identity(2,2));
            LogDensity target=[precision](const Eigen::VectorXd& q) {
                return DensityEvaluation{EvaluationStatus::valid,-.5*q.dot(precision*q),-precision*q};
            };
            if(kind==2) target=[](const Eigen::VectorXd& q) {
                const double y=q[1]-.5*(q[0]*q[0]-1);
                Eigen::VectorXd g(2);g<<-q[0]+q[0]*y,-y;
                return DensityEvaluation{EvaluationStatus::valid,-.5*(q[0]*q[0]+y*y),g};
            };
            NUTSOptions o;o.kernel=kernel;o.seed=seed;o.warmup=2000;o.draws=12000;o.target_accept=kind==2?.99:.95;
            if(kind==1) o.metric_adaptation=MetricAdaptation::dense;
            const auto r=sample_nuts(target,Eigen::VectorXd::Zero(2),kind==0?EuclideanMetric(precision):identity,o);
            Eigen::MatrixXd x=r.draws;
            if(kind<2) x=cov.llt().matrixL().solve(x.transpose()).transpose().eval();
            std::cout<<"kind "<<kind<<" seed "<<seed<<'\n';
            for(int j=0;j<2;++j) {moment(x.col(j),0);moment(x.col(j).array().square().matrix(),kind==2 && j==1?1.5:1);}
            moment((x.col(0).array()*x.col(1).array()).matrix(),0);
            for(int i=o.warmup;i<o.warmup+o.draws;++i) require(!r.transitions[i].divergent,"Distribution divergence");
            if(kind==1) require(r.mass.isApprox(r.metric_updates.back().mass,1e-12),"Final mass frozen");
        }
        // Check the sampled law at a known hard boundary, with repeated states.
        for(int kind:{0,1}) for(auto seed:{17,2026,99173}) {
            LogDensity bounded=[kind](const Eigen::VectorXd& q) {
                if(kind==0 && q[0]<=0) return DensityEvaluation{EvaluationStatus::outside_support,-INFINITY,{}};
                if(kind==1 && q.squaredNorm()>=1) return DensityEvaluation{EvaluationStatus::outside_support,-INFINITY,{}};
                if(kind==1) return DensityEvaluation{EvaluationStatus::valid,0,Eigen::VectorXd::Zero(2)};
                return DensityEvaluation{EvaluationStatus::valid,-.5*q.squaredNorm(),-q};
            };
            Eigen::VectorXd initial=Eigen::VectorXd::Zero(2);initial[0]=.5;
            NUTSOptions o;o.kernel=kernel;o.seed=seed;o.warmup=1000;o.draws=16000;o.target_accept=.8;
            const auto r=sample_nuts(bounded,initial,identity,o);int boundaries=0;
            for(int i=o.warmup;i<o.warmup+o.draws;++i) {
                require(!r.transitions[i].divergent,"Known boundary marked divergent");
                boundaries+=r.transitions[i].status==EvaluationStatus::outside_support;
            }
            require(boundaries>0,"Boundary test never exercised rejection");
            moment(r.draws.col(0),kind==0?std::sqrt(2/std::acos(-1.)):0);
            moment(r.draws.col(1),0);
            moment(r.draws.col(0).array().square().matrix(),kind==0?1:.25);
            moment(r.draws.col(1).array().square().matrix(),kind==0?1:.25);
            if(kind==0) require(r.draws.col(0).minCoeff()>0,"Half-normal support violated");
            else require(r.draws.rowwise().squaredNorm().maxCoeff()<1,"Disk support violated");
        }
        // Exercise recursive tree counts, endpoints, and candidate selection directly.
        {
            std::mt19937_64 rng(91);
            Eigen::VectorXd q=Eigen::VectorXd::Zero(2),p=Eigen::VectorXd::Zero(2);p[0]=1;
            HamiltonianState start{q,p,normal(q)};
            for(int direction:{-1,1}) {
                std::vector<double> positions;
                auto cursor=start;
                for(int j=0;j<16;++j) {
                    cursor=leapfrog(normal,cursor,identity,.01*direction,1).state;
                    positions.push_back(cursor.position[0]);
                }
                int counts[16]={};
                detail::Builder builder{normal,identity,rng,.01,-100.,-.5,1000.};
                for(int trial=0;trial<10000;++trial) {
                    const auto tree=builder.build(start,direction,4);
                    require(tree.go && tree.count==16 && tree.evaluations==16,"Full subtree counts");
                    require(std::abs(tree.left.position[0]-(direction<0?positions.back():positions.front()))<1e-14,"Left endpoint ordering");
                    require(std::abs(tree.right.position[0]-(direction<0?positions.front():positions.back()))<1e-14,"Right endpoint ordering");
                    bool found=false;
                    for(int j=0;j<16;++j) if(std::abs(tree.candidate.position[0]-positions[j])<1e-14) {++counts[j];found=true;break;}
                    require(found,"Candidate not a tree leaf");
                }
                for(int count:counts) require(std::abs(count-625)<6*std::sqrt(10000./16*15/16),"Subtree reservoir not uniform");
            }
            detail::Builder empty{normal,identity,rng,.01,1.,-.5,1000.};
            const auto excluded=empty.build(start,1,4);
            require(excluded.count==0 && excluded.go && excluded.evaluations==16,"Empty slice handling");
            LogDensity broken=[normal](const Eigen::VectorXd& x) {
                if(x[0]>.025) return DensityEvaluation{};
                return normal(x);
            };
            detail::Builder bad{broken,identity,rng,.01,-100.,-.5,1000.};
            const auto stopped=bad.build(start,1,4);
            require(!stopped.go && stopped.divergent && stopped.evaluations==3 && stopped.count==2,"Mid-subtree failure did not stop");
            require(stopped.status!=EvaluationStatus::valid,"Mid-subtree failure hidden");
        }
        NUTSOptions short_run;short_run.kernel=kernel;short_run.warmup=0;short_run.draws=20;short_run.max_depth=1;
        short_run.step_size=1e-6;short_run.adapt_step_size=false;
        const auto a=sample_nuts(normal,Eigen::VectorXd::Zero(2),identity,short_run);
        const auto b=sample_nuts(normal,Eigen::VectorXd::Zero(2),identity,short_run);
        require(a.draws==b.draws,"Seed replay");
        int observed=0;
        short_run.progress=[&](int i,const Eigen::VectorXd& q,const NUTSTransition&) {
            require(i==observed++ && q==a.draws.row(i).transpose(),"Progress callback state/order");
        };
        const auto observed_run=sample_nuts(normal,Eigen::VectorXd::Zero(2),identity,short_run);
        require(observed==short_run.draws && observed_run.draws==a.draws,"Progress changes RNG stream");
        short_run.progress={};
        for(const auto& t:a.transitions) require(t.max_depth_reached && t.evaluations==1,"Depth cap");
        auto adaptive=short_run;adaptive.warmup=200;adaptive.step_size=.1;adaptive.max_depth=8;
        adaptive.adapt_step_size=true;adaptive.metric_adaptation=MetricAdaptation::diagonal;
        const auto prefix=sample_nuts(normal,Eigen::VectorXd::Zero(2),identity,adaptive);
        adaptive.draws=40;
        const auto extended=sample_nuts(normal,Eigen::VectorXd::Zero(2),identity,adaptive);
        require(prefix.mass==extended.mass && prefix.step_size==extended.step_size && prefix.draws==extended.draws.topRows(20),"Draw budget affects adaptation or prefix");
        // Search overhead is accounted for, and re-search occurs after each mass update.
        adaptive.find_initial_step_size=true;adaptive.draws=200;
        for(double initial_step:{1e-6,100.}) {
            adaptive.step_size=initial_step;int calls=0;
            LogDensity counted=[&](const Eigen::VectorXd& x) {++calls;return normal(x);};
            const auto searched=sample_nuts(counted,Eigen::VectorXd::Zero(2),identity,adaptive);
            require(searched.step_size_searches.size()==1+searched.metric_updates.size(),"Missing metric step search");
            int expected=1;
            for(const auto& t:searched.transitions) expected+=t.evaluations;
            for(size_t j=0;j<searched.step_size_searches.size();++j) {
                const auto& search=searched.step_size_searches[j];expected+=search.evaluations;
                require(search.evaluations>=2 && search.evaluations<=50,"Unbounded step search");
                require(search.warmup_iteration==(j?searched.metric_updates[j-1].warmup_iteration:0),"Search schedule");
            }
            require(calls==expected,"Unaccounted target calls");
            auto longer=adaptive;longer.draws+=20;
            const auto extended_search=sample_nuts(normal,Eigen::VectorXd::Zero(2),identity,longer);
            require(searched.draws==extended_search.draws.topRows(adaptive.draws),"Search changes draw prefix");
            for(int i=adaptive.warmup;i<adaptive.warmup+adaptive.draws;++i)
                require(searched.transitions[i].step_size==searched.step_size,"Search during production");
        }
        auto reject=[&](NUTSOptions o) {
            bool rejected=false;
            try {sample_nuts(normal,Eigen::VectorXd::Zero(2),identity,o);}
            catch(const std::invalid_argument&) {rejected=true;}
            require(rejected,"Invalid options accepted");
        };
        auto invalid=short_run;invalid.max_depth=0;reject(invalid);
        invalid=short_run;invalid.max_depth=21;reject(invalid);
        invalid=short_run;invalid.step_size=NAN;reject(invalid);
        invalid=short_run;invalid.draws=0;reject(invalid);
        invalid=short_run;invalid.find_initial_step_size=true;reject(invalid);
        invalid=adaptive;invalid.adapt_step_size=false;invalid.metric_adaptation=MetricAdaptation::none;reject(invalid);
        invalid=short_run;invalid.kernel=static_cast<NUTSKernel>(42);reject(invalid);
        invalid=short_run;invalid.target_accept=1;reject(invalid);
        invalid=adaptive;invalid.warmup=10;reject(invalid);
        invalid=adaptive;invalid.adapt_step_size=false;reject(invalid);
        LogDensity point=[](const Eigen::VectorXd& q) {
            if(q.squaredNorm()>0) return DensityEvaluation{};
            return DensityEvaluation{EvaluationStatus::valid,0,Eigen::VectorXd::Zero(q.size())};
        };
        const auto failed=sample_nuts(point,Eigen::VectorXd::Zero(2),identity,short_run);
        require(failed.draws.isZero(),"Invalid proposals changed state");
        for(const auto& t:failed.transitions) require(t.divergent && t.status!=EvaluationStatus::valid && !t.accepted,"Invalid evaluation hidden");
        auto impossible=adaptive;impossible.step_size=.1;
        bool search_failed=false;
        try {sample_nuts(point,Eigen::VectorXd::Zero(2),identity,impossible);}
        catch(const std::runtime_error& e) {search_failed=std::string(e.what()).find("50 probes")!=std::string::npos;}
        require(search_failed,"Impossible step search did not fail explicitly");
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
