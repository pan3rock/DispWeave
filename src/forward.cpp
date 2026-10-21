#include "dispweave/forward.hpp"
#include "dispersion.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <climits>
#include <stdexcept>
namespace dw {
const char* to_string(RootStatus s) {
    switch(s) {
        case RootStatus::valid:return "valid";
        case RootStatus::missing_or_unresolved:return "missing_or_unresolved";
        case RootStatus::numerical_failure:return "numerical_failure";
    }
    return "unknown";
}
bool ForwardResult::all_valid() const {
    return !rows.empty() && std::all_of(rows.begin(),rows.end(),[](const auto& r) {
        return r.status==RootStatus::valid && std::isfinite(r.phase_velocity) && r.phase_velocity>0;
    });
}
ForwardResult InvLMForward::evaluate(const LayeredModel& m,const std::vector<DispersionQuery>& queries) const {
    m.validate();
    if(queries.empty()) throw std::invalid_argument("Empty query list");
    std::map<double,std::vector<size_t>> groups;
    for(size_t i=0;i<queries.size();++i) {
        const auto& q=queries[i];
        if(!std::isfinite(q.frequency_hz) || q.frequency_hz<=0 || q.mode<0 || q.mode>1023)
            throw std::invalid_argument("Require finite frequency>0 and mode in [0,1023]");
        groups[q.frequency_hz].push_back(i);
    }
    Eigen::ArrayXXd native(m.vs.size(),5);
    native.col(0)=Eigen::VectorXd::LinSpaced(m.vs.size(),0,m.vs.size()-1);
    native.col(1)=m.depth; native.col(2)=m.rho; native.col(3)=m.vs; native.col(4)=m.vp;
    fwd::Dispersion dispersion(native,false,method_==SecularMethod::dunkin?fwd::DispMethod::DUNKIN:fwd::DispMethod::FAST_SCALAR);
    ForwardResult result; result.rows.resize(queries.size());
    for(const auto& [frequency,indices]:groups) {
        int wanted=0; for(auto i:indices) wanted=std::max(wanted,queries[i].mode+1);
        std::vector<double> roots; bool issue=false; std::string message;
        try {
            // Guard the upstream double-to-int sample-count conversion before its cap.
            // An upper bound on 2*f*sum(h*vertical_slowness) avoids integer overflow.
            long double sample_bound=0;
            for(Eigen::Index layer=0;layer+1<m.vs.size();++layer)
                sample_bound+=2*static_cast<long double>(frequency)*(m.depth[layer+1]-m.depth[layer])
                    *(1/static_cast<long double>(m.vs[layer])+1/static_cast<long double>(m.vp[layer]));
            if(!std::isfinite(sample_bound) || sample_bound>INT_MAX-2)
                throw std::runtime_error("Frequency/thickness scale exceeds backend sample-count range");
            roots=dispersion.search(frequency,wanted); // Never search_mode: it may synthesize a fallback.
            issue=dispersion.search_diagnostics().numerical_issue;
            for(size_t i=0;i<roots.size();++i)
                if(!std::isfinite(roots[i]) || roots[i]<=0 || roots[i]>=m.vs.tail(1)[0] || (i && roots[i]<=roots[i-1])) issue=true;
        } catch(const std::runtime_error& e) { issue=true; message=e.what(); }
        for(auto i:indices) {
            auto& row=result.rows[i]; row.roots_found=static_cast<int>(roots.size());
            if(issue) {
                row.status=RootStatus::numerical_failure;
                row.message=message.empty()?"Root enumeration reported a numerical issue":message;
            } else if(queries[i].mode>=static_cast<int>(roots.size())) {
                row.message="Requested mode absent or unresolved; absence is not certified";
            } else { row.phase_velocity=roots[queries[i].mode]; row.status=RootStatus::valid; }
        }
    }
    return result;
}
}
