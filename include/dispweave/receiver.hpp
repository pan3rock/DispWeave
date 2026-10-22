#pragma once
#include "model.hpp"
#include <complex>
#include <cmath>
#include <string>
#include <vector>
namespace dw {
enum class ReceiverStatus { valid, numerical_failure };
const char* to_string(ReceiverStatus);
struct ReceiverSpectrumRow {
    ReceiverStatus status=ReceiverStatus::numerical_failure;
    std::complex<double> radial={NAN,NAN}, vertical={NAN,NAN};
    double reciprocal_condition=NAN, backward_error=NAN;
    std::string message;
};
// Upgoing unit-displacement P wave at the top of the elastic halfspace.
// x follows horizontal propagation; z is down internally. Output vertical is UP.
// Spectral convention: exp(-i*omega*t), propagation exp(+i*omega*q*h).
// Full coupled P-SV boundary solution, including all conversions/reverberations.
std::vector<ReceiverSpectrumRow> receiver_spectrum(const LayeredModel&,
    const std::vector<double>& frequency_hz,double slowness_s_km);
struct ReceiverOptions {
    double slowness_s_km=.06;
    int samples=1024;
    double dt=.1;
    double gaussian_width=2.5; // rad/s, G(omega)=exp(-omega^2/(4*a^2))
    double water_level=.01;  // fraction of max |Z*S|^2, in (0,1]
    double pre_arrival=5.;   // output time j*dt-pre_arrival
    double time_shift=0.;    // positive shifts the RF toward later output times
    Eigen::VectorXd source; // optional real source samples at t=j*dt; empty = unit impulse
    void validate() const;
};
struct ReceiverResult {
    ReceiverStatus status=ReceiverStatus::numerical_failure;
    Eigen::VectorXd time,radial_rf;
    std::vector<double> frequency_hz;
    std::vector<ReceiverSpectrumRow> spectrum;
    Eigen::VectorXcd source_spectrum,radial_transfer,vertical_transfer;
    double normalization=NAN;
    std::string message;
};
// Gaussian-filtered, water-level R/Z deconvolution, normalized by the vertical
// self-deconvolution peak at zero lag. No frequency bins may be silently dropped.
// Finite DFT window: callers must check window/sample convergence for their data.
ReceiverResult receiver_function(const LayeredModel&,const ReceiverOptions&);
}
