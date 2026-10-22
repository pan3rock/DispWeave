"""Independent, validation-only displacement/traction matrix exponential.

No DispWeave code or wave-amplitude boundary assembly is called. Integrate
dy/dz = i*omega*A*y for y=(ux, uz, txz/(i*omega), tzz/(i*omega)), derived
directly from isotropic stress/strain and momentum balance. The zero-frequency
value is the continuous limit. This growing propagator is suitable only for
moderate optical thickness; it is deliberately not a production solver.
"""
import numpy as np
from scipy.linalg import expm


def spectrum(model, p, frequencies):
    """Rows: h(km), Vs, Vp(km/s), rho(g/cm^3); last row is halfspace."""
    model = np.asarray(model)
    matrices = []
    for h, vs, vp, rho in model[:-1]:
        mu, longitudinal = rho*vs**2, rho*vp**2
        lam = longitudinal-2*mu
        a = np.array([[0, -p, 1/mu, 0],
                      [-lam*p/longitudinal, 0, 0, 1/longitudinal],
                      [rho-4*mu*(lam+mu)/longitudinal*p*p, 0, 0, -lam*p/longitudinal],
                      [0, rho, -p, 0]])
        matrices.append((h, a))
    _, vs, vp, rho = model[-1]
    mu, lam = rho*vs**2, rho*(vp**2-2*vs**2)
    qp, qs = np.sqrt(1/vp**2-p*p+0j), np.sqrt(1/vs**2-p*p+0j)
    # Only halfspace radiation vectors are needed. Finite layers are integrated
    # in physical state space, without forming P/SV eigenvectors or interfaces.
    down_p = vp*np.array([p, qp, 2*mu*p*qp, lam/vp**2+2*mu*qp**2])
    down_s = vs*np.array([-qs, p, mu*(p*p-qs*qs), 2*mu*p*qs])
    up_p = vp*np.array([p, -qp, -2*mu*p*qp, lam/vp**2+2*mu*qp**2])
    out = []
    for f in frequencies:
        propagator = np.eye(4, dtype=complex)
        for h, a in matrices:
            propagator = expm(2j*np.pi*f*h*a) @ propagator
        boundary = np.column_stack((propagator[:, :2], -down_p, -down_s))
        solution = np.linalg.solve(boundary, up_p)
        out.append([f, solution[0].real, solution[0].imag,
                    -solution[1].real, -solution[1].imag])
    return np.array(out)
