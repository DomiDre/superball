use gauss_quad::{GaussHermite, GaussLegendre};
use ndarray::Array1;

const PI_2: f64 = 1.5707963267948966;
const SQ_2: f64 = 1.4142135623730951;
const FRAC_SQ_PI: f64 = 0.56418958354775628;

/// Formfactor Amplitude F of a Superball Particle
fn amplitude(qx: f64, qy: f64, qz: f64, r: f64, p: f64, gl_quad: &GaussLegendre
             ) -> f64 {
    let two_p = 2.0 * p;
    let inv_two_p = 1.0 / two_p;
    let integral = gl_quad.integrate(
        0.0, 1.0,
        |x| (r*qx*x).cos() * gl_quad.integrate(
            0.0, (1.0 - x.powf(two_p)).powf(inv_two_p),
            |y| (r*qy*y).cos() * (r*qz*
                (1.0 - x.powf(two_p) - y.powf(two_p)).powf(inv_two_p)
            ).sin()
        )
    );
    8.0*r.powi(2) / qz * integral
}

/// Inner integral of orientation integral
///
/// Calculation of cosine and sine of theta is put before the inner integral
fn theta_integral(q: f64, r: f64, p: f64, theta: f64, gl_quad: &GaussLegendre) -> f64 {
    let cos_theta = theta.cos();
    let sin_theta = theta.sin();
    gl_quad.integrate(0.0, PI_2, |phi| {
        amplitude(
            q * phi.cos() * sin_theta,
            q * phi.sin() * sin_theta,
            q * cos_theta,
            r, p, &gl_quad
        )
        .powi(2)
    }) * sin_theta
}

/// Orientation integral.
///
/// Integrate in spherical coordinates over all possible angles. Cube symmetry
/// reduces the integral to the range 0..pi/2
fn orientation_averaged_formfactor(q: f64, r: f64, p: f64, gl_quad: &GaussLegendre) -> f64 {
    gl_quad.integrate(0.0, PI_2, |theta| theta_integral(q, r, p, theta, &gl_quad))
}

/// Size distribution integral.
///
/// The problem is trivially mapped on an integral over exp(-x^2) by a variable
/// transformation, which is solved by a Gauss-Hermite quadrature
fn size_distributed_formfactor(
    q: f64,
    r: f64,
    sigR: f64,
    p: f64,
    gh_quad: &GaussHermite,
    gl_quad: &GaussLegendre,
) -> f64 {
    let integral = gh_quad.integrate(|r_value| {
        orientation_averaged_formfactor(q, r * (SQ_2 * r_value * sigR).exp(),
                                        p, &gl_quad)
    });
    integral * FRAC_SQ_PI
}

/// Formfactor of a cubically shaped particle
///
/// P = N/V * V_p^2 * DeltaSLD^2 * F^2
/// F = sinc(q_x*a/2)*sinc(q_y*a/2)*sinc(q_z*a/2)
/// Additionally a orientation & size distribution average is performed
pub fn formfactor(p: &Array1<f64>, q: &Array1<f64>) -> Array1<f64> {
    let I0 = p[0];
    let r = p[1];
    let sigR = p[2];
    let p_exponent = p[3];
    let SLDparticle = p[4];
    let SLDmatrix = p[5];
    let gl_deg = p[6] as usize;
    let gh_deg = p[7] as usize;

    let I: Array1<f64>;
    let gl_quad = GaussLegendre::init(gl_deg);
    if sigR > 0.0 && gh_deg > 1 {
        let gh_quad = GaussHermite::init(gh_deg);
        I = q.map(|qval| size_distributed_formfactor(*qval, r, sigR, p_exponent, &gh_quad, &gl_quad));
    } else {
        I = q.map(|qval| orientation_averaged_formfactor(*qval, r, p_exponent, &gl_quad));
    }
    I0 * (SLDparticle - SLDmatrix).powi(2) * I
}
