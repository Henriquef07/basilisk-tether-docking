#include "dockingGNC.h"
#include <iostream>
#include <algorithm>
#include <cmath>

dockingGNC::dockingGNC() {
    // MEKF Initialization
    q_est = Eigen::VectorXd::Zero(4); q_est(0) = 1.0; 
    omega_est = Eigen::VectorXd::Zero(3);
    
    P_att = Eigen::MatrixXd::Zero(6,6);
    P_att.diagonal() << 0.01, 0.01, 0.01, 0.1, 0.1, 0.1;
    Q_att = Eigen::MatrixXd::Zero(6,6);
    Q_att.diagonal() << 1e-7, 1e-7, 1e-7, 1e-4, 1e-4, 1e-4;
    R_att = Eigen::MatrixXd::Identity(3,3) * 1e-6;
    
    H_att = Eigen::MatrixXd::Zero(3,6);
    H_att.block(0,0,3,3) = Eigen::MatrixXd::Identity(3,3);

    // EKF Initialization 
    x_kf_est = Eigen::VectorXd::Zero(6);
    P_kf = Eigen::MatrixXd::Identity(6,6);
    Q_kf = Eigen::MatrixXd::Identity(6,6) * 0.0001;
    R_kf = Eigen::MatrixXd::Zero(6,6);
    R_kf.diagonal() << 0.01, 0.01, 0.01, 0.0001, 0.0001, 0.0001;
}

dockingGNC::~dockingGNC() {}

// Utility to find the null space basis of a matrix using SVD
Eigen::MatrixXd dockingGNC::nullSpace(const Eigen::MatrixXd& a, double epsilon) {
    if (a.rows() == 0 || a.cols() == 0) return Eigen::MatrixXd::Identity(a.cols(), a.cols());

    Eigen::BDCSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeFullV);
    // Enforce a strict minimum floor (1e-10) to prevent collapse
    double tolerance = std::max(1e-10, epsilon * std::max(a.cols(), a.rows()) * svd.singularValues().array().abs()(0));
    
    int rank = 0;
    for (int i = 0; i < svd.singularValues().size(); ++i) {
        if (svd.singularValues()(i) > tolerance) {
            rank++;
        }
    }
    
    Eigen::MatrixXd V = svd.matrixV();
    return V.rightCols(V.cols() - rank);
}

namespace {
static int ny_svo = 6;
static Eigen::MatrixXd Cd_svo = Eigen::MatrixXd::Identity(6, 6);

Eigen::MatrixXd localNullSpace(const Eigen::MatrixXd& a, double epsilon) {
    if (a.rows() == 0 || a.cols() == 0) return Eigen::MatrixXd::Identity(a.cols(), a.cols());
    Eigen::BDCSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeFullV);
    double tolerance = std::max(1e-10, epsilon * std::max(a.cols(), a.rows()) * svd.singularValues().array().abs()(0));
    int rank = 0;
    for (int i = 0; i < svd.singularValues().size(); ++i)
        if (svd.singularValues()(i) > tolerance) rank++;
    Eigen::MatrixXd V = svd.matrixV();
    return V.rightCols(V.cols() - rank);
}

Eigen::MatrixXd localPseudoInverse(const Eigen::MatrixXd& a, double epsilon) {
    if (a.rows() == 0 || a.cols() == 0) return Eigen::MatrixXd::Zero(a.cols(), a.rows());
    Eigen::BDCSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
    double tolerance = std::max(1e-10, epsilon * std::max(a.cols(), a.rows()) * svd.singularValues().array().abs()(0));
    Eigen::VectorXd sInv = svd.singularValues();
    for (int i = 0; i < sInv.size(); ++i) sInv(i) = (sInv(i) > tolerance) ? 1.0 / sInv(i) : 0.0;
    return svd.matrixV() * sInv.asDiagonal() * svd.matrixU().adjoint();
}

struct CBFCoreResult {
    double h;                    // the CBF value
    Eigen::Vector3d grad_c_pos;  // dh/d(c_pos)
    bool converged;              // did the inner Newton solve actually reach a minimum?
    int iterations;              // debug- how many Newton steps it took 
};

CBFCoreResult solveCBFCore_KKT(
    const Eigen::MatrixXd& G_pos,
    const Eigen::VectorXd& c_pos,
    const Eigen::MatrixXd& A_geom,
    const Eigen::VectorXd& b_geom,
    const Eigen::Vector3d& p_target,
    const Eigen::Vector3d& radii,
    double gamma,
    int max_iter)
{
    const int ng_ccg = G_pos.cols();
    const int dim = ng_ccg + 3;
    const int num_funcs = ng_ccg + 1;

    // 1-  Minkowski sum with the target ellipsoid (Eq. 23/63)
    Eigen::MatrixXd G_comb = Eigen::MatrixXd::Zero(3, dim);
    G_comb.leftCols(ng_ccg) = G_pos;
    G_comb.rightCols(3).diagonal() = radii;

    //  2 -Eliminate the CCG's own equality constraint A*xi=b (Eq. 79-81)
    Eigen::MatrixXd N_A;
    Eigen::VectorXd xi_base;
    if (A_geom.rows() > 0) {
        Eigen::MatrixXd A_comb = Eigen::MatrixXd::Zero(A_geom.rows(), dim);
        A_comb.leftCols(ng_ccg) = A_geom;
        N_A = localNullSpace(A_comb, 1e-12);
        xi_base = localPseudoInverse(A_comb, 1e-12) * b_geom;
    } else {
        N_A = Eigen::MatrixXd::Identity(dim, dim);
        xi_base = Eigen::VectorXd::Zero(dim);
    }
    Eigen::MatrixXd G_tilde = G_comb * N_A;             // Eq. 87
    Eigen::VectorXd c_tilde = c_pos + G_comb * xi_base; // Eq. 87
    const int dim_eta = N_A.cols();

    // g_j(xi_j) and their gradients wrt xi, given xi directly 
    // f_val definitions as Eq. 84/113/115
    auto evalBlockXi = [&](const Eigen::VectorXd& xi, Eigen::VectorXd& f_val, Eigen::MatrixXd& df_dxi) {
        f_val.resize(num_funcs);
        df_dxi = Eigen::MatrixXd::Zero(num_funcs, dim);
        for (int j = 0; j < ng_ccg; ++j) {
            f_val(j) = 0.5 * xi(j) * xi(j) - 0.5;
            df_dxi(j, j) = xi(j); // 2*0.5*xi
        }
        // Ellipsoid block
        f_val(ng_ccg) = 0.5 * xi.tail(3).squaredNorm() - 0.5;
        df_dxi.row(ng_ccg).tail(3) = xi.tail(3).transpose();
    };

    // f(eta), grad_eta f(eta), Hess_eta f(eta) -Eq. 103/104/106, obtained
    auto evalEta = [&](const Eigen::VectorXd& eta,
                        Eigen::VectorXd& grad_eta, Eigen::MatrixXd& H_eta, Eigen::VectorXd& f_val_out) {
        Eigen::VectorXd xi = xi_base + N_A * eta; // recover xi from eta
        Eigen::VectorXd f_val;
        Eigen::MatrixXd df_dxi;
        evalBlockXi(xi, f_val, df_dxi);
        f_val_out = f_val;

        double max_f = f_val.maxCoeff();
        Eigen::VectorXd exp_terms = (gamma * (f_val.array() - max_f)).exp(); // -ln(G+1) in eq 105 is canceled by ln(G+1) in eq 88
        double sum_exp = exp_terms.sum();
        Eigen::VectorXd w = exp_terms / sum_exp; // pi_j, Eq. 105 sum_exp comes from  ln(-f(eta)) term 
                            // df_dxi is zero-padded outside block j
        Eigen::VectorXd grad_xi = df_dxi.transpose() * w;   // Eq. 104, in xi-space
        grad_eta = N_A.transpose() * grad_xi;                // Eq. 104's left-hand side
        
        Eigen::MatrixXd H_xi = Eigen::MatrixXd::Zero(dim, dim);
        for (int j = 0; j < ng_ccg; ++j) H_xi(j, j) += w(j);
        H_xi.bottomRightCorner(3, 3) += w(ng_ccg) * Eigen::Matrix3d::Identity();

        for (int j = 0; j < num_funcs; ++j) {
            Eigen::VectorXd diff = df_dxi.row(j).transpose() - grad_xi;
            H_xi += gamma * w(j) * (diff * diff.transpose());
        }
        
        H_eta = N_A.transpose() * H_xi * N_A; // Eq. 106, chain rule into eta-space
    };
    /// --------------------------------------------------------------
    // Newton's method on the (eta, lambda) system Eq. 95-96
    Eigen::MatrixXd G_tilde_pinv = localPseudoInverse(G_tilde, 1e-12);
    Eigen::MatrixXd N_Gtilde = localNullSpace(G_tilde, 1e-12);
    Eigen::VectorXd eta_particular = G_tilde_pinv * (p_target - c_tilde); //  Remark 8 / Eq. 108
    const int n_alpha0 = N_Gtilde.cols();
    Eigen::VectorXd alpha0 = Eigen::VectorXd::Zero(n_alpha0);

    auto lseAtAlpha = [&](const Eigen::VectorXd& a) {
        Eigen::VectorXd eta_try = eta_particular + N_Gtilde * a;
        Eigen::VectorXd ge, fv;
        Eigen::MatrixXd He;
        evalEta(eta_try, ge, He, fv);
        double mf = fv.maxCoeff();
        return mf + std::log((gamma * (fv.array() - mf)).exp().sum()) / gamma;
    };

    // alpha0 = argmin_alpha f(...) Eq. 108
    const double max_step_norm = 50.0; 
    double lm_lambda = 1e-3;   // current LM damping level
    const double lm_grow = 4.0, lm_shrink = 3.0;
    const int max_lm_tries = 25; // per outer iteration
    const double grad_tol = 1e-4;       // primary target, relaxed from 1e-9
    const double grad_tol_loose = 1e-2; 
    bool alpha0_ok = (n_alpha0 == 0); 

    if (n_alpha0 > 0) {
        for (int iter0 = 0; iter0 < max_iter; ++iter0) {
            Eigen::VectorXd eta_try = eta_particular + N_Gtilde * alpha0;  // eta(alpha0), Eq. 108
            Eigen::VectorXd grad_eta_try, f_val_try;
            Eigen::MatrixXd H_eta_try;
            evalEta(eta_try, grad_eta_try, H_eta_try, f_val_try);  // f, grad f, Hess f Eq. 88/103/106

            if (!grad_eta_try.allFinite() || !H_eta_try.allFinite() || !f_val_try.allFinite()) {
                break; // bail out, keep the last good alpha0
            }

            Eigen::VectorXd grad_alpha0 = N_Gtilde.transpose() * grad_eta_try; //chain rule into alpha0
            if (grad_alpha0.norm() < grad_tol) { alpha0_ok = true; break; } // stationary in alpha0 -> argmin reached

            Eigen::MatrixXd H_alpha0_base = N_Gtilde.transpose() * H_eta_try * N_Gtilde;  // Hess, alpha0-space
            double diag_scale = std::max(1e-12, H_alpha0_base.diagonal().cwiseAbs().maxCoeff());
            double g0 = lseAtAlpha(alpha0);  // f(alpha0), current Eq. 88

            bool improved = false;
            Eigen::VectorXd alpha0_new = alpha0;
            for (int lm_try = 0; lm_try < max_lm_tries; ++lm_try) {
                Eigen::MatrixXd H_damped = H_alpha0_base;
                H_damped.diagonal().array() += lm_lambda * diag_scale;   //LM damping, scaled to this problem

                Eigen::LDLT<Eigen::MatrixXd> ldlt0(H_damped);
                Eigen::VectorXd delta0 = ldlt0.solve(-grad_alpha0);
                if (!delta0.allFinite()) { lm_lambda *= lm_grow; continue; }

                double dnorm0 = delta0.norm();
                if (dnorm0 > max_step_norm) delta0 *= (max_step_norm / dnorm0); // trust-region cap, still applied

                Eigen::VectorXd candidate = alpha0 + delta0;
                double g_new = lseAtAlpha(candidate);

                if (std::isfinite(g_new) && g_new < g0) {
                    alpha0_new = candidate;
                    improved = true;
                    lm_lambda = std::max(1e-8, lm_lambda / lm_shrink); // success-> trust the model more next time
                    break;
                }
                lm_lambda *= lm_grow; // failed -> distrust the model more, retry THIS iteration with more damping
            }

            if (!improved) { // No damping level improved the objective further 
                if (grad_alpha0.norm() < grad_tol_loose) alpha0_ok = true;
                break;
            }
            alpha0 = alpha0_new;
        }
    }

    Eigen::VectorXd eta = eta_particular + N_Gtilde * alpha0; // Eq. 108 eta0, finished
    Eigen::VectorXd grad_eta_init, f_val_init;
    Eigen::MatrixXd H_eta_init;
    evalEta(eta, grad_eta_init, H_eta_init, f_val_init);
    Eigen::VectorXd lambda = -(G_tilde_pinv.transpose() * grad_eta_init); // Eq. 108 lambda0

    bool converged = false;
    int iters_used = 0;

    // Newton on the joint (eta,lambda) system -> Eq. 95-97.
    for (int iter = 0; iter < max_iter; ++iter) {
        iters_used = iter + 1;
        Eigen::VectorXd grad_eta, f_val;
        Eigen::MatrixXd H_eta;
        evalEta(eta, grad_eta, H_eta, f_val);   // grad f(eta), Hess f(eta)  Eq. 103/106

        if (!grad_eta.allFinite() || !H_eta.allFinite()) break; // bail out, keep last good (eta,lambda)

        Eigen::VectorXd psi(dim_eta + 3);
        psi.head(dim_eta) = grad_eta + G_tilde.transpose() * lambda; // top of Eq. 95: stationarity
        psi.tail(3) = G_tilde * eta + c_tilde - p_target; // bottom of Eq. 95: feasibility

        if (psi.norm() < grad_tol) { converged = true; break; }   // psi=0 solves Eq. 95 exactly (relaxed from 1e-9)

        double diag_scaleJ = std::max(1e-12, H_eta.diagonal().cwiseAbs().maxCoeff());
        double psi_norm0 = psi.norm();

        bool improved = false;
        Eigen::VectorXd eta_new = eta, lambda_new = lambda;
        for (int lm_try = 0; lm_try < max_lm_tries; ++lm_try) {
            Eigen::MatrixXd J(dim_eta + 3, dim_eta + 3);
            J.topLeftCorner(dim_eta, dim_eta) = H_eta;  // Eq. 96, top-left
            J.topRightCorner(dim_eta, 3) = G_tilde.transpose(); // Eq. 96, top-right
            J.bottomLeftCorner(3, dim_eta) = G_tilde;  // Eq. 96, bottom-left
            J.bottomRightCorner(3, 3).setZero(); // Eq. 96, bottom-right
            // LM damping on the Hessian block only
            J.topLeftCorner(dim_eta, dim_eta).diagonal().array() += lm_lambda * diag_scaleJ;

            // J indefinite (Eq. 97's determinant nonzero, but not PD) -> no LDLT/LLT.
            Eigen::PartialPivLU<Eigen::MatrixXd> lu(J);
            Eigen::VectorXd delta = lu.solve(-psi);   // one Newton step on Eq. 95
            if (!delta.allFinite()) { lm_lambda *= lm_grow; continue; }

            double dnorm = delta.norm();
            if (dnorm > max_step_norm) delta *= (max_step_norm / dnorm); // trust-region cap

            Eigen::VectorXd eta_cand = eta + delta.head(dim_eta);
            Eigen::VectorXd lambda_cand = lambda + delta.tail(3);
            Eigen::VectorXd ge, fv;
            Eigen::MatrixXd He;
            evalEta(eta_cand, ge, He, fv);
            if (!ge.allFinite()) { lm_lambda *= lm_grow; continue; }

            Eigen::VectorXd psi_new(dim_eta + 3);
            psi_new.head(dim_eta) = ge + G_tilde.transpose() * lambda_cand;
            psi_new.tail(3) = G_tilde * eta_cand + c_tilde - p_target;

            if (std::isfinite(psi_new.norm()) && psi_new.norm() < psi_norm0) {
                eta_new = eta_cand;
                lambda_new = lambda_cand;
                improved = true;
                lm_lambda = std::max(1e-8, lm_lambda / lm_shrink); // success -> trust the model more next time
                break;
            }
            lm_lambda *= lm_grow; // failed -> distrust the model more, retry THIS iteration with more damping
        }

        if (!improved) { // failing to improve further
            if (psi.norm() < grad_tol_loose) converged = true;
            break;
        }
        eta = eta_new;
        lambda = lambda_new;
    }

    // Final h value at the converged eta -- identical formula to Eq. 88.
    Eigen::VectorXd xi_opt = xi_base + N_A * eta;
    Eigen::VectorXd f_val_opt;
    Eigen::MatrixXd df_dxi_opt;
    evalBlockXi(xi_opt, f_val_opt, df_dxi_opt);
    double max_f_opt = f_val_opt.maxCoeff();
    double sum_exp_opt = (gamma * (f_val_opt.array() - max_f_opt)).exp().sum();
    double h_val = max_f_opt + std::log(sum_exp_opt) / gamma - std::log(num_funcs + 1.0) / gamma; // - ln(G+1)/gamma

    if (!std::isfinite(h_val) || !lambda.allFinite()) {
        converged = false;
    }

    CBFCoreResult result;
    result.h = h_val;
    result.grad_c_pos = lambda;
    result.converged = converged;
    result.iterations = iters_used;
    return result;
}

} 

// Helper to project a vector into a spherical L2 bound (ball)
Eigen::Vector3d projectBall(const Eigen::Vector3d& v, double a) {
    double norm = v.norm();
    if (norm <= a) return v;
    return v * (a / norm);
}

// shared helper: project u0 onto {g·u >= b} ∩ {|u| <= a}
static Eigen::Vector3d projHalfspaceBall(const Eigen::Vector3d& u0, const Eigen::Vector3d& g,
                                         double b, double a, bool& feasible) {
    feasible = true;
    Eigen::Vector3d u = projectBall(u0, a);
    if (g.dot(u) >= b) return u;
    double gn = g.norm();
    double hi = 1.0;
    while (g.dot(projectBall(u0 + hi*g, a)) < b && hi < 1e12) hi *= 2.0;
    if (g.dot(projectBall(u0 + hi*g, a)) < b) {
        feasible = false;
        return (gn > 1e-12) ? Eigen::Vector3d(a * g / gn) : u;   // also guards g = 0
    }
    double lo = 0.0;
    for (int i = 0; i < 100; ++i) {
        double mid = 0.5*(lo+hi);
        (g.dot(projectBall(u0 + mid*g, a)) < b) ? lo = mid : hi = mid;
    }
    return projectBall(u0 + hi*g, a);
}

// Entry point 1: h(p_target) one step ahead (k+1) under a candidate control u_opt
CBFResult dockingGNC::evaluateCBF_Uopt(
    std::vector<double> u_opt_flat,
    std::vector<double> p_target_flat,
    double exclusion_radius,
    double h_k,              
    double alpha_cbf,        
    double step_time,        
    double safety_margin,    
    double max_accel)        
{
    Eigen::VectorXd u_opt = Eigen::Map<Eigen::VectorXd>(u_opt_flat.data(), 3);
    Eigen::Vector3d p_target = Eigen::Map<Eigen::Vector3d>(p_target_flat.data());

    Eigen::MatrixXd Ad_pos = Ad_kf.topRows(3);
    Eigen::MatrixXd Bd_pos = Bd_kf.topRows(3);

    // Predict position at k+1
    Eigen::Vector3d c_pos = Ad_pos * c_X_k_out + Bd_pos * u_opt;
    
    // Spherical CBF and Gradient
    Eigen::Vector3d diff = c_pos - p_target;
    double h_pred = diff.squaredNorm() - (exclusion_radius * exclusion_radius);
    Eigen::Vector3d grad_c_pos = 2.0 * diff; 
    Eigen::Vector3d grad_u = Bd_pos.transpose() * grad_c_pos;

    // Dynamic decay boundary
    double required_h = safety_margin;
    if (std::isfinite(h_k)) {
        double decay_floor = (1.0 - alpha_cbf * step_time) * h_k;
        required_h = std::max(safety_margin, decay_floor);
    }

    bool was_modified = false;

    // Safety Filter with Decay Constraint
    if (h_pred < required_h) {
        if (grad_u.norm() > 1e-6) {
            // Minimum-Norm Projection onto the decaying boundary
            bool feas;
            double b = grad_u.dot(u_opt) + (required_h - h_pred);
            u_opt = projHalfspaceBall(u_opt, grad_u, b, max_accel, feas);
            was_modified = true;
        }
    }

    // Re-evaluate if modified
    if (was_modified) {
        c_pos = Ad_pos * c_X_k_out + Bd_pos * u_opt;
        diff = c_pos - p_target;
        h_pred = diff.squaredNorm() - (exclusion_radius * exclusion_radius);
        grad_c_pos = 2.0 * diff;
        grad_u = Bd_pos.transpose() * grad_c_pos;
    }    

    // Pack 
    CBFResult out;
    out.h = h_pred;
    flattenVector(grad_u, out.grad_u_flat);
    flattenVector(u_opt, out.u_safe_flat);
    out.converged = (h_pred >= required_h - 1e-9);
    out.iterations = 1;
    
    return out;
}

// Entry point 2: Instantaneous safety score
double dockingGNC::evaluateCBF_Current(
    std::vector<double> p_target_flat,
    double exclusion_radius)
{
    Eigen::Vector3d p_target = Eigen::Map<Eigen::Vector3d>(p_target_flat.data());
    Eigen::Vector3d c_pos = c_X_k_out.head(3);

    return (c_pos - p_target).squaredNorm() - (exclusion_radius * exclusion_radius);
}

// Entry point 3: Instantaneous safety score and spatial gradient
CBFResult dockingGNC::evaluateCBF_Gradient(
    std::vector<double> p_target_flat,
    double exclusion_radius)
{
    Eigen::Vector3d p_target = Eigen::Map<Eigen::Vector3d>(p_target_flat.data());
    Eigen::Vector3d c_pos = c_X_k_out.head(3);
    Eigen::Vector3d diff = c_pos - p_target;

    CBFResult out;
    out.h = diff.squaredNorm() - (exclusion_radius * exclusion_radius);
    
    Eigen::Vector3d grad_c_pos = 2.0 * diff;
    flattenVector(grad_c_pos, out.grad_u_flat); // Repurposed field for spatial gradient
    out.converged = true;
    out.iterations = 1;
    
    return out;
}

std::vector<double> dockingGNC::filterQPCBF_3D(
    std::vector<double> u_nom_flat,
    std::vector<double> grad_h_flat,
    double h_val,
    double alpha_cbf,
    double step_time_fast,
    double max_thrust,
    double mass,
    std::vector<double> pos_err_flat, // Relative position (rho)
    std::vector<double> v_curr_flat,  // Relative velocity (rho_dot)
    std::vector<double> rc_flat,      // Chief absolute inertial position
    std::vector<double> vc_flat,      // Chief absolute inertial velocity
    double mu)                        // Earth gravitational parameter
{
    // 1. Map Python flat lists to Eigen 3D Vectors
    Eigen::Vector3d u_nom = Eigen::Map<Eigen::Vector3d>(u_nom_flat.data());
    Eigen::Vector3d grad_h = Eigen::Map<Eigen::Vector3d>(grad_h_flat.data());
    Eigen::Vector3d pos_err = Eigen::Map<Eigen::Vector3d>(pos_err_flat.data());
    Eigen::Vector3d v_curr = Eigen::Map<Eigen::Vector3d>(v_curr_flat.data());
    Eigen::Vector3d rc = Eigen::Map<Eigen::Vector3d>(rc_flat.data());
    Eigen::Vector3d vc = Eigen::Map<Eigen::Vector3d>(vc_flat.data());

    double a_max = max_thrust / mass; // Max physical acceleration limit

    //  Calculate the Non-Linear Orbital Drift (f_nl)
    // First, find the orbital angular velocity vector of the Chief (omega = r x v / |r|^2)
    Eigen::Vector3d h_ang_mom = rc.cross(vc);
    Eigen::Vector3d omega = h_ang_mom / rc.squaredNorm();

    // Calculate exact gravitational pull on both spacecraft
    Eigen::Vector3d r_servicer = rc + pos_err;
    Eigen::Vector3d g_chief = -mu * rc / std::pow(rc.norm(), 3);
    Eigen::Vector3d g_servicer = -mu * r_servicer / std::pow(r_servicer.norm(), 3);
    
    // Differential gravity (how much gravity is pulling them apart)
    Eigen::Vector3d diff_gravity = g_servicer - g_chief;

    // Relative frame fictional forces (Coriolis and Centrifugal)
    Eigen::Vector3d coriolis = -2.0 * omega.cross(v_curr);
    Eigen::Vector3d centrifugal = -omega.cross(omega.cross(pos_err));

    // The true non-linear acceleration acting on the servicer (without thrusters)
    Eigen::Vector3d f_nl = diff_gravity + coriolis + centrifugal;

    //  Formulate the Halfspace boundary (g^T * u >= b)
    // 'g' defines the sensitivity vector
    Eigen::Vector3d g = grad_h * (step_time_fast * step_time_fast / 2.0);
    
    // 'b' bundles all uncontrollable forces. Notice how f_nl is now subtracted from the safety margin!
    double b = -alpha_cbf * step_time_fast * h_val 
               - grad_h.dot(v_curr) * step_time_fast 
               - grad_h.dot(f_nl) * (step_time_fast * step_time_fast / 2.0);

    //  Quick Check: Is the nominal PD controller already safe?
    Eigen::Vector3d u_clip = projectBall(u_nom, a_max);
    
    // If the clamped PD command is safe use it
    if (g.dot(u_clip) >= b || !std::isfinite(h_val)) {
        std::vector<double> out(4);
        for(int i=0; i<3; ++i) out[i] = u_clip(i);
        out[3] = 1.0; // Feasible flag
        return out;
    }

    //  Dual Bisection QP Solver strictly on the Ball intersection
    double hi = 1.0;
    while (g.dot(projectBall(u_nom + hi * g, a_max)) < b && hi < 1e12) {
        hi *= 2.0;
    }
    
    if (g.dot(projectBall(u_nom + hi * g, a_max)) < b) {
        // Infeasible: best effort (push as hard as possible in direction of g)
        double gn = g.norm();
        Eigen::Vector3d best = (gn > 1e-12) ? Eigen::Vector3d(a_max * g / gn) : u_clip;
        std::vector<double> out(4);
        for(int i=0; i<3; ++i) out[i] = best(i);
        out[3] = 0.0; // Infeasible flag
        return out;
    }

    double lo = 0.0;
    for (int iter = 0; iter < 100; ++iter) {
        double mid = 0.5 * (lo + hi);
        if (g.dot(projectBall(u_nom + mid * g, a_max)) < b) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    // Final exact optimal and safe control vector
    Eigen::Vector3d u_opt = projectBall(u_nom + hi * g, a_max);
    
    std::vector<double> out(4);
    for(int i=0; i<3; ++i) out[i] = u_opt(i);
    out[3] = 1.0; // Feasible flag
    return out;
}

// Alg 1 - pre-computation (supports both 6x6 full-state C = I_6 and 3x6 position-only C = [I_3, 0_3])
void dockingGNC::initSVO(std::vector<double> Ad_flat, std::vector<double> Bd_flat, std::vector<double> Cd_flat, double w_max, double v_max, int L_horizon, std::vector<double> x_hat_init_f) {
    svo_horizon = L_horizon;
    const int nx = 6, nu = 3;
    const int ny = static_cast<int>(Cd_flat.size()) / nx; // 6 for full-state, 3 for position-only
    ny_svo = ny;

    Eigen::MatrixXd Ad = Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(Ad_flat.data());
    Eigen::MatrixXd Bd = Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>>(Bd_flat.data());
    Eigen::MatrixXd Cd = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(Cd_flat.data(), ny, nx);
    Cd_svo = Cd;

    if (ny == 6) {
        // Full-state C = I_6 -> place poles at 0.3
        double target_pole = 0.3;
        L_luenberger = Ad - target_pole * Eigen::MatrixXd::Identity(nx, ny);
    } else {
        // Position-only C = [I_3, 0_3] -> exact 3x3 block pole placement at z1 = 0.4, z2 = 0.5
        Eigen::Matrix3d A11 = Ad.topLeftCorner(3, 3);
        Eigen::Matrix3d A12 = Ad.topRightCorner(3, 3);
        Eigen::Matrix3d A21 = Ad.bottomLeftCorner(3, 3);
        Eigen::Matrix3d A22 = Ad.bottomRightCorner(3, 3);
        Eigen::Matrix3d A12_inv = A12.inverse();
        Eigen::Matrix3d A22_tilde = A12 * A22 * A12_inv;

        double z1 = 0.4, z2 = 0.5;
        Eigen::Matrix3d M1 = (z1 + z2) * Eigen::Matrix3d::Identity() - A22_tilde;
        Eigen::Matrix3d L1 = A11 - M1;
        Eigen::Matrix3d L2 = A21 - A12_inv * (A22_tilde * M1 - (z1 * z2) * Eigen::Matrix3d::Identity());

        L_luenberger.resize(nx, 3);
        L_luenberger.topRows(3) = L1;
        L_luenberger.bottomRows(3) = L2;
    }

    Eigen::MatrixXd F_LC = Ad - L_luenberger * Cd;

    // Solve discrete Lyapunov equation: (F-LC)^T * P * (F-LC) - P = -I
    Eigen::MatrixXd P_lyap = Eigen::MatrixXd::Identity(nx, nx);
    Eigen::MatrixXd F_LC_T = F_LC.transpose();
    for (int iter = 0; iter < 1000; ++iter) {
        Eigen::MatrixXd P_next = F_LC_T * P_lyap * F_LC + Eigen::MatrixXd::Identity(nx, nx);
        if ((P_next - P_lyap).norm() < 1e-12) break;
        P_lyap = P_next;
    }

    Eigen::MatrixXd P_sqrt = Eigen::LLT<Eigen::MatrixXd>(P_lyap).matrixU();
    P_sqrt_inv = P_sqrt.inverse();

    Eigen::MatrixXd Gw = Eigen::MatrixXd::Zero(nx, nx);
    Gw.leftCols(3) = Bd * w_max;
    Eigen::MatrixXd Gv = Eigen::MatrixXd::Identity(ny, ny) * v_max;

    // Calculate decrease rate and Error Bounds
    a_decay = Eigen::JacobiSVD<Eigen::MatrixXd>(P_sqrt * F_LC * P_sqrt_inv).singularValues()(0);
    e_init = 0.0;
    Eigen::MatrixXd G_init = Eigen::MatrixXd::Identity(nx, nx) * v_max;
    for (int m = 0; m < (1 << nx); ++m) {
        Eigen::VectorXd s(nx);
        for (int i = 0; i < nx; ++i) s(i) = ((m >> i) & 1) ? 1.0 : -1.0;
        e_init = std::max(e_init, (P_sqrt * G_init * s).norm());
    }
    e_noise = 0.0;
    for (int mw = 0; mw < (1 << 3); ++mw) {
        for (int mv = 0; mv < (1 << ny); ++mv) {
            Eigen::VectorXd sw(3), sv(ny);
            for (int i = 0; i < 3; ++i)  sw(i) = ((mw >> i) & 1) ? 1.0 : -1.0;
            for (int i = 0; i < ny; ++i) sv(i) = ((mv >> i) & 1) ? 1.0 : -1.0;
            e_noise = std::max(e_noise, (P_sqrt * (Bd * (w_max * sw) - L_luenberger * Gv * sv)).norm());
        }
    }

    x_hat = Eigen::Map<Eigen::VectorXd>(x_hat_init_f.data(), nx);
    k = 0;

    // Algorithm 1
    Eigen::MatrixXd Gl = Eigen::MatrixXd::Zero(nx, 0);
    Eigen::MatrixXd Gl0 = P_sqrt_inv;
    Eigen::MatrixXd Al = Eigen::MatrixXd::Zero(0, 0);
    Eigen::MatrixXd Al0 = Eigen::MatrixXd::Zero(0, nx);
    Eigen::VectorXd cl = Eigen::VectorXd::Zero(nx);
    Eigen::MatrixXd clU = Eigen::MatrixXd::Zero(nx, 0);
    Eigen::MatrixXd cl0 = Eigen::MatrixXd::Identity(nx, nx);
    Eigen::VectorXd bl = Eigen::VectorXd::Zero(0);
    Eigen::MatrixXd blU = Eigen::MatrixXd::Zero(0, 0);
    Eigen::MatrixXd bl0 = Eigen::MatrixXd::Zero(0, nx);

    for (int l = 0; l < L_horizon; ++l) {
        int ng_l = Gl.cols(), ng_l0 = Gl0.cols(), nc_l = Al.rows();

        Eigen::MatrixXd Gl_next(nx, nx + ny + ng_l);
        Gl_next << Gw, Eigen::MatrixXd::Zero(nx, ny), Ad * Gl;

        Eigen::MatrixXd Gl0_next = Ad * Gl0;
        Eigen::VectorXd cl_next = Ad * cl;

        Eigen::MatrixXd clU_next(nx, nu + clU.cols());
        clU_next << Bd, Ad * clU;

        Eigen::MatrixXd cl0_next = Ad * cl0;

        int new_nc = ny + nc_l;
        int new_ng = nx + ny + ng_l;
        Eigen::MatrixXd Al_next = Eigen::MatrixXd::Zero(new_nc, new_ng);
        Al_next.block(0, nx, ny, ny) = Gv;
        if (ng_l > 0) Al_next.block(0, nx + ny, ny, ng_l) = Cd * Gl;
        if (nc_l > 0 && ng_l > 0) Al_next.block(ny, nx + ny, nc_l, ng_l) = Al;

        Eigen::MatrixXd Al0_next(ny + Al0.rows(), ng_l0);
        Al0_next.block(0, 0, ny, ng_l0) = Cd * Gl0;
        if (Al0.rows() > 0) Al0_next.block(ny, 0, Al0.rows(), ng_l0) = Al0;

        Eigen::VectorXd bl_next(ny + bl.size());
        bl_next.head(ny) = -(Cd * cl);
        bl_next.tail(bl.size()) = bl;

        int old_bU_cols = blU.cols();
        Eigen::MatrixXd blU_next = Eigen::MatrixXd::Zero(ny + nc_l, nu + old_bU_cols);
        blU_next.block(0, nu, ny, old_bU_cols) = Cd * clU;
        if (nc_l > 0 && old_bU_cols > 0) blU_next.block(ny, nu, nc_l, old_bU_cols) = blU;

        Eigen::MatrixXd bl0_next(ny + bl0.rows(), nx);
        bl0_next.block(0, 0, ny, nx) = Cd * cl0;
        if (bl0.rows() > 0) bl0_next.block(ny, 0, bl0.rows(), nx) = bl0;

        Gl = Gl_next; Gl0 = Gl0_next;
        Al = Al_next; Al0 = Al0_next;
        cl = cl_next; clU = clU_next; cl0 = cl0_next;
        bl = bl_next; blU = blU_next; bl0 = bl0_next;
    }

    G = Gl; G0 = Gl0; A = Al; A0 = Al0;
    c = cl; c_U = clU; c0 = cl0; b = bl; b_U = blU; b0 = bl0;

    double tail = (a_decay < 1.0 - 1e-10) ? (e_noise / (1.0 - a_decay)) : e_noise;
    int ng_p = G.cols(), ng_e = G0.cols(), nc_p = A.rows();

    Eigen::MatrixXd G_X_inf(nx, ng_p + ng_e);
    G_X_inf << G, tail * G0;

    Eigen::MatrixXd A_X_inf(nc_p, ng_p + ng_e);
    if (nc_p > 0) {
        A_X_inf.leftCols(ng_p) = A;
        A_X_inf.rightCols(ng_e) = tail * A0;
    }

    Z_X = G_X_inf * pseudoInverse(A_X_inf, 1e-12);

    x_hat = Eigen::Map<Eigen::VectorXd>(x_hat_init_f.data(), nx);
    k = 0;

    if (L_horizon < 0) {
        // Recursive CCG baseline initialization: X_0 = c_0 + e_init * P_sqrt_inv * B_inf
        G_X_k = e_init * P_sqrt_inv;
        c_X_k_out = x_hat;
        A_X_k.resize(0, G_X_k.cols());
        b_X_k_out.resize(0);
        G = Gw; // Store process noise generator matrix for recursive step
        A = Gv; // Store measurement noise generator matrix for recursive step
        return;
    }

}

// Explicit finite-horizon observer
CCG_Tuple dockingGNC::updateSVO(std::vector<double> U_hist_flat, std::vector<double> Y_hist_flat) {
    const int nx = 6, nu = 3, ny = ny_svo;
    const int hist_cols = std::max(1, svo_horizon);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> U_hist(U_hist_flat.data(), nu, hist_cols);
    // Y_hist_flat from Python always has 6 rows; slice top 'ny' rows (6 for full state, 3 for position-only)
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Y_hist_full(Y_hist_flat.data(), 6, hist_cols);

    Eigen::VectorXd u_oldest = U_hist.col(0);
    Eigen::VectorXd y_oldest = Y_hist_full.col(0).head(ny);
    k++;

    double tail = (a_decay < 1.0 - 1e-10) ? (e_noise / (1.0 - a_decay)) : e_noise;
    Eigen::VectorXd x_est;
    Eigen::VectorXd x_curr = x_hat;
    CCG_Tuple out;

    if (svo_horizon < 0) {
        // True Recursive CCG Baseline (no horizon reset: grows as n_g = 6 + 12*(k-1), n_c = 6*(k-1))
        Eigen::VectorXd u_prev = U_hist.col(hist_cols - 1);
        Eigen::VectorXd y_prev = Y_hist_full.col(hist_cols - 1).head(ny);

        if (k == 1) {
            // At k=1, U_hist and Y_hist are still unpopulated (zeros); keep initial state
            x_est = c_X_k_out;
            out.ng_p = 0;
            out.ng_e = G_X_k.cols();
        } else {
            // If k > 20 (n_g = 246 > N=15), drop oldest (nx+ny) cols and ny rows so x_est never freezes
            // and solveCBFCore_KKT does not hang the 850-step simulation
            const int k_cap = 20;
            if (k > k_cap + 1) {
                int keep_cols = G_X_k.cols() - (nx + ny);
                int keep_rows = A_X_k.rows() - ny;
                G_X_k = G_X_k.leftCols(keep_cols).eval();
                A_X_k = A_X_k.topLeftCorner(keep_rows, keep_cols).eval();
                b_X_k_out = b_X_k_out.head(keep_rows).eval();
            }

            int ng_old = G_X_k.cols();
            int nc_old = A_X_k.rows();

            // 1. Intersect X_{k-1} with y_{k-1} (measured before u_{k-1} was applied, matching initSVO)
            Eigen::MatrixXd A_new = Eigen::MatrixXd::Zero(ny + nc_old, nx + ny + ng_old);
            A_new.block(0, nx, ny, ny) = A; // Gv
            A_new.block(0, nx + ny, ny, ng_old) = Cd_svo * G_X_k;
            if (nc_old > 0) {
                A_new.block(ny, nx + ny, nc_old, ng_old) = A_X_k;
            }

            Eigen::VectorXd b_new(ny + nc_old);
            b_new.head(ny) = y_prev - Cd_svo * c_X_k_out;
            if (nc_old > 0) {
                b_new.tail(nc_old) = b_X_k_out;
            }

            // 2. Propagate X_{k-1} to X_k under u_{k-1}: x_k = Ad * x_{k-1} + Bd * u_{k-1} + Gw * w_{k-1}
            Eigen::MatrixXd G_pred(nx, nx + ny + ng_old);
            G_pred << G, Eigen::MatrixXd::Zero(nx, ny), Ad_kf * G_X_k;
            Eigen::VectorXd c_pred = Ad_kf * c_X_k_out + Bd_kf * u_prev;

            G_X_k = G_pred;
            c_X_k_out = c_pred;
            A_X_k = A_new;
            b_X_k_out = b_new;

            // Online SVD pseudoinverse required at every step for recursive CCG
            Eigen::MatrixXd Z_rec = G_X_k * pseudoInverse(A_X_k, 1e-12);
            x_est = c_X_k_out + Z_rec * b_X_k_out;

            x_hat = Ad_kf * x_hat + Bd_kf * u_prev + L_luenberger * (y_prev - Cd_svo * x_hat);
            x_curr = x_hat;

            out.ng_p = G_X_k.cols() - nx;
            out.ng_e = nx;
        }
    } else if (svo_horizon == 0) {
        if (k > 1) {
            x_hat = Ad_kf * x_hat + Bd_kf * u_oldest + L_luenberger * (y_oldest - Cd_svo * x_hat);
            x_curr = x_hat;
        }
        double dyn_bound = std::pow(a_decay, k - 1) * e_init + tail;

        G_X_k = dyn_bound * P_sqrt_inv;
        c_X_k_out = x_curr;
        A_X_k.resize(0, G_X_k.cols());
        b_X_k_out.resize(0);

        x_est = c_X_k_out;
        out.ng_p = 0;
        out.ng_e = G_X_k.cols();
    } else {
        int n_valid = std::min(k - 1, svo_horizon);

        for (int i = svo_horizon - n_valid; i < svo_horizon; ++i) {
            Eigen::VectorXd u_i = U_hist.col(i);
            Eigen::VectorXd y_i = Y_hist_full.col(i).head(ny);
            x_curr = Ad_kf * x_curr + Bd_kf * u_i + L_luenberger * (y_i - Cd_svo * x_curr);
        }

        Eigen::VectorXd U_stack(nu * svo_horizon);
        Eigen::VectorXd Y_stack(ny * svo_horizon);
        for (int l = 0; l < svo_horizon; ++l) {
            int src = svo_horizon - 1 - l;
            U_stack.segment(l * nu, nu) = U_hist.col(src);
            Y_stack.segment(l * ny, ny) = Y_hist_full.col(src).head(ny);
        }

        if (n_valid < svo_horizon) {
            double dyn_bound = std::pow(a_decay, n_valid) * e_init + tail;

            G_X_k = dyn_bound * P_sqrt_inv;
            c_X_k_out = x_curr;
            A_X_k.resize(0, G_X_k.cols());
            b_X_k_out.resize(0);

            x_est = c_X_k_out;
            out.ng_p = 0;
            out.ng_e = G_X_k.cols();
        } else {
            double dyn_bound = std::pow(a_decay, k - 1 - svo_horizon) * e_init + tail;
            int ng_p = G.cols(), ng_e = G0.cols();

            G_X_k.resize(nx, ng_p + ng_e);
            G_X_k << G, dyn_bound * G0;

            c_X_k_out = c + c_U * U_stack + c0 * x_hat;

            A_X_k.resize(A.rows(), ng_p + ng_e);
            if (A.rows() > 0) {
                A_X_k.leftCols(ng_p) = A;
                A_X_k.rightCols(ng_e) = dyn_bound * A0;
            }

            b_X_k_out = Y_stack;
            if (b.size() > 0) b_X_k_out += b;
            if (b_U.size() > 0) b_X_k_out -= b_U * U_stack;
            if (b0.size() > 0) b_X_k_out -= b0 * x_hat;

            x_est = c_X_k_out + Z_X * b_X_k_out;
            if (ny * svo_horizon < nx) {
                // For N=1 with position-only C=[I_3, 0_3], velocity is unobservable from 1 position sample;
                // use the Luenberger velocity estimate rather than the rank-deficient pseudoinverse projection.
                x_est.tail(nx - ny) = x_curr.tail(nx - ny);
            }
            out.ng_p = ng_p;
            out.ng_e = ng_e;

            x_hat = Ad_kf * x_hat + Bd_kf * u_oldest + L_luenberger * (y_oldest - Cd_svo * x_hat);
        }
    }

    flattenMatrix(G_X_k, out.G_flat);
    flattenVector(c_X_k_out, out.c_flat);
    flattenMatrix(A_X_k, out.A_flat);
    flattenVector(b_X_k_out, out.b_flat);
    flattenVector(x_est, out.center_flat);
    flattenVector(x_curr, out.x_hat_flat);

    out.G_rows = G_X_k.rows(); out.G_cols = G_X_k.cols();
    out.A_rows = A_X_k.rows(); out.A_cols = A_X_k.cols();

    return out;
}

// Pseudo-Inverse preventing NaN generations
Eigen::MatrixXd dockingGNC::pseudoInverse(const Eigen::MatrixXd& a, double epsilon) {
    if (a.rows() == 0 || a.cols() == 0) return Eigen::MatrixXd::Zero(a.cols(), a.rows());
    
    Eigen::BDCSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // Enforce a strict minimum floor (1e-10) to prevent collapse
    double tolerance = std::max(1e-10, epsilon * std::max(a.cols(), a.rows()) * svd.singularValues().array().abs()(0));
    
    Eigen::VectorXd singularValuesInv = svd.singularValues();
    for (int i = 0; i < singularValuesInv.size(); ++i) {
        if (singularValuesInv(i) > tolerance) {
            singularValuesInv(i) = 1.0 / singularValuesInv(i);
        } else {
            singularValuesInv(i) = 0.0;
        }
    }
    return svd.matrixV() * singularValuesInv.asDiagonal() * svd.matrixU().adjoint();
}

Eigen::VectorXd dockingGNC::quatMult(const Eigen::VectorXd& q1, const Eigen::VectorXd& q2) {
    Eigen::VectorXd q_out(4);
    q_out(0) = q1(0)*q2(0) - q1(1)*q2(1) - q1(2)*q2(2) - q1(3)*q2(3);
    q_out(1) = q1(0)*q2(1) + q1(1)*q2(0) + q1(2)*q2(3) - q1(3)*q2(2);
    q_out(2) = q1(0)*q2(2) - q1(1)*q2(3) + q1(2)*q2(0) + q1(3)*q2(1);
    q_out(3) = q1(0)*q2(3) + q1(1)*q2(2) - q1(2)*q2(1) + q1(3)*q2(0);
    return q_out;
}

Eigen::VectorXd dockingGNC::quatInv(const Eigen::VectorXd& q) {
    Eigen::VectorXd q_inv(4);
    q_inv(0) = q(0);
    q_inv.tail(3) = -q.tail(3);
    return q_inv;
}

Eigen::MatrixXd dockingGNC::skewSym(const Eigen::VectorXd& v) {
    Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(3,3);
    mat(0,1) = -v(2); mat(0,2) = v(1);
    mat(1,0) = v(2);  mat(1,2) = -v(0);
    mat(2,0) = -v(1); mat(2,1) = v(0);
    return mat;
}

Eigen::VectorXd dockingGNC::MRP2EP(const Eigen::VectorXd& mrp) {
    double mag2 = mrp.squaredNorm();
    Eigen::VectorXd ep(4);
    ep(0) = (1.0 - mag2) / (1.0 + mag2);
    ep.tail(3) = 2.0 * mrp / (1.0 + mag2);
    return ep;
}

Eigen::VectorXd dockingGNC::EP2MRP(const Eigen::VectorXd& q) {
    Eigen::VectorXd mrp(3);
    double denom = 1.0 + q(0);
    if (denom < 1e-12) { 
        mrp = q.tail(3);
    } else {
        mrp = q.tail(3) / denom;
    }
    return mrp;
}

void dockingGNC::setEKFDynamics(std::vector<double> Ad_flat, std::vector<double> Bd_flat) {
    Ad_kf = Eigen::MatrixXd(6, 6);
    for(int i = 0; i < 6; ++i) {
        for(int j = 0; j < 6; ++j) Ad_kf(i,j) = Ad_flat[i*6 + j];
    }
    
    Bd_kf = Eigen::MatrixXd(6, 3);
    for(int i = 0; i < 6; ++i) {
        for(int j = 0; j < 3; ++j) Bd_kf(i,j) = Bd_flat[i*3 + j];
    }
}

KinematicTrajectory dockingGNC::propagateTargetKinematics(
    std::vector<double> dcm_CN_flat, std::vector<double> omega_CN_B_flat, 
    std::vector<double> dcm_NH_flat, int N_horizon, double step_time, 
    double L_chief, double L_servicer) 
{
    Eigen::VectorXd sigma_CN(3); for(int i=0; i<3; ++i) sigma_CN(i) = dcm_CN_flat[i];
    Eigen::VectorXd omega_CN_B(3); for(int i=0; i<3; ++i) omega_CN_B(i) = omega_CN_B_flat[i];
    
    Eigen::MatrixXd dcm_NH(3,3);
    for(int i=0; i<3; ++i) {
        for(int j=0; j<3; ++j) dcm_NH(i,j) = dcm_NH_flat[i*3 + j];
    }

    // read states
    Eigen::MatrixXd dcm_CN(3,3);
    for(int i=0; i<3; ++i) {
        for(int j=0; j<3; ++j) dcm_CN(i,j) = dcm_CN_flat[i*3 + j];
    }

    Eigen::Vector3d port_C(0.0, 0.0, -L_chief);
    Eigen::Vector3d z_axis_C(0.0, 0.0, 1.0);

    Eigen::Vector3d port_N_k = dcm_CN.transpose() * port_C;
    Eigen::Vector3d z_axis_N_k = dcm_CN.transpose() * z_axis_C;

    Eigen::Vector3d omega_N = dcm_CN.transpose() * omega_CN_B;
    double theta = omega_N.norm() * step_time;

    Eigen::MatrixXd R_inc = Eigen::MatrixXd::Identity(3,3);
    if (theta > 1e-8) {
        Eigen::Vector3d axis = omega_N / omega_N.norm();
        Eigen::MatrixXd K_mat = skewSym(axis);
        R_inc = Eigen::MatrixXd::Identity(3,3) + std::sin(theta)*K_mat + (1.0-std::cos(theta))*(K_mat*K_mat);
    }

    KinematicTrajectory traj;
    traj.port_chief_flat.reserve(3 * N_horizon);
    traj.port_offset_flat.reserve(3 * N_horizon);
    traj.vel_chief_flat.reserve(3 * N_horizon);
    traj.z_axis_flat.reserve(3 * N_horizon);

    for (int k = 0; k < N_horizon; ++k) {
        Eigen::Vector3d port_H_chief_k = dcm_NH.transpose() * port_N_k;
        Eigen::Vector3d z_axis_H_k = dcm_NH.transpose() * z_axis_N_k;

        Eigen::Vector3d v_N_k = omega_N.cross(port_N_k);
        Eigen::Vector3d v_H_k = dcm_NH.transpose() * v_N_k;
        Eigen::Vector3d vel_chief_k = v_H_k * L_servicer / L_chief;

        Eigen::Vector3d port_offset_k = -port_H_chief_k * (L_servicer / L_chief);

        for(int i=0; i<3; ++i) {
            traj.port_chief_flat.push_back(port_H_chief_k(i));
            traj.port_offset_flat.push_back(port_offset_k(i));
            traj.vel_chief_flat.push_back(vel_chief_k(i));
            traj.z_axis_flat.push_back(z_axis_H_k(i));
        }

        port_N_k = R_inc * port_N_k;
        z_axis_N_k = R_inc * z_axis_N_k;
    }
    return traj;
}

void dockingGNC::flattenMatrix(const Eigen::MatrixXd& mat, std::vector<double>& flat) {
    flat.clear();
    // For every row run store every column, left to right top to bottom 
    // Equivalent to python's C-order (Row Major)
    for(int i = 0; i < mat.rows(); ++i) {
        for(int j = 0; j < mat.cols(); ++j) flat.push_back(mat(i,j));
    }
}

void dockingGNC::flattenVector(const Eigen::VectorXd& vec, std::vector<double>& flat) {
    flat.clear();
    for(int i = 0; i < vec.size(); ++i) flat.push_back(vec(i));
}

std::vector<double> dockingGNC::updateMEKF(std::vector<double> sigma_meas_flat, double step_time) {
    // Automatically detect if gyro measurements (elements 3..5) were provided
    const bool has_gyro = (sigma_meas_flat.size() >= 6);

    Eigen::VectorXd sigma_meas(3);
    for (int i = 0; i < 3; i++) sigma_meas(i) = sigma_meas_flat[i];

    Eigen::VectorXd omega_meas = Eigen::VectorXd::Zero(3);
    if (has_gyro) {
        for (int i = 0; i < 3; i++) omega_meas(i) = sigma_meas_flat[i + 3];
    }

    if (!mekf_initialized) {
        q_est = MRP2EP(sigma_meas);
        omega_est = has_gyro ? omega_meas : Eigen::VectorXd::Zero(3);
        mekf_initialized = true;
    }

    // Body-frame quaternion propagation (right-multiply)
    double omega_mag = omega_est.norm();
    Eigen::VectorXd q_pred = q_est;
    if (omega_mag > 1e-8) {
        double theta = omega_mag * step_time;
        Eigen::VectorXd axis = omega_est / omega_mag;
        Eigen::VectorXd q_update(4);
        q_update(0) = std::cos(theta / 2.0);
        q_update.tail(3) = axis * std::sin(theta / 2.0);
        q_pred = quatMult(q_est, q_update);
    }
    q_pred.normalize();

    // Covariance propagation: d(alpha_B)/dt = -[omega_B x] alpha_B + d_omega_B
    Eigen::MatrixXd F_mat = Eigen::MatrixXd::Zero(6, 6);
    F_mat.block(0,0,3,3) = -skewSym(omega_est);
    F_mat.block(0,3,3,3) =  Eigen::MatrixXd::Identity(3, 3); // +I for rate estimation (not gyro bias)
    
    Eigen::MatrixXd I_6 = Eigen::MatrixXd::Identity(6, 6);
    Eigen::MatrixXd Phi = I_6 + F_mat * step_time;
    Eigen::MatrixXd P_att_pred = Phi * P_att * Phi.transpose() + Q_att;

    // right-multiply: q_meas = q_pred (x) dq_B 
    Eigen::VectorXd q_meas = MRP2EP(sigma_meas);
    Eigen::VectorXd dq_meas = quatMult(quatInv(q_pred), q_meas);
    if (dq_meas(0) < 0.0) dq_meas = -dq_meas;
    Eigen::VectorXd z_att = 2.0 * dq_meas.tail(3);

    Eigen::VectorXd dx(6);
    if (has_gyro) {
        // 6-DOF update (Servicer: uses sigma + gyro)
        Eigen::VectorXd z_res(6);
        z_res.head(3) = z_att;
        z_res.tail(3) = omega_meas - omega_est;

        Eigen::MatrixXd R_full = Eigen::MatrixXd::Zero(6, 6);
        R_full.topLeftCorner(3, 3) = R_att;
        R_full.bottomRightCorner(3, 3) = Eigen::MatrixXd::Identity(3, 3) * 1e-5;

        Eigen::MatrixXd S_full = P_att_pred + R_full;// H = [I3 | I3], no need to use
        Eigen::MatrixXd K_full = P_att_pred * S_full.inverse();
        dx = K_full * z_res;
        P_att = (I_6 - K_full) * P_att_pred;
    } else {
        // 3-DOF update (Chief: uses sigma only, H_att 3x6 )
        Eigen::MatrixXd S_att = H_att * P_att_pred * H_att.transpose() + R_att;
        Eigen::MatrixXd K_att = P_att_pred * H_att.transpose() * S_att.inverse();
        dx = K_att * z_att;
        P_att = (I_6 - K_att * H_att) * P_att_pred;
    }

    Eigen::VectorXd d_alpha = dx.head(3);
    Eigen::VectorXd d_omega = dx.tail(3);

    // Posteriori state update
    Eigen::VectorXd dq_update(4);
    dq_update(0) = 1.0;
    dq_update.tail(3) = 0.5 * d_alpha;
    dq_update.normalize();

    q_est = quatMult(q_pred, dq_update);
    q_est.normalize();
    if (q_est(0) < 0.0) q_est = -q_est;

    omega_est += d_omega;

    // Pack 6D output: [sigma_est_clean (0..2), omega_est (3..5)]
    Eigen::VectorXd sigma_est_clean = EP2MRP(q_est);
    std::vector<double> out(6);
    for (int i = 0; i < 3; ++i) out[i]     = sigma_est_clean(i);
    for (int i = 0; i < 3; ++i) out[i + 3] = omega_est(i);
    return out;
}
