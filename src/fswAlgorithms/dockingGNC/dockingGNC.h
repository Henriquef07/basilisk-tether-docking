#ifndef REACHABLE_SETS_H
#define REACHABLE_SETS_H

#include <Eigen/Dense>
#include <vector>


struct KinematicTrajectory {
    std::vector<double> port_chief_flat;
    std::vector<double> port_offset_flat;
    std::vector<double> vel_chief_flat;
    std::vector<double> z_axis_flat;
};

struct CCG_Tuple {
    std::vector<double> G_flat;
    std::vector<double> c_flat;
    std::vector<double> A_flat;
    std::vector<double> b_flat;
    std::vector<double> center_flat;
    std::vector<double> x_hat_flat;
    
    int G_rows, G_cols;
    int A_rows, A_cols;
    
    int ng_p; // Number of infinity norm generators
    int ng_e; // Number of ellipsoidal generators
};
struct CBFResult {
        double h;
        std::vector<double> grad_u_flat;
        bool converged;
        int iterations;
        std::vector<double> u_safe_flat;
    };
class dockingGNC {
public:
    dockingGNC();
    ~dockingGNC();

    std::vector<double> updateMEKF(std::vector<double> sigma_meas, double step_time);

    void setEKFDynamics(std::vector<double> Ad_flat, std::vector<double> Bd_flat);

    KinematicTrajectory propagateTargetKinematics(std::vector<double> dcm_CN_flat, 
        std::vector<double> omega_CN_B_flat, std::vector<double> dcm_NH_flat, 
        int N_horizon, double step_time, double L_chief, double L_servicer);

     
    Eigen::MatrixXd G;    
    Eigen::MatrixXd G0; 
    Eigen::MatrixXd A;  
    Eigen::MatrixXd A0;   
    Eigen::VectorXd c;    
    Eigen::MatrixXd c_U;  
    Eigen::MatrixXd c0;   
    Eigen::VectorXd b;    
    Eigen::MatrixXd b_U;  
    Eigen::MatrixXd b0;   
    Eigen::MatrixXd Z_X;      
    int svo_horizon;          

    Eigen::VectorXd x_hat;   
    Eigen::MatrixXd L_luenberger; 
    Eigen::MatrixXd P_sqrt_inv;  

    Eigen::MatrixXd G_X_k;
    Eigen::MatrixXd A_X_k;
    Eigen::VectorXd c_X_k_out;
    Eigen::VectorXd b_X_k_out;
    Eigen::VectorXd x_hat_current;

    double a_decay;              
    double e_noise;          
    double e_init;           
    int k;               
    
    void initSVO(std::vector<double> Ad_flat, std::vector<double> Bd_flat, std::vector<double> Cd_flat, 
        double w_max, double v_max, int L_horizon, std::vector<double> x_hat_init);
    CCG_Tuple updateSVO( std::vector<double> U_hist_flat, std::vector<double> Y_hist_flat);

    CBFResult evaluateCBF_Uopt( std::vector<double> u_opt_flat, std::vector<double> p_target_flat, double exclusion_radius,
    double h_k, double alpha_cbf, double step_time, double safety_margin, double max_accel);

    double evaluateCBF_Current( std::vector<double> p_target_flat, double exclusion_radius);
    CBFResult evaluateCBF_Gradient( std::vector<double> p_target_flat, double exclusion_radius);
    
    std::vector<double> filterQPCBF_3D(std::vector<double> u_nom_flat, std::vector<double> grad_h_flat, double h_val, 
        double alpha_cbf, double step_time_fast, double max_thrust, double mass, std::vector<double> pos_err_flat, 
        std::vector<double> v_curr_flat, std::vector<double> rc_flat, std::vector<double> vc_flat, double mu);


private:
    bool mekf_initialized = false;
    Eigen::VectorXd q_est;
    Eigen::VectorXd omega_est;
    Eigen::MatrixXd P_att;
    Eigen::MatrixXd Q_att;
    Eigen::MatrixXd R_att;
    Eigen::MatrixXd H_att;
    
    Eigen::VectorXd x_kf_est;
    Eigen::MatrixXd P_kf;
    Eigen::MatrixXd Q_kf;
    Eigen::MatrixXd R_kf;
    Eigen::MatrixXd Ad_kf;
    Eigen::MatrixXd Bd_kf;

    // Helper functions
    Eigen::VectorXd quatMult(const Eigen::VectorXd& q1, const Eigen::VectorXd& q2);
    Eigen::VectorXd quatInv(const Eigen::VectorXd& q);
    Eigen::MatrixXd skewSym(const Eigen::VectorXd& v);
    Eigen::VectorXd MRP2EP(const Eigen::VectorXd& mrp);
    Eigen::VectorXd EP2MRP(const Eigen::VectorXd& q);
    Eigen::MatrixXd pseudoInverse(const Eigen::MatrixXd& a, double epsilon = 1e-10);
    void flattenMatrix(const Eigen::MatrixXd& mat, std::vector<double>& flat);
    void flattenVector(const Eigen::VectorXd& vec, std::vector<double>& flat);

    Eigen::MatrixXd nullSpace(const Eigen::MatrixXd& a, double epsilon = 1e-12);
};

#endif