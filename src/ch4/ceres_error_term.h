#ifndef SLAM_IN_AUTO_DRIVING_CH4_CERES_ERROR_TERMS_H
#define SLAM_IN_AUTO_DRIVING_CH4_CERES_ERROR TERMS_H

#include <memory>

#include "ceres/ceres.h"

#include "ch4/imu_preintegration.h"
#include "common/eigen_types.h"

namespace sad {
/*
The ceres cost function related to imu preintegration.
1. The residual is of dimension 9. The order is er, ev, ep.
2. The variabiles are:
   a. rotation and translation of last frame (4D + 3D)
   b. velocity of last frame (3D)
   c. gyro bias of last frame (3D)
   d. accelerometer bias of last frame (3D)
   e. rotation and translation of the current frame (4D + 3D)
   f. velocity of the current frame (3D)
*/
class PreInteCostFunction : public ceres::SizedCostFunction</*residual*/ 9, /*a*/ 7, /*b*/ 3, /*c*/ 3, /*d*/ 3, /*e*/ 7,
                                                            /*f*/ 3> {
   public:
    PreInteCostFunction() = delete;

    PreInteCostFunction(std::shared_ptr<IMUPreintegration> preinteg, const Vec3d &gravity, double weight)
        : preint_(preinteg), grav_(gravity), dt_(preinteg->dt_), dt2_(preinteg->dt2_) {
        sqrt_information_ =
            Eigen::LLT<Eigen::Matrix<double, 9, 9>>(preint_->cov_.inverse() * weight).matrixL().transpose();
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        // variables.
        Vec3d p_i(parameters[0][0], parameters[0][1], parameters[0][2]);
        Quatd q_i(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
        Vec3d v_i(parameters[1][0], parameters[1][1], parameters[1][2]);
        Vec3d bg_i(parameters[2][0], parameters[2][1], parameters[2][2]);
        Vec3d ba_i(parameters[3][0], parameters[3][1], parameters[3][2]);

        Vec3d p_j(parameters[4][0], parameters[4][1], parameters[4][2]);
        Quatd q_j(parameters[4][6], parameters[4][3], parameters[4][4], parameters[4][5]);
        Vec3d v_j(parameters[5][0], parameters[5][1], parameters[5][2]);

        // residuals.
        const SO3 dR = preint_->GetDeltaRotation(bg_i);
        const Vec3d dv = preint_->GetDeltaVelocity(bg_i, ba_i);
        const Vec3d dp = preint_->GetDeltaPosition(bg_i, ba_i);

        // from eq (4.41)
        const SO3 Ri(q_i);
        const SO3 Rj(q_j);
        const SO3 RiT = Ri.inverse();
        const SO3 eR = dR.inverse() * RiT * Rj;
        const Vec3d er = eR.log();
        const Mat3d RiT_Mat = RiT.matrix();
        const Vec3d ev = RiT_Mat * (v_j - v_i - grav_ * dt_) - dv;
        const Vec3d ep = RiT_Mat * (p_j - p_i - v_i * dt_ - grav_ * dt2_ * 0.5) - dp;

        Eigen::Map<Eigen::Matrix<double, 9, 1>> residual(residuals);
        residual.block<3, 1>(0, 0) = er;
        residual.block<3, 1>(3, 0) = ev;
        residual.block<3, 1>(6, 0) = ep;
        residual = sqrt_information_ * residual;

        if (!jacobians) {
            return true;
        }

        // 一些中间符号
        auto dR_dbg = preint_->dR_dbg_;
        auto dv_dbg = preint_->dV_dbg_;
        auto dp_dbg = preint_->dP_dbg_;
        auto dv_dba = preint_->dV_dba_;
        auto dp_dba = preint_->dP_dba_;

        const Vec3d dbg = bg_i - preint_->bg_;

        const Mat3d invJr = SO3::jr_inv(eR);

        if (jacobians[0]) {
            // residual w.r.t pose_i.
            Eigen::Map<Eigen::Matrix<double, 9, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
            jacobian_pose_i.setZero();
            // dR/dR1.
            jacobian_pose_i.block<3, 3>(0, 3) = -invJr * (Rj.inverse() * Ri).matrix();
            // dv/dR1.
            jacobian_pose_i.block<3, 3>(3, 3) = SO3::hat(RiT * (v_j - v_i - grav_ * dt_));
            // dp/dR1.
            jacobian_pose_i.block<3, 3>(6, 3) = SO3::hat(RiT * (p_j - p_i - v_i * dt_ - 0.5 * grav_ * dt2_));
            // dp/dp1.
            jacobian_pose_i.block<3, 3>(6, 0) = -RiT_Mat;

            jacobian_pose_i = sqrt_information_ * jacobian_pose_i;
        }

        if (jacobians[1]) {
            // residual w.r.t v_i.
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacobian_v_i(jacobians[1]);
            jacobian_v_i.setZero();
            // dv/dv1.
            jacobian_v_i.block<3, 3>(3, 0) = -RiT_Mat;
            // dp/dv1.
            jacobian_v_i.block<3, 3>(6, 0) = -RiT_Mat * dt_;

            jacobian_v_i = sqrt_information_ * jacobian_v_i;
        }

        if (jacobians[2]) {
            // residual w.r.t bg_i.
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacobian_bg_i(jacobians[2]);
            jacobian_bg_i.setZero();
            // dR/dbg1.
            jacobian_bg_i.block<3, 3>(0, 0) = -invJr * eR.inverse().matrix() * SO3::jr((dR_dbg * dbg).eval()) * dR_dbg;
            // dv/dbg1.
            jacobian_bg_i.block<3, 3>(3, 0) = -dv_dbg;
            // dp/dgb1.
            jacobian_bg_i.block<3, 3>(6, 0) = -dp_dbg;

            jacobian_bg_i = sqrt_information_ * jacobian_bg_i;
        }

        if (jacobians[3]) {
            // residual w.r.t ba_i.
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacobian_ba_i(jacobians[3]);
            jacobian_ba_i.setZero();
            // dv/dba1.
            jacobian_ba_i.block<3, 3>(3, 0) = -dv_dba;
            // dp/dba1.
            jacobian_ba_i.block<3, 3>(6, 0) = -dp_dba;

            jacobian_ba_i = sqrt_information_ * jacobian_ba_i;
        }

        if (jacobians[4]) {
            // residual w.r.t. pose_j.
            Eigen::Map<Eigen::Matrix<double, 9, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[4]);
            jacobian_pose_j.setZero();
            // dR/dp2.
            jacobian_pose_j.block<3, 3>(0, 0) = invJr;
            // dR/dR2.
            jacobian_pose_j.block<3, 3>(0, 3) = RiT_Mat;

            jacobian_pose_j = sqrt_information_ * jacobian_pose_j;
        }

        if (jacobians[5]) {
            // residual w.r.t v_j.
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacobian_v_j(jacobians[5]);
            jacobian_v_j.setZero();
            // dv/dv2.
            jacobian_v_j.block<3, 3>(3, 0) = RiT_Mat;

            jacobian_v_j = sqrt_information_ * jacobian_v_j;
        }

        return true;
    }

   private:
    std::shared_ptr<IMUPreintegration> preint_ = nullptr;
    Vec3d grav_;
    const double dt_;
    const double dt2_;
    Eigen::Matrix<double, 9, 9> sqrt_information_;
};

/*
The variables are:
a. translation of last frame.
b. rotation of last frame.
c. velocity of last frame.
d. gyro bias of last frame.
e. accelerometer of last frame.
f. translation of this frame.
g. rotation of this frame.
h. velocity of this frame.
*/
class PreInteCostFunctionSep
    : public ceres::SizedCostFunction<9, /*a*/ 3, /*b*/ 4, /*c*/ 3, /*d*/ 3, /*e*/ 3, /*f*/ 3, /*g*/ 4, /*h*/ 3> {
   public:
    PreInteCostFunctionSep() = delete;

    PreInteCostFunctionSep(std::shared_ptr<IMUPreintegration> preinteg, const Vec3d &gravity, double weight)
        : preint_(preinteg), grav_(gravity), dt_(preinteg->dt_), dt2_(preinteg->dt2_) {
        sqrt_information_ =
            Eigen::LLT<Eigen::Matrix<double, 9, 9>>(preint_->cov_.inverse() * weight).matrixL().transpose();
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        // variables.
        Eigen::Map<const Vec3d> p_i(parameters[0]);
        Eigen::Map<const Quatd> q_i(parameters[1]);
        Eigen::Map<const Vec3d> v_i(parameters[2]);
        Eigen::Map<const Vec3d> bg_i(parameters[3]);
        Eigen::Map<const Vec3d> ba_i(parameters[4]);
        Eigen::Map<const Vec3d> p_j(parameters[5]);
        Eigen::Map<const Quatd> q_j(parameters[6]);
        Eigen::Map<const Vec3d> v_j(parameters[7]);

        // residuals.
        const SO3 dR = preint_->GetDeltaRotation(bg_i);
        const Vec3d dv = preint_->GetDeltaVelocity(bg_i, ba_i);
        const Vec3d dp = preint_->GetDeltaPosition(bg_i, ba_i);

        // from eq (4.41)
        const SO3 Ri(q_i);
        const SO3 Rj(q_j);
        const SO3 RiT = Ri.inverse();
        const SO3 eR = dR.inverse() * RiT * Rj;
        const Vec3d er = eR.log();
        const Mat3d RiT_Mat = RiT.matrix();
        const Vec3d ev = RiT_Mat * (v_j - v_i - grav_ * dt_) - dv;
        const Vec3d ep = RiT_Mat * (p_j - p_i - v_i * dt_ - grav_ * dt2_ * 0.5) - dp;

        Eigen::Map<Eigen::Matrix<double, 9, 1>> residual(residuals);
        residual.block<3, 1>(0, 0) = er;
        residual.block<3, 1>(3, 0) = ev;
        residual.block<3, 1>(6, 0) = ep;
        residual = sqrt_information_ * residual;

        if (!jacobians) {
            return true;
        }

        // 一些中间符号
        auto dR_dbg = preint_->dR_dbg_;
        auto dv_dbg = preint_->dV_dbg_;
        auto dp_dbg = preint_->dP_dbg_;
        auto dv_dba = preint_->dV_dba_;
        auto dp_dba = preint_->dP_dba_;

        const Vec3d dbg = bg_i - preint_->bg_;

        const Mat3d invJr = SO3::jr_inv(eR);

        if (jacobians[0]) {
            // residual w.r.t p_i.
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacobian_p_i(jacobians[0]);
            jacobian_p_i.setZero();
            // dp/dp1.
            jacobian_p_i.block<3, 3>(6, 0) = -RiT_Mat;

            jacobian_p_i = sqrt_information_ * jacobian_p_i;
        }

        if (jacobians[1]) {
            // residual w.r.t R_i.
            Eigen::Map<Eigen::Matrix<double, 9, 4, Eigen::RowMajor>> jacobian_R_i(jacobians[1]);
            jacobian_R_i.setZero();
            // dR/dR1.
            jacobian_R_i.block<3, 3>(0, 0) = -invJr * (Rj.inverse() * Ri).matrix();
            // dv/dR1.
            jacobian_R_i.block<3, 3>(3, 0) = SO3::hat(RiT * (v_j - v_i - grav_ * dt_));
            // dp/dR1.
            jacobian_R_i.block<3, 3>(6, 0) = SO3::hat(RiT * (p_j - p_i - v_i * dt_ - 0.5 * grav_ * dt2_));

            jacobian_R_i = sqrt_information_ * jacobian_R_i;
        }

        if (jacobians[2]) {
            // residual w.r.t v_i.
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacobian_v_i(jacobians[2]);
            jacobian_v_i.setZero();
            // dv/dv1.
            jacobian_v_i.block<3, 3>(3, 0) = -RiT_Mat;
            // dp/dv1.
            jacobian_v_i.block<3, 3>(6, 0) = -RiT_Mat * dt_;

            jacobian_v_i = sqrt_information_ * jacobian_v_i;
        }

        if (jacobians[3]) {
            // residual w.r.t bg_i.
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacobian_bg_i(jacobians[3]);
            jacobian_bg_i.setZero();
            // dR/dbg1.
            jacobian_bg_i.block<3, 3>(0, 0) = -invJr * eR.inverse().matrix() * SO3::jr((dR_dbg * dbg).eval()) * dR_dbg;
            // dv/dbg1.
            jacobian_bg_i.block<3, 3>(3, 0) = -dv_dbg;
            // dp/dgb1.
            jacobian_bg_i.block<3, 3>(6, 0) = -dp_dbg;

            jacobian_bg_i = sqrt_information_ * jacobian_bg_i;
        }

        if (jacobians[4]) {
            // residual w.r.t ba_i.
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacobian_ba_i(jacobians[4]);
            jacobian_ba_i.setZero();
            // dv/dba1.
            jacobian_ba_i.block<3, 3>(3, 0) = -dv_dba;
            // dp/dba1.
            jacobian_ba_i.block<3, 3>(6, 0) = -dp_dba;

            jacobian_ba_i = sqrt_information_ * jacobian_ba_i;
        }

        if (jacobians[5]) {
            // residual w.r.t p_j.
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacobian_p_j(jacobians[5]);
            jacobian_p_j.setZero();
            // dR/dp2.
            jacobian_p_j.block<3, 3>(0, 0) = invJr;

            jacobian_p_j = sqrt_information_ * jacobian_p_j;
        }

        if (jacobians[6]) {
            // residual w.r.t R_j.
            Eigen::Map<Eigen::Matrix<double, 9, 4, Eigen::RowMajor>> jacobian_R_j(jacobians[6]);
            jacobian_R_j.setZero();
            // dR/dR2.
            jacobian_R_j.block<3, 3>(0, 0) = RiT_Mat;

            jacobian_R_j = sqrt_information_ * jacobian_R_j;
        }

        if (jacobians[7]) {
            // residual w.r.t v_j.
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacobian_v_j(jacobians[7]);
            jacobian_v_j.setZero();
            // dv/dv2.
            jacobian_v_j.block<3, 3>(3, 0) = RiT_Mat;

            jacobian_v_j = sqrt_information_ * jacobian_v_j;
        }

        return true;
    }

   private:
    std::shared_ptr<IMUPreintegration> preint_ = nullptr;
    Vec3d grav_;
    const double dt_;
    const double dt2_;
    Eigen::Matrix<double, 9, 9> sqrt_information_;
};

/*
The Gnss cost fuction for RTK measurement with position and orientation.
*/
class GnssCostFunction : public ceres::SizedCostFunction<6, 7> {
   public:
    GnssCostFunction() = delete;

    GnssCostFunction(const SE3 &obs, const Mat6d &information) : obs_(obs) {
        sqrt_information_ = Eigen::LLT<Mat6d>(information).matrixL().transpose();
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        const Eigen::Vector3d translation(parameters[0][0], parameters[0][1], parameters[0][2]);
        const SO3 R(Eigen::Quaterniond(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]));

        Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);
        residual.head<3>() = translation - obs_.translation();
        residual.tail<3>() = (obs_.so3().inverse() * R).log();
        residual = sqrt_information_ * residual;

        if (!jacobians) {
            return true;
        }

        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian(jacobians[0]);
            jacobian.setZero();

            // dp/dp
            jacobian.block<3, 3>(0, 0) = Mat3d::Identity();
            // dR/dR
            jacobian.block<3, 3>(3, 3) = (obs_.so3().inverse() * R).jr_inv();
            jacobian = sqrt_information_ * jacobian;
        }

        return true;
    }

   private:
    SE3 obs_;
    Mat6d sqrt_information_;
};

/*
The Gnss cost fuction for RTK measurement with position and orientation.
*/
class GnssCostFunctionSep : public ceres::SizedCostFunction<6, 3, 4> {
   public:
    GnssCostFunctionSep() = delete;

    GnssCostFunctionSep(const SE3 &obs, const Mat6d &information) : obs_(obs) {
        sqrt_information_ = Eigen::LLT<Mat6d>(information).matrixL().transpose();
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Map<const Vec3d> translation(parameters[0]);
        Eigen::Map<const Quatd> q(parameters[1]);
        const SO3 R(q);

        Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);
        residual.head<3>() = translation - obs_.translation();
        residual.tail<3>() = (obs_.so3().inverse() * R).log();
        residual = sqrt_information_ * residual;

        if (!jacobians) {
            return true;
        }

        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> jacobian_p(jacobians[0]);
            jacobian_p.setZero();
            // dp/dp.
            jacobian_p.block<3, 3>(0, 0) = Mat3d::Identity();
            jacobian_p = sqrt_information_ * jacobian_p;
        }

        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> jacobian_R(jacobians[1]);
            jacobian_R.setZero();
            // dR/dR.
            jacobian_R.block<3, 3>(0, 0) = (obs_.so3().inverse() * R).jr_inv();
            jacobian_R = sqrt_information_ * jacobian_R;
        }

        return true;
    }

   private:
    SE3 obs_;
    Mat6d sqrt_information_;
};

/*
The Gyro or accelerometer bias random walk cost function.
*/
class BiasRwCostFunction : public ceres::SizedCostFunction<3, 3, 3> {
   public:
    BiasRwCostFunction() = delete;

    BiasRwCostFunction(const Mat3d &information) {
        sqrt_information_ = Eigen::LLT<Mat3d>(information).matrixL().transpose();
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Vector3d bias_i(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Vector3d bias_j(parameters[1][0], parameters[1][1], parameters[1][2]);

        Eigen::Map<Vec3d> residual(residuals);
        residual = sqrt_information_ * (bias_i - bias_j);

        if (!jacobians) {
            return true;
        }

        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian(jacobians[0]);
            jacobian = Mat3d::Identity();
        }

        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian(jacobians[1]);
            jacobian = -Mat3d::Identity();
        }
        return true;
    }

   private:
    Mat3d sqrt_information_;
};

}  // namespace sad

#endif
