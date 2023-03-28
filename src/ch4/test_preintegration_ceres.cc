#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "ch4/ceres_error_term.h"
#include "ch4/pose_local_parameterization.h"

#include "ch3/eskf.hpp"
#include "ch3/static_imu_init.h"
#include "ch4/imu_preintegration.h"
#include "ch4/g2o_types.h"
#include "common/g2o_types.h"
#include "common/io_utils.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

DEFINE_string(txt_path, "./data/ch3/10.txt", "数据文件路径");
DEFINE_double(antenna_angle, 12.06, "RTK天线安装偏角（角度）");
DEFINE_double(antenna_pox_x, -0.17, "RTK天线安装偏移X");
DEFINE_double(antenna_pox_y, -0.20, "RTK天线安装偏移Y");

void OptimizeCeres(sad::NavStated& last_state, sad::NavStated& this_state, sad::GNSS& last_gnss, sad::GNSS& this_gnss,
              std::shared_ptr<sad::IMUPreintegration>& preinteg, const Vec3d& grav);
            
void OptimizeCeresSep(sad::NavStated& last_state, sad::NavStated& this_state, sad::GNSS& last_gnss, sad::GNSS& this_gnss,
              std::shared_ptr<sad::IMUPreintegration>& preinteg, const Vec3d& grav);

void OptimizeG2o(sad::NavStated& last_state, sad::NavStated& this_state, sad::GNSS& last_gnss, sad::GNSS& this_gnss,
              std::shared_ptr<sad::IMUPreintegration>& preinteg, const Vec3d& grav);

/// 使用ESKF的Predict, Update来验证预积分的优化过程
TEST(PREINTEGRATION_TEST, ESKF_TEST) {
    if (fLS::FLAGS_txt_path.empty()) {
        FAIL();
    }

    // 初始化器
    sad::StaticIMUInit imu_init;  // 使用默认配置
    sad::ESKFD eskf;

    sad::TxtIO io(FLAGS_txt_path);
    Vec2d antenna_pos(FLAGS_antenna_pox_x, FLAGS_antenna_pox_y);

    std::ofstream fout("./data/ch3/gins.txt");
    bool imu_inited = false, gnss_inited = false;

    /// 设置各类回调函数
    bool first_gnss_set = false;
    Vec3d origin = Vec3d::Zero();

    std::shared_ptr<sad::IMUPreintegration> preinteg = nullptr;

    sad::NavStated last_state;
    bool last_state_set = false;

    sad::GNSS last_gnss;
    bool last_gnss_set = false;

    io.SetIMUProcessFunc([&](const sad::IMU& imu) {
          /// IMU 处理函数
          if (!imu_init.InitSuccess()) {
              imu_init.AddIMU(imu);
              return;
          }

          /// 需要IMU初始化
          if (!imu_inited) {
              // 读取初始零偏，设置ESKF
              sad::ESKFD::Options options;
              // 噪声由初始化器估计
              options.gyro_var_ = sqrt(imu_init.GetCovGyro()[0]);
              options.acce_var_ = sqrt(imu_init.GetCovAcce()[0]);
              eskf.SetInitialConditions(options, imu_init.GetInitBg(), imu_init.GetInitBa(), imu_init.GetGravity());

              imu_inited = true;
              return;
          }

          if (!gnss_inited) {
              /// 等待有效的RTK数据
              return;
          }

          /// GNSS 也接收到之后，再开始进行预测
          double current_time = eskf.GetNominalState().timestamp_;
          eskf.Predict(imu);

          if (preinteg) {
              preinteg->Integrate(imu, imu.timestamp_ - current_time);

              if (last_state_set) {
                  auto pred_of_preinteg = preinteg->Predict(last_state, eskf.GetGravity());
                  auto pred_of_eskf = eskf.GetNominalState();

                  /// 这两个预测值的误差应该非常接近
                  EXPECT_NEAR((pred_of_preinteg.p_ - pred_of_eskf.p_).norm(), 0, 1e-2);
                  EXPECT_NEAR((pred_of_preinteg.R_.inverse() * pred_of_eskf.R_).log().norm(), 0, 1e-2);
                  EXPECT_NEAR((pred_of_preinteg.v_ - pred_of_eskf.v_).norm(), 0, 1e-2);
              }
          }
      })
        .SetGNSSProcessFunc([&](const sad::GNSS& gnss) {
            /// GNSS 处理函数
            if (!imu_inited) {
                return;
            }

            sad::GNSS gnss_convert = gnss;
            if (!sad::ConvertGps2UTM(gnss_convert, antenna_pos, FLAGS_antenna_angle) || !gnss_convert.heading_valid_) {
                return;
            }

            /// 去掉原点
            if (!first_gnss_set) {
                origin = gnss_convert.utm_pose_.translation();
                first_gnss_set = true;
            }
            gnss_convert.utm_pose_.translation() -= origin;

            // 要求RTK heading有效，才能合入EKF
            auto state_bef_update = eskf.GetNominalState();

            eskf.ObserveGps(gnss_convert);

            // 验证优化过程是否正确
            if (last_state_set && last_gnss_set) {
                auto update_state = eskf.GetNominalState();

                // LOG(INFO) << "state before eskf update: " << state_bef_update;
                // LOG(INFO) << "state after  eskf update: " << update_state;

                auto state_pred = preinteg->Predict(last_state, eskf.GetGravity());
                // LOG(INFO) << "state in pred: " << state_pred;
                
                sad::NavStated last_state_copy = last_state;

                sad::NavStated g2o_state_copy = update_state;
                OptimizeG2o(last_state, g2o_state_copy, last_gnss, gnss_convert, preinteg, eskf.GetGravity());
                LOG(INFO) << "g2o: " << g2o_state_copy;
                
                // Puts ceres second because it will alter last_state's value.
                sad::NavStated ceres_state_copy = update_state;
                // OptimizeCeres(last_state_copy, ceres_state_copy, last_gnss, gnss_convert, preinteg, eskf.GetGravity());
                OptimizeCeresSep(last_state_copy, ceres_state_copy, last_gnss, gnss_convert, preinteg, eskf.GetGravity());
                LOG(INFO) << "ceres: " << ceres_state_copy;

                LOG(INFO) << "gnss " << gnss_convert.utm_pose_.translation().transpose();
                
                constexpr double kError = 1e-1;
                EXPECT_NEAR((g2o_state_copy.p_ - ceres_state_copy.p_).norm(), 0, kError);
                EXPECT_NEAR((g2o_state_copy.R_.inverse() * ceres_state_copy.R_).log().norm(), 0, kError);
                EXPECT_NEAR((g2o_state_copy.v_ - ceres_state_copy.v_).norm(), 0, kError);
                // assert((g2o_state_copy.p_ - ceres_state_copy.p_).norm() < 1e-2);
            }

            last_state = eskf.GetNominalState();
            last_state_set = true;

            // 重置预积分
            sad::IMUPreintegration::Options options;
            options.init_bg_ = last_state.bg_;
            options.init_ba_ = last_state.ba_;
            preinteg = std::make_shared<sad::IMUPreintegration>(options);

            gnss_inited = true;
            last_gnss = gnss_convert;
            last_gnss_set = true;
        })
        .SetOdomProcessFunc([&](const sad::Odom& odom) { imu_init.AddOdom(odom); })
        .Go();

    SUCCEED();
}

void OptimizeCeres(sad::NavStated& last_state, sad::NavStated& this_state, sad::GNSS& last_gnss, sad::GNSS& this_gnss,
              std::shared_ptr<sad::IMUPreintegration>& pre_integ, const Vec3d& grav) {
    assert(pre_integ != nullptr);

    if (pre_integ->dt_ < 1e-3) {
        // 未得到积分
        return;
    }

    ceres::Problem problem; 

    // Last pose, v, bg, ba.
    // Last pose.
    ceres::LocalParameterization *local_pose_parameterization = new PoseLocalParameterization();
    const Quatd last_q = last_state.R_.unit_quaternion();
    // tranlsation followed by a quarternion.
    double last_para_pose[7] = {last_state.p_.x(), last_state.p_.y(), last_state.p_.z(), last_q.x(), last_q.y(), last_q.z(), last_q.w()}; 
    problem.AddParameterBlock(last_para_pose, 7, local_pose_parameterization); 

    // Last v.
    problem.AddParameterBlock(last_state.v_.data(), 3);

    // Last bg.
    problem.AddParameterBlock(last_state.bg_.data(), 3);

    // Last ba.
    problem.AddParameterBlock(last_state.ba_.data(), 3);

    // This pose, v, bg, ba.
    // This pose.
    const Quatd this_q = this_state.R_.unit_quaternion();
    double this_para_pose[7] = {this_state.p_.x(), this_state.p_.y(), this_state.p_.z(), this_q.x(), this_q.y(), this_q.z(), this_q.w()};
    problem.AddParameterBlock(this_para_pose, 7, local_pose_parameterization);

    // This v.
    problem.AddParameterBlock(this_state.v_.data(), 3);

    // This bg.
    problem.AddParameterBlock(this_state.bg_.data(), 3);

    // This ba.
    problem.AddParameterBlock(this_state.ba_.data(), 3);

    // Preintegration edges.
    sad::PreInteCostFunction* preint_edge = new sad::PreInteCostFunction(pre_integ, grav, 1.0);
    problem.AddResidualBlock(preint_edge, new ceres::HuberLoss(200.0), last_para_pose, last_state.v_.data(), last_state.bg_.data(), last_state.ba_.data(), this_para_pose, this_state.v_.data());
    // Gyro and accelerometer bias random walk edges.
    sad::BiasRwCostFunction* gyro_bias_rw_edge = new sad::BiasRwCostFunction(Mat3d::Identity() * 1e6);
    problem.AddResidualBlock(gyro_bias_rw_edge, new ceres::HuberLoss(5.0), last_state.bg_.data(), this_state.bg_.data());
    sad::BiasRwCostFunction* acc_bias_rw_edge = new sad::BiasRwCostFunction(Mat3d::Identity() * 1e6);
    problem.AddResidualBlock(acc_bias_rw_edge, new ceres::HuberLoss(5.0), last_state.ba_.data(), this_state.ba_.data());

    // Gnss edges.
    sad::GnssCostFunction* last_gnss_edge = new sad::GnssCostFunction(last_gnss.utm_pose_, Mat6d::Identity() * 1e2);    
    problem.AddResidualBlock(last_gnss_edge, new ceres::HuberLoss(5.0), last_para_pose);
    sad::GnssCostFunction* this_gnss_edge = new sad::GnssCostFunction(this_gnss.utm_pose_, Mat6d::Identity() * 1e2);
    problem.AddResidualBlock(this_gnss_edge, new ceres::HuberLoss(5.0), this_para_pose);

    // Solves the problem.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 2000;
    // options.minimizer_progress_to_stdout = true;   // 输出到cout

    ceres::Solver::Summary summary;                // 优化信息
    ceres::Solve(options, &problem, &summary);  // 开始优化

    LOG(INFO) << summary.BriefReport();

    // Gathers the results.
    this_state.R_ = SO3(Quatd(this_para_pose[6], this_para_pose[3], this_para_pose[4], this_para_pose[5]).normalized());
    this_state.p_ = Vec3d(this_para_pose[0], this_para_pose[1], this_para_pose[2]);
    // The rest were already updated via pointers.
}

void OptimizeCeresSep(sad::NavStated& last_state, sad::NavStated& this_state, sad::GNSS& last_gnss, sad::GNSS& this_gnss,
              std::shared_ptr<sad::IMUPreintegration>& pre_integ, const Vec3d& grav) {
    assert(pre_integ != nullptr);

    if (pre_integ->dt_ < 1e-3) {
        // 未得到积分
        return;
    }

    ceres::Problem problem; 

    // Last pose, v, bg, ba.
    // Last pose.
    Quatd last_q = last_state.R_.unit_quaternion();
    // tranlsation followed by a quarternion.
    problem.AddParameterBlock(last_state.p_.data(), 3);
    problem.AddParameterBlock(last_q.coeffs().data(), 4, new ceres::QuaternionParameterization()); 

    // Last v.
    problem.AddParameterBlock(last_state.v_.data(), 3);

    // Last bg.
    problem.AddParameterBlock(last_state.bg_.data(), 3);

    // Last ba.
    problem.AddParameterBlock(last_state.ba_.data(), 3);

    // This pose, v, bg, ba.
    // This pose.
    Quatd this_q = this_state.R_.unit_quaternion();
    problem.AddParameterBlock(this_state.p_.data(), 3);
    problem.AddParameterBlock(this_q.coeffs().data(), 4, new ceres::QuaternionParameterization());

    // This v.
    problem.AddParameterBlock(this_state.v_.data(), 3);

    // This bg.
    problem.AddParameterBlock(this_state.bg_.data(), 3);

    // This ba.
    problem.AddParameterBlock(this_state.ba_.data(), 3);

    // Preintegration edges.
    sad::PreInteCostFunctionSep* preint_edge = new sad::PreInteCostFunctionSep(pre_integ, grav, 1.0);
    problem.AddResidualBlock(preint_edge, new ceres::HuberLoss(200.0), last_state.p_.data(), last_q.coeffs().data(), last_state.v_.data(), last_state.bg_.data(), last_state.ba_.data(), this_state.p_.data(), this_q.coeffs().data(), this_state.v_.data());

    // Gyro and accelerometer bias random walk edges.
    sad::BiasRwCostFunction* gyro_bias_rw_edge = new sad::BiasRwCostFunction(Mat3d::Identity() * 1e6);
    problem.AddResidualBlock(gyro_bias_rw_edge, nullptr, last_state.bg_.data(), this_state.bg_.data());
    sad::BiasRwCostFunction* acc_bias_rw_edge = new sad::BiasRwCostFunction(Mat3d::Identity() * 1e6);
    problem.AddResidualBlock(acc_bias_rw_edge, nullptr, last_state.ba_.data(), this_state.ba_.data());

    // Gnss edges.
    sad::GnssCostFunctionSep* last_gnss_edge = new sad::GnssCostFunctionSep(last_gnss.utm_pose_, Mat6d::Identity() * 1e2);    
    problem.AddResidualBlock(last_gnss_edge, nullptr, last_state.p_.data(), last_q.coeffs().data());
    sad::GnssCostFunctionSep* this_gnss_edge = new sad::GnssCostFunctionSep(this_gnss.utm_pose_, Mat6d::Identity() * 1e2);
    problem.AddResidualBlock(this_gnss_edge, nullptr, this_state.p_.data(), this_q.coeffs().data());

    // Solves the problem.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 1000;
    // options.minimizer_progress_to_stdout = true;   // 输出到cout

    ceres::Solver::Summary summary;                // 优化信息
    ceres::Solve(options, &problem, &summary);  // 开始优化

    LOG(INFO) << summary.BriefReport();

    // Gathers the results.
    this_state.R_ = SO3(this_q);
    // The rest were already updated via pointers.
}

void OptimizeG2o(sad::NavStated& last_state, sad::NavStated& this_state, sad::GNSS& last_gnss, sad::GNSS& this_gnss,
              std::shared_ptr<sad::IMUPreintegration>& pre_integ, const Vec3d& grav) {
    assert(pre_integ != nullptr);

    if (pre_integ->dt_ < 1e-3) {
        // 未得到积分
        return;
    }

    using BlockSolverType = g2o::BlockSolverX;
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;

    auto* solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // 上时刻顶点， pose, v, bg, ba
    auto v0_pose = new sad::VertexPose();
    v0_pose->setId(0);
    v0_pose->setEstimate(last_state.GetSE3());
    optimizer.addVertex(v0_pose);

    auto v0_vel = new sad::VertexVelocity();
    v0_vel->setId(1);
    v0_vel->setEstimate(last_state.v_);
    optimizer.addVertex(v0_vel);

    auto v0_bg = new sad::VertexGyroBias();
    v0_bg->setId(2);
    v0_bg->setEstimate(last_state.bg_);
    optimizer.addVertex(v0_bg);

    auto v0_ba = new sad::VertexAccBias();
    v0_ba->setId(3);
    v0_ba->setEstimate(last_state.ba_);
    optimizer.addVertex(v0_ba);

    // 本时刻顶点，pose, v, bg, ba
    auto v1_pose = new sad::VertexPose();
    v1_pose->setId(4);
    v1_pose->setEstimate(this_state.GetSE3());
    optimizer.addVertex(v1_pose);

    auto v1_vel = new sad::VertexVelocity();
    v1_vel->setId(5);
    v1_vel->setEstimate(this_state.v_);
    optimizer.addVertex(v1_vel);

    auto v1_bg = new sad::VertexGyroBias();
    v1_bg->setId(6);
    v1_bg->setEstimate(this_state.bg_);
    optimizer.addVertex(v1_bg);

    auto v1_ba = new sad::VertexAccBias();
    v1_ba->setId(7);
    v1_ba->setEstimate(this_state.ba_);
    optimizer.addVertex(v1_ba);

    // 预积分边
    auto edge_inertial = new sad::EdgeInertial(pre_integ, grav);
    edge_inertial->setVertex(0, v0_pose);
    edge_inertial->setVertex(1, v0_vel);
    edge_inertial->setVertex(2, v0_bg);
    edge_inertial->setVertex(3, v0_ba);
    edge_inertial->setVertex(4, v1_pose);
    edge_inertial->setVertex(5, v1_vel);

    auto* rk = new g2o::RobustKernelHuber();
    rk->setDelta(200.0);
    edge_inertial->setRobustKernel(rk);

    optimizer.addEdge(edge_inertial);
    edge_inertial->computeError();

    auto* edge_gyro_rw = new sad::EdgeGyroRW();
    edge_gyro_rw->setVertex(0, v0_bg);
    edge_gyro_rw->setVertex(1, v1_bg);
    edge_gyro_rw->setInformation(Mat3d::Identity() * 1e6);
    optimizer.addEdge(edge_gyro_rw);

    edge_gyro_rw->computeError();

    auto* edge_acc_rw = new sad::EdgeAccRW();
    edge_acc_rw->setVertex(0, v0_ba);
    edge_acc_rw->setVertex(1, v1_ba);
    edge_acc_rw->setInformation(Mat3d::Identity() * 1e6);
    optimizer.addEdge(edge_acc_rw);

    edge_acc_rw->computeError();

    // GNSS边
    auto edge_gnss0 = new sad::EdgeGNSS(v0_pose, last_gnss.utm_pose_);
    edge_gnss0->setInformation(Mat6d::Identity() * 1e2);
    optimizer.addEdge(edge_gnss0);

    edge_gnss0->computeError();

    auto edge_gnss1 = new sad::EdgeGNSS(v1_pose, this_gnss.utm_pose_);
    edge_gnss1->setInformation(Mat6d::Identity() * 1e2);
    optimizer.addEdge(edge_gnss1);

    edge_gnss1->computeError();

    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    sad::NavStated corr_state(this_state.timestamp_, v1_pose->estimate().so3(), v1_pose->estimate().translation(),
                              v1_vel->estimate(), v1_bg->estimate(), v1_ba->estimate());
    this_state = corr_state;
    /*
    // 获取结果，统计各类误差
    LOG(INFO) << "chi2/error: ";
    LOG(INFO) << "preintegration: " << edge_inertial->chi2() << "/" << edge_inertial->error().transpose();
    LOG(INFO) << "gnss0: " << edge_gnss0->chi2() << ", " << edge_gnss0->error().transpose();
    LOG(INFO) << "gnss1: " << edge_gnss1->chi2() << ", " << edge_gnss1->error().transpose();
    LOG(INFO) << "bias: " << edge_gyro_rw->chi2() << "/" << edge_acc_rw->error().transpose();
    */
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;

    testing::InitGoogleTest(&argc, argv);
    google::ParseCommandLineFlags(&argc, &argv, true);
    return RUN_ALL_TESTS();
}
