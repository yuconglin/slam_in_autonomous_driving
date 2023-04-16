#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "ch3/eskf.hpp"
#include "ch3/static_imu_init.h"
#include "ch4/g2o_types.h"
#include "ch4/imu_preintegration.h"
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
DEFINE_bool(with_ui, true, "是否显示图形界面");

void Optimize(sad::NavStated& last_state, sad::NavStated& this_state, sad::GNSS& last_gnss, sad::GNSS& this_gnss,
              std::shared_ptr<sad::IMUPreintegration>& preinteg, const Vec3d& grav);

void OptimizeNew(sad::NavStated& last_state, sad::NavStated& this_state, sad::GNSS& last_gnss, sad::GNSS& this_gnss,
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
                auto state_pred = preinteg->Predict(last_state, eskf.GetGravity());

                auto last_state_copy = last_state;
                auto update_state_copy = update_state;
                LOG(INFO) << "..............................................";
                Optimize(last_state, update_state, last_gnss, gnss_convert, preinteg, eskf.GetGravity());
                LOG(INFO) << "update_state: " << update_state;

                OptimizeNew(last_state_copy, update_state_copy, last_gnss, gnss_convert, preinteg, eskf.GetGravity());
                LOG(INFO) << "new update_state: " << update_state_copy;

                constexpr double kError = 1e-1;
                EXPECT_NEAR((update_state_copy.p_ - update_state.p_).norm(), 0, kError);
                EXPECT_NEAR((update_state.R_.inverse() * update_state_copy.R_).log().norm(), 0, kError);
                EXPECT_NEAR((update_state_copy.v_ - update_state.v_).norm(), 0, kError);
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

void Optimize(sad::NavStated& last_state, sad::NavStated& this_state, sad::GNSS& last_gnss, sad::GNSS& this_gnss,
              std::shared_ptr<sad::IMUPreintegration>& pre_integ, const Vec3d& grav) {
    assert(pre_integ != nullptr);

    if (pre_integ->dt_ < 1e-3) {
        // 未得到积分
        return;
    }

    using BlockSolverType = g2o::BlockSolverX;
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;

    auto* solver = new g2o::OptimizationAlgorithmGaussNewton(
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

    this_state = sad::NavStated(this_state.timestamp_, v1_pose->estimate().so3(), v1_pose->estimate().translation(),
                                v1_vel->estimate(), v1_bg->estimate(), v1_ba->estimate());
}

void OptimizeNew(sad::NavStated& last_state, sad::NavStated& this_state, sad::GNSS& last_gnss, sad::GNSS& this_gnss,
                 std::shared_ptr<sad::IMUPreintegration>& pre_integ, const Vec3d& grav) {
    assert(pre_integ != nullptr);

    if (pre_integ->dt_ < 1e-3) {
        // 未得到积分
        return;
    }

    using BlockSolverType = g2o::BlockSolverX;
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;

    auto* solver = new g2o::OptimizationAlgorithmGaussNewton(
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
    auto edge_inertial = new sad::EdgeInertialNew(pre_integ, grav);

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

    this_state = sad::NavStated(this_state.timestamp_, v1_pose->estimate().so3(), v1_pose->estimate().translation(),
                                v1_vel->estimate(), v1_bg->estimate(), v1_ba->estimate());
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;

    testing::InitGoogleTest(&argc, argv);
    google::ParseCommandLineFlags(&argc, &argv, true);
    return RUN_ALL_TESTS();
}