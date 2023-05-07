#include "ch8/lio-iekf/icp_3d_inc.h"
#include "ch7/loam-like/feature_extraction.h"
#include "common/lidar_utils.h"
#include "common/math_utils.h"

#include <pcl/common/transforms.h>
#include <execution>

namespace sad {

Icp3dInc::Icp3dInc(const Icp3dInc::Options& options) : options_(options), feature_extraction_(new FeatureExtraction) {
    kdtree_.SetEnableANN();
    kdtree_edge_.SetEnableANN();
    kdtree_surf_.SetEnableANN();
}

bool Icp3dInc::ComputeResidualAndJacobiansPointToPoint(const SE3& pose, Mat18d& HTVH, Vec18d& HTVr) {
    assert(source_ && local_map_);

    // 对点的索引，预先生成
    std::vector<int> index(source_->points.size());
    std::iota(index.begin(), index.end(), 0);

    // 我们来写一些并发代码
    std::vector<bool> effect_pts(index.size(), false);
    std::vector<Eigen::Matrix<double, 3, 18>> jacobians(index.size());
    std::vector<Vec3d> errors(index.size());

    std::for_each(std::execution::par_unseq, index.begin(), index.end(),
                  [this, &pose, &index, &effect_pts, &jacobians, &errors](int idx) {
                      auto q = ToVec3d(source_->points[idx]);
                      Vec3d qs = pose * q;  // 转换之后的q
                      std::vector<int> nn;
                      kdtree_.GetClosestPoint(ToPointType(qs), nn, 1);

                      if (!nn.empty()) {
                          Vec3d p = ToVec3d(local_map_->points[nn[0]]);
                          double dis2 = (p - qs).squaredNorm();
                          if (dis2 > options_.max_nn_distance_) {
                              // 点离的太远了不要
                              effect_pts[idx] = false;
                              return;
                          }

                          effect_pts[idx] = true;

                          // build residual
                          Vec3d e = p - qs;
                          Eigen::Matrix<double, 3, 18> J;
                          J.setZero();
                          J.block<3, 3>(0, 6) = pose.so3().matrix() * SO3::hat(q);
                          J.block<3, 3>(0, 0) = -Mat3d::Identity();

                          jacobians[idx] = J;
                          errors[idx] = e;
                      } else {
                          effect_pts[idx] = false;
                      }
                  });

    // 累加Hessian和error,计算dx
    // 原则上可以用reduce并发，写起来比较麻烦，这里写成accumulate
    double total_res = 0;
    int effective_num = 0;
    auto H_and_err = std::accumulate(
        index.begin(), index.end(), std::pair<Mat18d, Vec18d>(Mat18d::Zero(), Vec18d::Zero()),
        [&jacobians, &errors, &effect_pts, &total_res, &effective_num](const std::pair<Mat18d, Vec18d>& pre,
                                                                       int idx) -> std::pair<Mat18d, Vec18d> {
            if (!effect_pts[idx]) {
                return pre;
            } else {
                total_res += errors[idx].dot(errors[idx]);
                effective_num++;
                return std::pair<Mat18d, Vec18d>(pre.first + jacobians[idx].transpose() * jacobians[idx],
                                                 pre.second - jacobians[idx].transpose() * errors[idx]);
            }
        });

    HTVH = H_and_err.first;
    HTVr = H_and_err.second;

    return effective_num >= options_.min_effective_pts_;
}

bool Icp3dInc::ComputeResidualAndJacobiansPointToPlane(const SE3& pose, Mat18d& HTVH, Vec18d& HTVr) {
    assert(source_ && local_map_);

    // 对点的索引，预先生成
    std::vector<int> index(source_->points.size());
    std::iota(index.begin(), index.end(), 0);

    // 我们来写一些并发代码
    std::vector<bool> effect_pts(index.size(), false);
    std::vector<Eigen::Matrix<double, 1, 18>> jacobians(index.size());
    std::vector<double> errors(index.size());

    std::for_each(std::execution::par_unseq, index.begin(), index.end(),
                  [this, &pose, &index, &effect_pts, &jacobians, &errors](int idx) {
                      auto q = ToVec3d(source_->points[idx]);
                      Vec3d qs = pose * q;  // 转换之后的q
                      std::vector<int> nn;
                      kdtree_.GetClosestPoint(ToPointType(qs), nn, 5);  // 这里取5个最近邻
                      if (nn.size() > 3) {
                          // convert to eigen
                          std::vector<Vec3d> nn_eigen;
                          for (int i = 0; i < nn.size(); ++i) {
                              nn_eigen.emplace_back(ToVec3d(local_map_->points[nn[i]]));
                          }

                          Vec4d n;
                          if (!math::FitPlane(nn_eigen, n)) {
                              // 失败的不要
                              effect_pts[idx] = false;
                              return;
                          }

                          const double dis = n.head<3>().dot(qs) + n[3];
                          if (fabs(dis) > options_.max_plane_distance_) {
                              // 点离的太远了不要
                              effect_pts[idx] = false;
                              return;
                          }

                          effect_pts[idx] = true;

                          // build residual
                          Eigen::Matrix<double, 1, 18> J;
                          J.setZero();
                          J.block<1, 3>(0, 0) = n.head<3>().transpose();  // w.r.t p
                          J.block<1, 3>(0, 6) =
                              -n.head<3>().transpose() * pose.so3().matrix() * SO3::hat(q);  // w.r.t R

                          jacobians[idx] = J;
                          errors[idx] = dis;
                      } else {
                          effect_pts[idx] = false;
                      }
                  });

    // 累加Hessian和error,计算dx
    // 原则上可以用reduce并发，写起来比较麻烦，这里写成accumulate
    double total_res = 0;
    int effective_num = 0;
    const auto H_and_err = std::accumulate(
        index.begin(), index.end(), std::pair<Mat18d, Vec18d>(Mat18d::Zero(), Vec18d::Zero()),
        [&jacobians, &errors, &effect_pts, &total_res, &effective_num](const std::pair<Mat18d, Vec18d>& pre, int idx) {
            if (!effect_pts[idx]) {
                return pre;
            } else {
                total_res += errors[idx] * errors[idx];
                effective_num++;
                return std::pair<Mat18d, Vec18d>(pre.first + jacobians[idx].transpose() * jacobians[idx],
                                                 pre.second - jacobians[idx].transpose() * errors[idx]);
            }
        });

    HTVH = H_and_err.first;
    HTVr = H_and_err.second;

    return effective_num > options_.min_effective_pts_;
}

bool Icp3dInc::ComputeResidualAndJacobiansSeparate(const SE3& pose, Mat18d& HTVH, Vec18d& HTVr) {
    assert(local_map_edge_);
    assert(local_map_surf_);

    source_surf_ = CloudPtr(new PointCloudType);
    source_edge_ = CloudPtr(new PointCloudType);
    feature_extraction_->ExtractSurfAndOther(full_source_, source_edge_, source_surf_, options_.num_scans_);

    double total_res = 0;
    int effective_num = 0;
    HTVH.setZero();
    HTVr.setZero();

    if (options_.use_surf_points_) {
        const int surf_size = source_surf_->size();
        std::vector<bool> effect_surf(surf_size, false);
        std::vector<Eigen::Matrix<double, 1, 18>> jacob_surf(surf_size);  // 点面的残差是1维的
        std::vector<double> errors_surf(surf_size);
        std::vector<int> index_surf(surf_size);
        std::iota(index_surf.begin(), index_surf.end(), 0);  // 填入

        std::for_each(std::execution::par_unseq, index_surf.begin(), index_surf.end(),
                      [this, &pose, &effect_surf, &errors_surf, &jacob_surf](int idx) {
                          Vec3d q = ToVec3d(source_surf_->points[idx]);
                          Vec3d qs = pose * q;

                          // 检查最近邻
                          std::vector<int> nn_indices;

                          kdtree_surf_.GetClosestPoint(ToPointType(qs), nn_indices, 5);
                          effect_surf[idx] = false;

                          if (nn_indices.size() == 5) {
                              std::vector<Vec3d> nn_eigen;
                              for (auto& n : nn_indices) {
                                  nn_eigen.emplace_back(ToVec3d(local_map_surf_->points[n]));
                              }

                              // 点面残差
                              Vec4d n;
                              if (!math::FitPlane(nn_eigen, n)) {
                                  return;
                              }

                              double dis = n.head<3>().dot(qs) + n[3];
                              if (fabs(dis) > options_.max_plane_distance_) {
                                  return;
                              }

                              effect_surf[idx] = true;

                              // build residual
                              Eigen::Matrix<double, 1, 18> J;
                              J.setZero();
                              J.block<1, 3>(0, 6) = -n.head<3>().transpose() * pose.so3().matrix() * SO3::hat(q);
                              J.block<1, 3>(0, 0) = n.head<3>().transpose();

                              jacob_surf[idx] = J;
                              errors_surf[idx] = dis;
                          }
                      });

        for (const auto& idx : index_surf) {
            if (effect_surf[idx]) {
                HTVH += jacob_surf[idx].transpose() * jacob_surf[idx];
                HTVr += -jacob_surf[idx].transpose() * errors_surf[idx];
                effective_num++;
                total_res += errors_surf[idx] * errors_surf[idx];
            }
        }
    }

    if (options_.use_edge_points_) {
        const int edge_size = source_edge_->size();
        std::vector<bool> effect_edge(edge_size, false);
        std::vector<Eigen::Matrix<double, 3, 18>> jacob_edge(edge_size);  // 点线的残差是3维的
        std::vector<Vec3d> errors_edge(edge_size);
        std::vector<int> index_edge(edge_size);
        std::iota(index_edge.begin(), index_edge.end(), 0);  // 填入

        std::for_each(std::execution::par_unseq, index_edge.begin(), index_edge.end(),
                      [this, &pose, &effect_edge, &jacob_edge, &errors_edge](int idx) {
                          auto q = ToVec3d(source_edge_->points[idx]);
                          Vec3d qs = pose * q;  // 转换之后的q
                          std::vector<int> nn;
                          kdtree_.GetClosestPoint(ToPointType(qs), nn, 1);

                          if (!nn.empty()) {
                              Vec3d p = ToVec3d(local_map_edge_->points[nn[0]]);
                              double dis2 = (p - qs).squaredNorm();
                              if (dis2 > options_.max_nn_distance_) {
                                  // 点离的太远了不要
                                  effect_edge[idx] = false;
                                  return;
                              }

                              effect_edge[idx] = true;

                              // build residual
                              Vec3d e = p - qs;
                              Eigen::Matrix<double, 3, 18> J;
                              J.setZero();
                              J.block<3, 3>(0, 6) = pose.so3().matrix() * SO3::hat(q);
                              J.block<3, 3>(0, 0) = -Mat3d::Identity();

                              jacob_edge[idx] = J;
                              errors_edge[idx] = e;
                          } else {
                              effect_edge[idx] = false;
                          }
                      });

        for (const auto& idx : index_edge) {
            if (effect_edge[idx]) {
                HTVH += jacob_edge[idx].transpose() * jacob_edge[idx];
                HTVr += -jacob_edge[idx].transpose() * errors_edge[idx];
                effective_num++;
                total_res += errors_edge[idx].squaredNorm();
            }
        }
    }

    return effective_num > options_.min_effective_pts_;
}

void Icp3dInc::AddCloud(CloudPtr cloud_world) {
    if (cloud_world->size() < options_.min_surf_pts_) {
        LOG(ERROR) << "not enough pts: " << cloud_world->size();
        return;
    }

    if (local_map_ == nullptr) {
        // 首帧特殊处理
        local_map_ = cloud_world;
        kdtree_.BuildTree(local_map_);
        clouds_.emplace_back(cloud_world);
        return;
    }

    // 重建local map
    clouds_.emplace_back(cloud_world);
    if (clouds_.size() > options_.num_kfs_in_local_map_) {
        clouds_.pop_front();
    }

    local_map_.reset(new PointCloudType);
    for (auto& s : clouds_) {
        *local_map_ += *s;
    }

    local_map_ = VoxelCloud(local_map_, options_.voxel_size_);

    LOG(INFO) << "insert keyframe, surf pts: " << local_map_->size();

    kdtree_.BuildTree(local_map_);
}

void Icp3dInc::AddCloudSeparate(const SE3& pose) {
    assert(source_edge_ && source_surf_);
    assert(local_map_edge_ && local_map_surf_);

    CloudPtr edge_world(new PointCloudType);
    CloudPtr surf_world(new PointCloudType);
    pcl::transformPointCloud(*source_edge_, *edge_world, pose.matrix());
    pcl::transformPointCloud(*source_surf_, *surf_world, pose.matrix());

    // 重建local map
    edges_.emplace_back(edge_world);
    surfs_.emplace_back(surf_world);

    if (edges_.size() > options_.num_kfs_in_local_map_) {
        edges_.pop_front();
    }
    if (surfs_.size() > options_.num_kfs_in_local_map_) {
        surfs_.pop_front();
    }

    local_map_surf_.reset(new PointCloudType);
    local_map_edge_.reset(new PointCloudType);

    for (auto& s : edges_) {
        *local_map_edge_ += *s;
    }
    for (auto& s : surfs_) {
        *local_map_surf_ += *s;
    }

    local_map_surf_ = VoxelCloud(local_map_surf_, options_.voxel_size_);
    local_map_edge_ = VoxelCloud(local_map_edge_, options_.voxel_size_);

    kdtree_surf_.BuildTree(local_map_surf_);
    kdtree_edge_.BuildTree(local_map_edge_);
}

void Icp3dInc::AddFirstFullCloud(FullCloudPtr full_cloud) {
    source_surf_ = CloudPtr(new PointCloudType);
    source_edge_ = CloudPtr(new PointCloudType);
    feature_extraction_->ExtractSurfAndOther(full_cloud, source_edge_, source_surf_, options_.num_scans_);

    local_map_edge_ = source_edge_;
    local_map_surf_ = source_surf_;

    kdtree_edge_.BuildTree(local_map_edge_);
    kdtree_surf_.BuildTree(local_map_surf_);

    edges_.emplace_back(source_edge_);
    surfs_.emplace_back(source_surf_);
}

}  // namespace sad
