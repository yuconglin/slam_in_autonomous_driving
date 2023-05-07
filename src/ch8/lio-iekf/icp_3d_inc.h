#ifndef SLAM_IN_AUTO_DRIVING_ICP_3D_INC_H
#define SLAM_IN_AUTO_DRIVING_ICP_3D_INC_H

#include <deque>

#include "ch5/kdtree.h"
#include "ch7/icp_3d.h"
#include "common/eigen_types.h"
#include "common/point_types.h"

namespace sad {

class FeatureExtraction;
/*
3D ICP with accumulated local point cloud map.
Only used in IEKF based LIO.
*/
class Icp3dInc {
   public:
    struct Options {
        Options() {}

        int min_edge_pts_ = 20;          // 最小边缘点数
        int min_surf_pts_ = 20;          // 最小平面点数
        int num_kfs_in_local_map_ = 30;  // 局部地图含有多少个关键帧

        // ICP 参数
        double max_nn_distance_ = 1.0;      // 点到点最近邻查找时阈值
        double max_plane_distance_ = 0.05;  // 平面最近邻查找时阈值
        double max_line_distance_ = 0.5;    // 点线最近邻查找时阈值
        int min_effective_pts_ = 10;        // 最近邻点数阈值
        double eps_ = 1e-3;                 // 收敛判定条件

        double voxel_size_ = 1.0;
        int num_scans_ = 32;

        bool use_surf_points_ = true;
        bool use_edge_points_ = false;  // not working well for NCLT dataset.
    };

    explicit Icp3dInc(const Options& options = Options());

    /**
     * 计算给定Pose下的雅可比和残差矩阵，符合IEKF中符号（8.17, 8.19）
     * @param pose
     * @param HTVH
     * @param HTVr
     */
    bool ComputeResidualAndJacobiansPointToPoint(const SE3& init_pose, Mat18d& HTVH, Vec18d& HTVr);

    bool ComputeResidualAndJacobiansPointToPlane(const SE3& init_pose, Mat18d& HTVH, Vec18d& HTVr);

    bool ComputeResidualAndJacobiansSeparate(const SE3& init_pose, Mat18d& HTVH, Vec18d& HTVr);

    void AddCloud(CloudPtr cloud_world);
    void AddCloudSeparate(const SE3& pose);
    void AddFirstFullCloud(FullCloudPtr full_cloud);

    void SetSource(CloudPtr source) { source_ = source; }
    void SetFullSource(FullCloudPtr full_source) { full_source_ = full_source; }

    void SetScanNumber(int num_scans) { options_.num_scans_ = num_scans; }

   private:
    Options options_;

    CloudPtr local_map_ = nullptr;  // 局部地图的local map
    CloudPtr local_map_edge_ = nullptr;
    CloudPtr local_map_surf_ = nullptr;  // 局部地图的local map

    std::deque<CloudPtr> clouds_;  // total clouds
    std::deque<CloudPtr> edges_;
    std::deque<CloudPtr> surfs_;  // 缓存的角点和平面点

    KdTree kdtree_;
    KdTree kdtree_edge_;
    KdTree kdtree_surf_;

    CloudPtr source_ = nullptr;
    FullCloudPtr full_source_ = nullptr;
    CloudPtr source_edge_ = nullptr;
    CloudPtr source_surf_ = nullptr;

    std::shared_ptr<FeatureExtraction> feature_extraction_ = nullptr;
};
}  // namespace sad

#endif
