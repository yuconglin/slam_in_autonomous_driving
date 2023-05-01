//
// Created by xiang on 2022/7/21.
//
#include "ch7/loam-like/feature_extraction.h"
#include "common/math_utils.h"

#include <execution>

#include <glog/logging.h>

namespace sad {
namespace {
constexpr double kGroundThreshold = 0.1;  // [m]

void SeparateGround(FullCloudPtr pc_in, CloudPtr pc_out_ground, FullCloudPtr pc_out_rest) {
    std::vector<int> low_indice;
    std::vector<Vec3d> low_points;
    for (int i = 0; i < pc_in->points.size(); ++i) {
        const auto &pt = pc_in->points[i];
        if (pt.z < kGroundThreshold) {
            low_indice.emplace_back(i);
            low_points.emplace_back(pt.x, pt.y, pt.z);
        } else {
            pc_out_rest->push_back(pt);
        }
    }

    LOG(INFO) << "low points size: " << low_points.size();
    Vec4d plane_coeffs;
    if (!math::FitPlane(low_points, plane_coeffs, /*very big threshold*/ 1e2)) {
        // all the points will be included in pc_out_rest.
        LOG(INFO) << "no points were close to the plane !";
        for (int idx : low_indice) {
            pc_out_rest->push_back(pc_in->points[idx]);
        }
        return;
    }

    for (int j = 0; j < low_indice.size(); ++j) {
        // get the points close to the plane
        const double err = low_points[j].dot(plane_coeffs.head<3>()) + plane_coeffs[3];
        if (err * err < 0.04) {
            const auto &pt = pc_in->points[low_indice[j]];
            PointType p;
            p.x = pt.x;
            p.y = pt.y;
            p.z = pt.z;
            p.intensity = pt.intensity;
            pc_out_ground->push_back(p);
        } else {
            pc_out_rest->push_back(pc_in->points[low_indice[j]]);
        }
    }
}
}  // namespace

void FeatureExtraction::Extract(FullCloudPtr pc_in, CloudPtr pc_out_edge, CloudPtr pc_out_surf) {
    int num_scans = 16;
    std::vector<CloudPtr> scans_in_each_line;  // 分线数的点云
    for (int i = 0; i < num_scans; i++) {
        scans_in_each_line.emplace_back(new PointCloudType);
    }

    for (auto &pt : pc_in->points) {
        assert(pt.ring >= 0 && pt.ring < num_scans);
        PointType p;
        p.x = pt.x;
        p.y = pt.y;
        p.z = pt.z;
        p.intensity = pt.intensity;

        scans_in_each_line[pt.ring]->points.emplace_back(p);
    }

    // 处理曲率
    for (int i = 0; i < num_scans; i++) {
        if (scans_in_each_line[i]->points.size() < 131) {
            continue;
        }
        std::vector<IdAndValue> cloud_curvature;  // 每条线对应的曲率
        int total_points = scans_in_each_line[i]->points.size() - 10;
        for (int j = 5; j < (int)scans_in_each_line[i]->points.size() - 5; j++) {
            // 两头留一定余量，采样周围10个点取平均值
            double diffX = scans_in_each_line[i]->points[j - 5].x + scans_in_each_line[i]->points[j - 4].x +
                           scans_in_each_line[i]->points[j - 3].x + scans_in_each_line[i]->points[j - 2].x +
                           scans_in_each_line[i]->points[j - 1].x - 10 * scans_in_each_line[i]->points[j].x +
                           scans_in_each_line[i]->points[j + 1].x + scans_in_each_line[i]->points[j + 2].x +
                           scans_in_each_line[i]->points[j + 3].x + scans_in_each_line[i]->points[j + 4].x +
                           scans_in_each_line[i]->points[j + 5].x;
            double diffY = scans_in_each_line[i]->points[j - 5].y + scans_in_each_line[i]->points[j - 4].y +
                           scans_in_each_line[i]->points[j - 3].y + scans_in_each_line[i]->points[j - 2].y +
                           scans_in_each_line[i]->points[j - 1].y - 10 * scans_in_each_line[i]->points[j].y +
                           scans_in_each_line[i]->points[j + 1].y + scans_in_each_line[i]->points[j + 2].y +
                           scans_in_each_line[i]->points[j + 3].y + scans_in_each_line[i]->points[j + 4].y +
                           scans_in_each_line[i]->points[j + 5].y;
            double diffZ = scans_in_each_line[i]->points[j - 5].z + scans_in_each_line[i]->points[j - 4].z +
                           scans_in_each_line[i]->points[j - 3].z + scans_in_each_line[i]->points[j - 2].z +
                           scans_in_each_line[i]->points[j - 1].z - 10 * scans_in_each_line[i]->points[j].z +
                           scans_in_each_line[i]->points[j + 1].z + scans_in_each_line[i]->points[j + 2].z +
                           scans_in_each_line[i]->points[j + 3].z + scans_in_each_line[i]->points[j + 4].z +
                           scans_in_each_line[i]->points[j + 5].z;
            IdAndValue distance(j, diffX * diffX + diffY * diffY + diffZ * diffZ);
            cloud_curvature.push_back(distance);
        }
        /*
        // From j = 5 to scans_in_each_line[i]->points.size() - 5 (end not included).
        const int total_points = scans_in_each_line[i]->points.size() - 10;
        std::vector<IdAndValue> cloud_curvature(total_points);
        std::vector<int> scan_indice(total_points);
        std::iota(scan_indice.begin(), scan_indice.end(), 5);
        std::for_each(std::execution::par_unseq, scan_indice.begin(), scan_indice.end(), [&](int j) {
            assert(j >= 5 && j + 5 < total_points + 10);
            // 两头留一定余量，采样周围10个点取平均值
            const double diffX = scans_in_each_line[i]->points[j - 5].x + scans_in_each_line[i]->points[j - 4].x +
                                 scans_in_each_line[i]->points[j - 3].x + scans_in_each_line[i]->points[j - 2].x +
                                 scans_in_each_line[i]->points[j - 1].x - 10 * scans_in_each_line[i]->points[j].x +
                                 scans_in_each_line[i]->points[j + 1].x + scans_in_each_line[i]->points[j + 2].x +
                                 scans_in_each_line[i]->points[j + 3].x + scans_in_each_line[i]->points[j + 4].x +
                                 scans_in_each_line[i]->points[j + 5].x;
            const double diffY = scans_in_each_line[i]->points[j - 5].y + scans_in_each_line[i]->points[j - 4].y +
                                 scans_in_each_line[i]->points[j - 3].y + scans_in_each_line[i]->points[j - 2].y +
                                 scans_in_each_line[i]->points[j - 1].y - 10 * scans_in_each_line[i]->points[j].y +
                                 scans_in_each_line[i]->points[j + 1].y + scans_in_each_line[i]->points[j + 2].y +
                                 scans_in_each_line[i]->points[j + 3].y + scans_in_each_line[i]->points[j + 4].y +
                                 scans_in_each_line[i]->points[j + 5].y;
            const double diffZ = scans_in_each_line[i]->points[j - 5].z + scans_in_each_line[i]->points[j - 4].z +
                                 scans_in_each_line[i]->points[j - 3].z + scans_in_each_line[i]->points[j - 2].z +
                                 scans_in_each_line[i]->points[j - 1].z - 10 * scans_in_each_line[i]->points[j].z +
                                 scans_in_each_line[i]->points[j + 1].z + scans_in_each_line[i]->points[j + 2].z +
                                 scans_in_each_line[i]->points[j + 3].z + scans_in_each_line[i]->points[j + 4].z +
                                 scans_in_each_line[i]->points[j + 5].z;
            cloud_curvature[j - 5] = IdAndValue(j, diffX * diffX + diffY * diffY + diffZ * diffZ);
        });
        */
        // 对每个区间选取特征，把360度分为6个区间
        for (int j = 0; j < 6; j++) {
            int sector_length = (int)(total_points / 6);
            int sector_start = sector_length * j;
            int sector_end = sector_length * (j + 1) - 1;
            if (j == 5) {
                sector_end = total_points - 1;
            }

            std::vector<IdAndValue> sub_cloud_curvature(cloud_curvature.begin() + sector_start,
                                                        cloud_curvature.begin() + sector_end);

            ExtractFromSector(scans_in_each_line[i], sub_cloud_curvature, pc_out_edge, pc_out_surf);
        }
    }
}

void FeatureExtraction::ExtractWithGround(FullCloudPtr pc_in, CloudPtr pc_out_ground, CloudPtr pc_out_edge,
                                          CloudPtr pc_out_surf) {
    // Separate the ground and non-ground points.
    CloudPtr current_surf(new PointCloudType);
    FullCloudPtr pc_out_rest(new FullPointCloudType);
    SeparateGround(pc_in, pc_out_ground, pc_out_rest);
    Extract(pc_out_rest, pc_out_edge, pc_out_surf);
}

void FeatureExtraction::ExtractFromSector(const CloudPtr &pc_in, std::vector<IdAndValue> &cloud_curvature,
                                          CloudPtr &pc_out_edge, CloudPtr &pc_out_surf) {
    // 按曲率排序
    std::sort(cloud_curvature.begin(), cloud_curvature.end(),
              [](const IdAndValue &a, const IdAndValue &b) { return a.value_ < b.value_; });

    int largest_picked_num = 0;
    int point_info_count = 0;

    /// 按照曲率最大的开始搜，选取曲率最大的角点
    std::vector<int> picked_points;  // 标记被选中的角点，角点附近的点都不会被选取
    for (int i = cloud_curvature.size() - 1; i >= 0; i--) {
        int ind = cloud_curvature[i].id_;
        if (std::find(picked_points.begin(), picked_points.end(), ind) == picked_points.end()) {
            if (cloud_curvature[i].value_ <= 0.1) {
                break;
            }

            largest_picked_num++;
            picked_points.push_back(ind);

            if (largest_picked_num <= 20) {
                pc_out_edge->push_back(pc_in->points[ind]);
                point_info_count++;
            } else {
                break;
            }

            for (int k = 1; k <= 5; k++) {
                double diffX = pc_in->points[ind + k].x - pc_in->points[ind + k - 1].x;
                double diffY = pc_in->points[ind + k].y - pc_in->points[ind + k - 1].y;
                double diffZ = pc_in->points[ind + k].z - pc_in->points[ind + k - 1].z;
                if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
                    break;
                }
                picked_points.push_back(ind + k);
            }
            for (int k = -1; k >= -5; k--) {
                double diffX = pc_in->points[ind + k].x - pc_in->points[ind + k + 1].x;
                double diffY = pc_in->points[ind + k].y - pc_in->points[ind + k + 1].y;
                double diffZ = pc_in->points[ind + k].z - pc_in->points[ind + k + 1].z;
                if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
                    break;
                }
                picked_points.push_back(ind + k);
            }
        }
    }

    /// 选取曲率较小的平面点
    for (int i = 0; i <= (int)cloud_curvature.size() - 1; i++) {
        int ind = cloud_curvature[i].id_;
        if (std::find(picked_points.begin(), picked_points.end(), ind) == picked_points.end()) {
            pc_out_surf->push_back(pc_in->points[ind]);
        }
    }
}

}  // namespace sad