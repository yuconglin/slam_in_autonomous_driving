//
// Created by xiang on 2021/9/22.
//

#ifndef SLAM_IN_AUTO_DRIVING_KDTREE_TEMPLATE_H
#define SLAM_IN_AUTO_DRIVING_KDTREE_TEMPLATE_H

#include "common/eigen_types.h"
#include "common/math_utils.h"
#include "common/point_types.h"

#include <glog/logging.h>
#include <execution>
#include <map>
#include <queue>

namespace sad {

/// Kd树节点，二叉树结构，内部用祼指针，对外一个root的shared_ptr
struct KdTreeTempNode {
    int id_ = -1;
    int point_idx_ = 0;                // 点的索引
    int axis_index_ = 0;               // 分割轴
    float split_thresh_ = 0.0;         // 分割位置
    KdTreeTempNode* left_ = nullptr;   // 左子树
    KdTreeTempNode* right_ = nullptr;  // 右子树

    bool IsLeaf() const { return left_ == nullptr && right_ == nullptr; }  // 是否为叶子
};

/// 用于记录knn结果
struct NodeAndDistanceTemp {
    NodeAndDistanceTemp(KdTreeTempNode* node, float dis2) : node_(node), distance2_(dis2) {}
    KdTreeTempNode* node_ = nullptr;
    float distance2_ = 0;  // 平方距离，用于比较

    bool operator<(const NodeAndDistanceTemp& other) const { return distance2_ < other.distance2_; }
};

/**
 * 手写kd树
 * 测试这个kd树的召回!
 */
class KdTreeTemp {
   public:
    explicit KdTreeTemp() = default;
    ~KdTreeTemp() { Clear(); }

    template <typename PointType>
    bool BuildTree(const typename pcl::PointCloud<PointType>::Ptr& cloud) {
        if (cloud->empty()) {
            return false;
        }

        cloud_.clear();
        cloud_.resize(cloud->size());
        for (size_t i = 0; i < cloud->points.size(); ++i) {
            cloud_[i] = ToVec3f(cloud->points[i]);
        }

        Clear();
        Reset();

        IndexVec idx(cloud->size());
        for (int i = 0; i < cloud->points.size(); ++i) {
            idx[i] = i;
        }

        Insert(idx, root_.get());
        return true;
    }

    /// 获取k最近邻
    template <typename PointType>
    bool GetClosestPoint(const PointType& pt, std::vector<int>& closest_idx, int k = 5) {
        if (k > size_) {
            LOG(ERROR) << "cannot set k larger than cloud size: " << k << ", " << size_;
            return false;
        }
        k_ = k;

        std::priority_queue<NodeAndDistanceTemp> knn_result;
        Knn(ToVec3f(pt), root_.get(), knn_result);

        // 排序并返回结果
        closest_idx.resize(knn_result.size());
        for (int i = closest_idx.size() - 1; i >= 0; --i) {
            // 倒序插入
            closest_idx[i] = knn_result.top().node_->point_idx_;
            knn_result.pop();
        }
        return true;
    }

    /// 并行为点云寻找最近邻
    template <typename PointType>
    bool GetClosestPointMT(const typename pcl::PointCloud<PointType>::Ptr& cloud,
                           std::vector<std::pair<size_t, size_t>>& matches, int k = 5) {
        matches.clear();
        matches.resize(cloud->size() * k);

        // 索引
        std::vector<int> index(cloud->size());
        for (int i = 0; i < cloud->points.size(); ++i) {
            index[i] = i;
        }

        std::for_each(std::execution::par_unseq, index.begin(), index.end(), [this, &cloud, &matches, &k](int idx) {
            std::vector<int> closest_idx;
            GetClosestPoint(cloud->points[idx], closest_idx, k);
            for (int i = 0; i < k; ++i) {
                matches[idx * k + i].second = idx;
                if (i < closest_idx.size()) {
                    matches[idx * k + i].first = closest_idx[i];
                } else {
                    matches[idx * k + i].first = math::kINVALID_ID;
                }
            }
        });

        return true;
    }

    /// 这个被用于计算最近邻的倍数
    void SetEnableANN(bool use_ann = true, float alpha = 0.1) {
        approximate_ = use_ann;
        alpha_ = alpha;
    }

    /// 返回节点数量
    size_t size() const { return size_; }

    /// 清理数据
    void Clear();

    /// 打印所有节点信息
    void PrintAll();

   private:
    /// kdtree 构建相关
    /**
     * 在node处插入点
     * @param points
     * @param node
     */
    void Insert(const IndexVec& points, KdTreeTempNode* node);

    /**
     * 计算点集的分割面
     * @param points 输入点云
     * @param axis   轴
     * @param th     阈值
     * @param left   左子树
     * @param right  右子树
     * @return
     */
    bool FindSplitAxisAndThresh(const IndexVec& point_idx, int& axis, float& th, IndexVec& left, IndexVec& right);

    void Reset();

    /// 两个点的平方距离
    static inline float Dis2(const Vec3f& p1, const Vec3f& p2) { return (p1 - p2).squaredNorm(); }

    // Knn 相关
    /**
     * 检查给定点在kdtree node上的knn，可以递归调用
     * @param pt     查询点
     * @param node   kdtree 节点
     */
    void Knn(const Vec3f& pt, KdTreeTempNode* node, std::priority_queue<NodeAndDistanceTemp>& result) const;

    /**
     * 对叶子节点，计算它和查询点的距离，尝试放入结果中
     * @param pt    查询点
     * @param node  Kdtree 节点
     */
    void ComputeDisForLeaf(const Vec3f& pt, KdTreeTempNode* node,
                           std::priority_queue<NodeAndDistanceTemp>& result) const;

    /**
     * 检查node下是否需要展开
     * @param pt   查询点
     * @param node Kdtree 节点
     * @return true if 需要展开
     */
    bool NeedExpand(const Vec3f& pt, KdTreeTempNode* node, std::priority_queue<NodeAndDistanceTemp>& knn_result) const;

    int k_ = 5;                                       // knn最近邻数量
    std::shared_ptr<KdTreeTempNode> root_ = nullptr;  // 叶子节点
    std::vector<Vec3f> cloud_;                        // 输入点云
    std::unordered_map<int, KdTreeTempNode*> nodes_;  // for bookkeeping

    size_t size_ = 0;       // 叶子节点数量
    int tree_node_id_ = 0;  // 为kdtree node 分配id

    // 近似最近邻
    bool approximate_ = true;
    float alpha_ = 0.1;
};

}  // namespace sad

#endif  // SLAM_IN_AUTO_DRIVING_KDTREE_H
