// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// @author Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2020 Ignacio Vizzo, Cyrill Stachniss, University of Bonn.
// ----------------------------------------------------------------------------

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>
#include <iostream>

#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/Keypoint.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/utility/Eigen.h"
#include "open3d/utility/Logging.h"

namespace open3d {

namespace {


// double ComputeModelResolution(const std::vector<Eigen::Vector3d>& points,
//                               const geometry::KDTreeFlann& kdtree) {
//     std::vector<int> indices(2);
//     std::vector<double> distances(2);
//     const double resolution = std::accumulate(
//             points.begin(), points.end(), 0.,
//             [&](double state, const Eigen::Vector3d& point) {
//                 if (kdtree.SearchKNN(point, 2, indices, distances) >= 2) {
//                     state += std::sqrt(distances[1]);
//                 }
//                 return state;
//             });

//     return resolution / static_cast<double>(points.size());
// }

}  // namespace

namespace geometry {
void PointCloud::EstimateISS(const KDTreeSearchParam &search_param /* = KDTreeSearchParamKNN()*/,
                    double salient_radius /* = 0.0*/,
                    // double gamma_21 /* = 0.975*/,
                    // double gamma_32 /* = 0.975*/,
                    int min_neighbors /* = 5*/){

    if (this->points_.empty()) {
        utility::LogWarning("[EstimateISS] Input PointCloud is empty!");
        // return std::make_shared<PointCloud>();
    }
    const auto& points = this->points_;
    // std::cout << "points.size(): " << points.size() << std::endl;
    KDTreeFlann kdtree(*this);

    // if (salient_radius == 0.0 || non_max_radius == 0.0) {
    //     const double resolution = ComputeModelResolution(points, kdtree);
    //     salient_radius = 6 * resolution;
    //     non_max_radius = 4 * resolution;
    //     utility::LogDebug(
    //             "[EstimateISS] Computed salient_radius = {}, "
    //             "non_max_radius = {} from input model",
    //             salient_radius, non_max_radius);
    // }

    this->eigen_values_.resize(points.size());
    this->covariances_.resize(points.size());
    this->eigen_vectors_.resize(points.size());
#pragma omp parallel for schedule(static) shared(eigen_values_, covariances_)
    for (int i = 0; i < (int)points.size(); i++) {

        std::vector<int> indices;
        std::vector<double> dist;
        int nb_neighbors =
                kdtree.SearchRadius(points[i], salient_radius, indices, dist);
        // std::cout << i << " nb_neighbors: " << nb_neighbors << std::endl;
        if (nb_neighbors < min_neighbors) {
            continue;
        }

        Eigen::Matrix3d cov = utility::ComputeCovariance(points, indices);
        covariances_[i] = cov;
        if (cov.isZero()) {
            std::vector<double> eigen_values = {-1, -1, -1};
            continue;
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
        const double& e1c = solver.eigenvalues()[2];
        const double& e2c = solver.eigenvalues()[1];
        const double& e3c = solver.eigenvalues()[0];
        // std::cout << "eigen_values: " << e1c << " " << e2c << " " << e3c << std::endl;
        Eigen::Vector3d eigen_values = {e1c, e2c, e3c};
        eigen_values_[i] = eigen_values;

        const Eigen::Matrix3d& eigenvectors = solver.eigenvectors();
        eigen_vectors_[i] = eigenvectors;
        // if ((e2c / e1c) < gamma_21 && e3c / e2c < gamma_32) {
        //     third_eigen_values[i] = e3c;
        // }
    }

//     std::vector<size_t> kp_indices;
//     kp_indices.reserve(points.size());
// #pragma omp parallel for schedule(static) shared(kp_indices)
//     for (int i = 0; i < (int)points.size(); i++) {
//         if (third_eigen_values[i] > 0.0) {
//             // std::vector<int> nn_indices;
//             // std::vector<double> dist;
//             // int nb_neighbors = kdtree.SearchRadius(points[i], non_max_radius,
//             //                                        nn_indices, dist);

//             // if (nb_neighbors >= min_neighbors)
//                 // IsLocalMaxima(i, nn_indices, third_eigen_values)) {
// #pragma omp critical
//                 kp_indices.emplace_back(i);
//             // }
//         }
//     }

//     utility::LogDebug("[EstimateISS] Extracted {} keypoints",
//                       kp_indices.size());
    // return this->SelectByIndex(kp_indices);
}

// }  // namespace keypoint
}  // namespace geometry
}  // namespace open3d
