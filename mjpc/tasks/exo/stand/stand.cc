// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/tasks/exo/stand/stand.h"

#include <string>
#include <cmath>
#include <map>
#include <vector>
#include <Eigen/Dense>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

// Get time module
#include <chrono>

namespace mjpc::exo {
  
std::string Stand::XmlPath() const {
  return mjpc::GetModelPath("exo/stand/task.xml");
}
std::string Stand::Name() const { return "Exo Stand"; }

// ------------------ Residuals for humanoid stand task ------------
//   Number of residuals: 8
//     Residual (0) : Balance (dim=1)
//     Residual (1) : Centroidal (dim=1)
//     Residual (2) : CoM Vel (dim=2)
//     Residual (3) : Joint Vel (dim=10)
//     Residual (4) : Control (dim=8)
//     Residual (5) : Feet Level (dim=1)
//     Residual (6) : Balance Stance (dim=10)
//     Residual (7) : Centroidal Momentum (dim=3)
// ----------------------------------------------------------------

void Stand::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                 double* residual) const {


  int counter = 0;

  double* backpack_orientation = SensorByName(model, data, "backpack_orientation");
  double* torso_angular_velocity = SensorByName(model, data, "vel_imu_backpack");
  double* torso_linear_acceleration = SensorByName(model, data, "acc_imu_backpack");

  // ----- Torso Angle ----- //
  // ----- Minimize the angle of the torso wrt the z-axis ----- //
  // Calculate the angle of the torso wrt the z-axis from the quaternion angles 
  // and add to residual
  double w = backpack_orientation[0];
  double x = backpack_orientation[1];
  double y = backpack_orientation[2];
  double z = backpack_orientation[3];

  double torso_angle = -M_PI/2 + 2*atan2(sqrt(1+2*(w*y-x*z)), sqrt(1-2*(w*y-x*z)));
  residual[counter++] = torso_angle;

  // ----- Joint Velocity ----- //
  // ----- Minimize the velocity of the joints ----- //
  // TODO: Do not include the humanoid joints in the residual

  std::vector<std::string> humanoid_joints = {"right_shoulder",
                                              "right_shoulder2",
                                              "right_elbow",
                                              "right_wrist",
                                              "left_shoulder",
                                              "left_shoulder2",
                                              "left_elbow",
                                              "left_wrist"};

  // Get the indices of the humanoid joints
  std::vector<int> humanoid_joint_indices;
  for (auto& joint_name : humanoid_joints) {
    int joint_id = mj_name2id(model, mjOBJ_JOINT, joint_name.c_str());
    humanoid_joint_indices.push_back(joint_id);
  }


  // Add all but the humanoid joints to the residual. The humanoid joints are the first 6 joints
  for (int i = 1; i < model->nv - 5; ++i) {
    if (std::find(humanoid_joint_indices.begin(), humanoid_joint_indices.end(), i) == humanoid_joint_indices.end()) {
      residual[counter++] = data->qvel[i];
    } 
  }

  // mju_copy(&residual[counter], data->qvel + 6, model->nv - 6);
  // counter += model->nv - 6;
  
  // ----- Control ----- //
  // ----- Minimize the amount of control needed ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;

  // ----- Balanced Stance ----- //
  // ----- Enforce a stable stance ----- //
  std::map<std::string, double> stable_stance = {
                                                {"left_ankle_dpf", 2.635804130064104411e-01},
                                                {"left_hip_aa", -8.473640292014880671e-02},
                                                {"left_hip_fe", 1.518568016167828327e-01},
                                                {"left_knee", 4.164058390732758852e-01},
                                                {"right_ankle_dpf", 2.635804130064104411e-01},
                                                {"right_hip_aa", -8.473640292014880671e-02},
                                                {"right_hip_fe", 1.518568016167828327e-01},
                                                {"right_knee", 4.164058390732758852e-01},
                                                {"left_ankle_ie", 0},
                                                {"right_ankle_ie", 0}};

  for (auto& [joint_name, stable_angle] : stable_stance) {
    int joint_id = mj_name2id(model, mjOBJ_JOINT, joint_name.c_str());
    int qpos_id = model->jnt_qposadr[joint_id];
    double actual_angle = data->qpos[qpos_id];
    residual[counter++] = abs(stable_angle - actual_angle);
  }

  // ----- Torso angular velocity ----- //
  // ----- Minimize the angular velocity of the torso ----- //
  mju_copy(&residual[counter], torso_angular_velocity, 3);
  counter += 3;

  // ----- Torso linear acceleration ----- //
  // ----- Minimize the linear acceleration of the torso ----- //
  mju_copy(&residual[counter], torso_linear_acceleration, 3);
  counter += 3;



  // ----- Centroidal Momentum Terms ----- //
  // Define a 6NxN matrix A where N is the number of joints
  // This matrix is used to calculate the centroidal momentum of the robot
  int num_links = model->nbody - 1; // -1 for the world body
  
  // Initialize matrices with correct sizes
  Eigen::MatrixXd I = Eigen::MatrixXd::Zero(model->nv, model->nv);  // Full inertia matrix
  mj_fullM(model, I.data(), data->qM);

  // Initialize Jacobian vectors with correct size (3 dimensions * nv DOFs)
  Eigen::VectorXd jacp(3 * model->nv);
  Eigen::VectorXd jacr(3 * model->nv);

  // Initialize momentum matrix (6 rows per body)
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6 * num_links, model->nv);

  // Initialize the rest of the matrices used in the calculation
  Eigen::MatrixXd I_i(6, 6);
  Eigen::MatrixXd J_i(6, model->nv);
  Eigen::Matrix3d inertia;
  Eigen::MatrixXd A_i(6, model->nv);

  // Initialize th location of the body inertia
  double* body_inertia;
  
  // For each joint, compute A_i = I_i * J_i
  for (int i = 1; i < model->nbody; i++) {
    // Compute the Jacobian of current link
    mj_jacBodyCom(model, data, jacp.data(), jacr.data(), i);

    // Reshape Jacobians into 3xnv matrices
    Eigen::Map<Eigen::MatrixXd> Jp(jacp.data(), 3, model->nv);
    Eigen::Map<Eigen::MatrixXd> Jr(jacr.data(), 3, model->nv);

    // Reshape the jacobian to 6xnv
    J_i.topRows(3) = Jr;
    J_i.bottomRows(3) = Jp;

    // Note: The body inertia does not change for each body, so we can just get it once
    // Get body's spatial inertia (6x6 matrix)
    body_inertia = &model->body_inertia[i*6];
    inertia << body_inertia[0], body_inertia[1], body_inertia[2],
                body_inertia[1], body_inertia[3], body_inertia[4],
                body_inertia[2], body_inertia[4], body_inertia[5];

    // Construct the spatial inertia matrix
    I_i.setZero();
    I_i.block<3,3>(0,0) = inertia;
    I_i.block<3,3>(3,3) = model->body_mass[i] * Eigen::Matrix3d::Identity();

    // Compute A_i = I_i * J_i
    A_i = I_i * J_i;

    // Store the result in the system momentum matrix A
    A.block(6 * (i - 1), 0, 6, model->nv) = A_i;
  }

  // Centroidal momentum terms
  // Compute h = A*v and add its norm to the residual
  Eigen::Map<Eigen::VectorXd> v(data->qvel, model->nv);
  Eigen::VectorXd h = A * v;
  Eigen::Map<Eigen::VectorXd> v_dot(data->qacc, model->nv);
  Eigen::VectorXd h_dot = A * v_dot;
  residual[counter++] = h.norm() + h_dot.norm();

  // sensor dim sanity check
  // TODO: use this pattern everywhere and make this a utility function
  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error(
        "mismatch between total user-sensor dimension %d and actual length of residual %d",
        user_sensor_dim, counter);
  }
}

}  // namespace mjpc::humanoid

// Previous code
// // Get positions of feet sensors
  // double* f1_position = SensorByName(model, data, "sp0"); // left foot back right
  // double* f2_position = SensorByName(model, data, "sp1"); // left foot back left
  // double* f3_position = SensorByName(model, data, "sp2"); // left foot front right
  // double* f4_position = SensorByName(model, data, "sp3"); // left foot front left
  // double* f5_position = SensorByName(model, data, "sp4"); // right foot back right
  // double* f6_position = SensorByName(model, data, "sp5"); // right foot back left
  // double* f7_position = SensorByName(model, data, "sp6"); // right foot front right
  // double* f8_position = SensorByName(model, data, "sp7"); // right foot front left
  
  // Get position, velocity, and orientation of the center of mass
  // double* com_position = SensorByName(model, data, "torso_subtreecom");  
  // double* com_velocity = SensorByName(model, data, "torso_subtreelinvel");

  // double* torso_angular_momentum = SensorByName(model, data, "torso_subtreeangmom");
  // double* leftleg_angular_momentum = SensorByName(model, data, "leftleg_subtreeangmom");
  // double* rightleg_angular_momentum = SensorByName(model, data, "rightleg_subtreeangmom");

  // ----- Balance: CoM-feet xy error ----- //

  // // Calculate the capture point = CoM + CoM velocity * kFallTime
  // double kFallTime = 0.2;                                                       
  // double capture_point[3] = {com_position[0], com_position[1], com_position[2]}; 
  // mju_addToScl3(capture_point, com_velocity, kFallTime);

  // // Calculate the average feet xy position
  // double fxy_avg[2] = {0.0, 0.0};
  // mju_addTo(fxy_avg, f1_position, 2);
  // mju_addTo(fxy_avg, f2_position, 2);
  // mju_addTo(fxy_avg, f3_position, 2);
  // mju_addTo(fxy_avg, f4_position, 2);
  // mju_addTo(fxy_avg, f5_position, 2);
  // mju_addTo(fxy_avg, f6_position, 2);
  // mju_addTo(fxy_avg, f7_position, 2);
  // mju_addTo(fxy_avg, f8_position, 2);
  // mju_scl(fxy_avg, fxy_avg, 0.125, 2);

  // // ----- Minimize the distance of the CoM to the average feet position ----- //
  // // Calculate the distance between COM and feet and add to residual
  // mju_subFrom(fxy_avg, capture_point, 2);           // fxy_avg = fxy_avg - capture_point
  // double com_feet_distance = mju_norm(fxy_avg, 2);  
  // residual[counter++] = com_feet_distance;

  // // ----- CoM Velocity ----- //
  // // ----- Minimize the velocity of the CoM in the xy plane ----- //
  // mju_copy(&residual[counter], com_velocity, 2);
  // counter += 2;

    // // ----- Feet Level ----- //
  // // ----- Minimize the angles the feet make with the ground ----- //
  // residual[counter++] = abs(f1_position[2] - f3_position[2]) + abs(f2_position[2] - f4_position[2]) // left dpf
  //                       + abs(f5_position[2] - f7_position[2]) + abs(f6_position[2] - f8_position[2]) // right dpf
  //                       + abs(f1_position[2] - f2_position[2]) + abs(f3_position[2] - f4_position[2]) // left ie
  //                       + abs(f5_position[2] - f6_position[2]) + abs(f7_position[2] - f8_position[2]); // right ie


  // // ----- Centroidal Momentum ----- //
  // // ----- Minimize the momentum of the CoM ----- //
  // mju_copy(&residual[counter], torso_angular_momentum, 3);
  // mju_addTo(&residual[counter], leftleg_angular_momentum, 3);
  // mju_addTo(&residual[counter], rightleg_angular_momentum, 3);
  // counter += 3;
