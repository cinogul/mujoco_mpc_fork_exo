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

#ifndef MJPC_TASKS_HUMANOID_STAND_TASK_H_
#define MJPC_TASKS_HUMANOID_STAND_TASK_H_

#include "mjpc/task.h"
#include <memory>
#include <mujoco/mujoco.h>
#include <string>

namespace mjpc {
namespace exo {

class Stand : public Task {
  public:
    class ResidualFn : public mjpc::BaseResidualFn {
      public:
        explicit ResidualFn(const Stand* task) : mjpc::BaseResidualFn(task) {}

        // Residual for task
        void Residual(const mjModel* model, const mjData* data, double* residual) const override;
    };

    Stand() : residual_(this) {}

    std::string Name() const override;
    std::string XmlPath() const override;

  protected:
    std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override { return std::make_unique<ResidualFn>(this); }
    ResidualFn* InternalResidual() override { return &residual_; }

  private:
    ResidualFn residual_;
};

} // namespace exo
} // namespace mjpc

#endif // MJPC_TASKS_HUMANOID_STAND_TASK_H_
