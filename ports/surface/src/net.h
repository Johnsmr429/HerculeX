#pragma once

#include <mlx/mlx.h>

#include <cstdint>
#include <memory>
#include <string>

namespace hx {

namespace mx = mlx::core;

enum class Rule { NNUNET, VESUVIUS };

struct NetState;

class SurfaceNet {
  public:
    SurfaceNet(const std::string &spec_path, const std::string &weights_path, double cache_gb);
    ~SurfaceNet();
    int patch() const { return patch_; }
    Rule rule() const { return rule_; }
    const std::string &name() const { return name_; }
    void normalise(const uint8_t *ct, size_t n, float *out) const;
    mx::array forward(const mx::array &x);

  private:
    int patch_ = 0;
    Rule rule_ = Rule::NNUNET;
    std::string name_;
    float lower_ = 0, upper_ = 0, mean_ = 0, std_ = 1;
    std::unique_ptr<NetState> state_;
};

}
