//
// Created by ali on 2/22/26.
//

#include "profiler.hpp"
#include "../core/runtime/runtime.hpp"

namespace llaisys {

Profiler::Profiler(llaisysDeviceType_t device_type,
                   core::Runtime *rt,
                   bool enabled,
                   const char *label)
    : _device_type(device_type)
    , _rt(rt)
    , _enabled(enabled)
    , _label(label)
    , _t_start(clock::now())
    , _t_prev(_t_start) {}

void Profiler::gpu_sync() {
    if (_device_type != LLAISYS_DEVICE_CPU && _rt) {
        _rt->synchronize();
    }
}

void Profiler::lap(const char *name) {
    if (!_enabled) return;

    gpu_sync();

    auto now = clock::now();
    double ms = std::chrono::duration<double, std::milli>(now - _t_prev).count();
    _t_prev = now;

    _entries.push_back({name, ms});

    fprintf(stderr, "%s %s: %.2fms\n", _label, name, ms);
    fflush(stderr);
}

void Profiler::report() const {
    if (!_enabled) return;

    double total = 0;
    for (auto &e : _entries) total += e.ms;

    fprintf(stderr, "%s Total: %.2fms (%zu sections)\n", _label, total, _entries.size());

    // Breakdown
    for (auto &e : _entries) {
        fprintf(stderr, "%s   %-30s %8.2fms  (%5.1f%%)\n",
                _label, e.name.c_str(), e.ms, (total > 0 ? 100.0 * e.ms / total : 0));
    }
    fflush(stderr);
}

double Profiler::total_ms() const {
    auto now = clock::now();
    return std::chrono::duration<double, std::milli>(now - _t_start).count();
}

} // namespace llaisys
