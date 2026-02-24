//
// Created by ali on 2/22/26.
//
// GPU-aware profiling utilities for timing kernel and op execution.
//
// Usage:
//   #include "utils/profiler.hpp"
//
//   llaisys::Profiler prof(device_type, &runtime);
//   // ... do work ...
//   prof.lap("Embedding");        // prints elapsed since last lap (GPU-synced)
//   // ... do work ...
//   prof.lap("QKV Linear");
//   prof.report();                // prints total elapsed
//
// The profiler can be enabled/disabled at construction time so you can
// leave instrumentation in place without runtime cost:
//
//   llaisys::Profiler prof(device_type, &runtime, /* enabled = */ pos == 0);
//
#pragma once

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "llaisys.h"

namespace llaisys {

// Forward-declare to avoid pulling the full runtime header.
namespace core { class Runtime; }

class Profiler {
public:
    /// @param device_type  LLAISYS_DEVICE_CPU or LLAISYS_DEVICE_NVIDIA
    /// @param rt           Pointer to the active runtime (used for GPU sync). May be nullptr for CPU.
    /// @param enabled      If false, all methods are no-ops (zero overhead).
    /// @param label        Optional prefix printed before each line.
    Profiler(llaisysDeviceType_t device_type,
             core::Runtime *rt,
             bool enabled = true,
             const char *label = "[PROFILE]");

    /// Record a lap.  Synchronizes the GPU (if applicable), prints a line with
    /// the time elapsed since the previous lap (or construction), and saves the
    /// entry for the final report.
    void lap(const char *name);

    /// Print a summary line with total elapsed time.
    void report() const;

    /// Return milliseconds since construction (CPU-only, no GPU sync).
    double total_ms() const;

    bool enabled() const { return _enabled; }

private:
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;

    struct Entry {
        std::string name;
        double ms;
    };

    void gpu_sync();

    llaisysDeviceType_t _device_type;
    core::Runtime *_rt;
    bool _enabled;
    const char *_label;

    time_point _t_start;
    time_point _t_prev;
    std::vector<Entry> _entries;
};

} // namespace llaisys
