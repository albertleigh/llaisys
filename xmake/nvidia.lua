
-- Require CUDA toolchain
add_rules("cuda")

target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")

    -- CUDA settings
    add_rules("cuda")
    add_cugencodes("native")
    add_cuflags("-std=c++17", "--relocatable-device-code=true", {force = true})

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("--compiler-options", "-fPIC", {force = true})
    end

    add_files("../src/device/nvidia/*.cu")

    -- Debug-friendly flags for CUDA
    if is_mode("debug") then
        add_cuflags("-G", "-lineinfo", "-g", {force = true})
        add_cxflags("-g")
    end

    on_install(function (target) end)
target_end()

target("llaisys-ops-cuda")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")

    -- CUDA settings
    add_rules("cuda")
    add_cugencodes("native")
    add_cuflags("-std=c++17", "--relocatable-device-code=true", {force = true})

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("--compiler-options", "-fPIC", {force = true})
    end

    add_files("../src/cuda_utils/*.cu")
    add_files("../src/ops/*/cuda/*.cu")

    -- Debug-friendly flags for CUDA
    if is_mode("debug") then
        add_cuflags("-G", "-lineinfo", "-g", {force = true})
        add_cxflags("-g")
    end

    on_install(function (target) end)
target_end()

