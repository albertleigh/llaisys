
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
    else
        -- Match MSVC dynamic CRT used by C++ targets
        if is_mode("debug") then
            add_cuflags("-Xcompiler", "/MDd", {force = true})
        else
            add_cuflags("-Xcompiler", "/MD", {force = true})
        end
    end

    add_files("../src/device/nvidia/*.cu")

    -- CUDA debug: -G enables device-code breakpoints in cuda-gdb/nsight.
    -- -G is a superset of -lineinfo so we don't add both.
    -- Kernels must use cudaOccupancyMaxPotentialBlockSize to handle the
    -- increased register pressure from -G.
    if is_mode("debug") then
        add_cuflags("-G", "-g", {force = true})
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
    else
        -- Match MSVC dynamic CRT used by C++ targets
        if is_mode("debug") then
            add_cuflags("-Xcompiler", "/MDd", {force = true})
        else
            add_cuflags("-Xcompiler", "/MD", {force = true})
        end
    end

    add_files("../src/cuda_utils/*.cu")
    add_files("../src/ops/*/cuda/*.cu")

    -- CUDA debug: -G enables device-code breakpoints in cuda-gdb/nsight.
    -- -G is a superset of -lineinfo so we don't add both.
    -- Kernels must use cudaOccupancyMaxPotentialBlockSize to handle the
    -- increased register pressure from -G.
    if is_mode("debug") then
        add_cuflags("-G", "-g", {force = true})
        add_cxflags("-g")
    end

    on_install(function (target) end)
target_end()

