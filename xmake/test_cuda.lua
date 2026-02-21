-- CUDA kernel GTest target
-- Produces a native executable linked with cuda-gdb-friendly flags.
-- Usage:
--   xmake f -m debug --nv-gpu=y
--   xmake build test-cuda
--   cuda-gdb ./build/linux/x86_64/debug/test-cuda
--
-- Set breakpoints in device code normally:
--   (cuda-gdb) break argmax_single_block
--   (cuda-gdb) run

local vcpkg_root = os.getenv("VCPKG_ROOT")
                    or (is_plat("windows") and "C:/opt/vcpkg" or "~/opt/vcpkg")
local triplet    = is_plat("windows") and "x64-windows" or "x64-linux"

target("test-cuda")
    set_kind("binary")
    set_languages("cxx17")
    set_default(false)  -- not built unless explicitly requested

    -- CUDA settings – NO relocatable device code so -G works reliably
    add_rules("cuda")
    add_cugencodes("native")
    add_cuflags("-std=c++17", {force = true})

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("--compiler-options", "-fPIC", {force = true})
    end

    -- Debug-friendly flags for cuda-gdb breakpoints in device code
    if is_mode("debug") then
        add_cuflags("-G", "-g", {force = true})
        add_cxflags("-g")
    end

    -- Sources: all .cu test files under test/cuda/
    add_files("../test/cuda/*.cu")

    -- Project headers
    add_includedirs("../include")
    -- Internal src headers (for direct kernel calls)
    add_includedirs("../src")

    -- Link the CUDA op static libraries directly so device code is available
    add_deps("llaisys-ops-cuda")
    add_deps("llaisys-device-nvidia")
    add_deps("llaisys-tensor")
    add_deps("llaisys-core")
    add_deps("llaisys-device")
    add_deps("llaisys-utils")

    -- GTest
    add_includedirs(path.join(vcpkg_root, "installed", triplet, "include"))
    add_linkdirs(path.join(vcpkg_root, "installed", triplet, "lib"))
    add_links("gtest")
    add_syslinks("pthread")

    -- CUDA runtime
    add_links("cudart")

    on_install(function (target) end)
target_end()
