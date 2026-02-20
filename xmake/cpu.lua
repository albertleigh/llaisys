target("llaisys-device-cpu")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/device/cpu/*.cpp")

    on_install(function (target) end)
target_end()

-- Intel MKL option
option("mkl")
    set_default(true)
    set_showmenu(true)
    set_description("Use Intel MKL instead of OpenBLAS for BLAS operations")
option_end()

target("llaisys-ops-cpu")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")

    -- Add OpenMP support for parallelization
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cxflags("-fopenmp")
        add_ldflags("-fopenmp")
    else
        add_cxflags("/openmp")
    end

    if has_config("mkl") then
        -- Intel MKL (preferred when available)
        local mkl_root = os.getenv("MKLROOT") or "/opt/intel/oneapi/mkl/latest"
        add_defines("ENABLE_MKL")
        add_includedirs(path.join(mkl_root, "include"))
        add_linkdirs(path.join(mkl_root, "lib", "intel64"))
    else
        -- Fallback to OpenBLAS from vcpkg
        local vcpkg_root = os.getenv("VCPKG_ROOT") or (is_plat("windows") and "C:/opt/vcpkg" or "~/opt/vcpkg")
        local triplet = is_plat("windows") and "x64-windows" or "x64-linux"

        add_includedirs(path.join(vcpkg_root, "installed", triplet, "include"))
        add_linkdirs(path.join(vcpkg_root, "installed", triplet, "lib"))
        add_links("openblas")
    end

    add_files("../src/ops/*/cpu/*.cpp")

    on_install(function (target) end)
target_end()

