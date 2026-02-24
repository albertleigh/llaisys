-- xmake/pylib.lua
-- Sets up the llaisys Python package (python/)
-- Handles: venv symlink, pip install -e

target("pylib")
    set_kind("phony")
    set_default(false)  -- opt-in: `xmake build pylib`

    on_build(function (target)
        local proj = os.projectdir()
        local pkg_dir = path.join(proj, "python")

        -- ── 1. Ensure root .venv exists ─────────────────────────────
        local venv_dir = path.join(proj, ".venv")
        if not os.isdir(venv_dir) then
            print("pylib: creating Python venv ...")
            local python = is_plat("windows") and "python" or "python3"
            os.vrunv(python, {"-m", "venv", venv_dir})
        end

        -- ── 2. Symlink .venv into python/ dir ──────────────────────
        local pkg_venv = path.join(pkg_dir, ".venv")
        if not os.isdir(pkg_venv) and not os.islink(pkg_venv) then
            if is_plat("windows") then
                os.vrunv("cmd", {"/c", "mklink", "/J", pkg_venv, venv_dir})
            else
                local rel = path.relative(venv_dir, pkg_dir)
                os.vrunv("ln", {"-sfn", rel, pkg_venv})
            end
            print("pylib: linked .venv → " .. venv_dir)
        end

        -- ── 3. Resolve pip executable ───────────────────────────────
        local pip
        if is_plat("windows") then
            pip = path.join(venv_dir, "Scripts", "pip.exe")
        else
            pip = path.join(venv_dir, "bin", "pip")
        end

        -- ── 4. pip install -e python/ ───────────────────────────────
        print("pylib: installing llaisys Python package ...")
        os.vrunv(pip, {"install", "-e", pkg_dir})

        cprint("${bright green}pylib: setup complete${clear}")
    end)

    on_clean(function (target)
        local pkg_dir = path.join(os.projectdir(), "python")

        -- Remove .venv symlink
        local pkg_venv = path.join(pkg_dir, ".venv")
        if os.islink(pkg_venv) then
            os.rm(pkg_venv)
            print("pylib: removed .venv symlink")
        end

        -- Remove egg-info
        local egg = path.join(pkg_dir, "llaisys.egg-info")
        if os.isdir(egg) then
            os.rmdir(egg)
        end
    end)

    on_install(function (target) end)
target_end()
