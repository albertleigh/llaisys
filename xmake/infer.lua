-- xmake/infer.lua
-- Sets up and builds the llaisys-infer Python service (services/infer)
-- Handles: venv symlink, pip install -e, and portal UI integration

target("infer-service")
    set_kind("phony")
    set_default(false)  -- opt-in: `xmake build infer-service`

    on_build(function (target)
        local proj = os.projectdir()
        local svc_dir = path.join(proj, "services", "infer")

        -- ── 1. Ensure root .venv exists ─────────────────────────────
        local venv_dir = path.join(proj, ".venv")
        if not os.isdir(venv_dir) then
            print("infer-service: creating Python venv ...")
            local python = is_plat("windows") and "python" or "python3"
            os.vrunv(python, {"-m", "venv", venv_dir})
        end

        -- ── 2. Symlink .venv into service dir ──────────────────────
        local svc_venv = path.join(svc_dir, ".venv")
        if not os.isdir(svc_venv) and not os.islink(svc_venv) then
            if is_plat("windows") then
                -- Windows: use directory junction (no admin needed)
                os.vrunv("cmd", {"/c", "mklink", "/J", svc_venv, venv_dir})
            else
                local rel = path.relative(venv_dir, svc_dir)
                os.vrunv("ln", {"-sfn", rel, svc_venv})
            end
            print("infer-service: linked .venv → " .. venv_dir)
        end

        -- ── 3. Resolve pip executable ───────────────────────────────
        local pip
        if is_plat("windows") then
            pip = path.join(venv_dir, "Scripts", "pip.exe")
        else
            pip = path.join(venv_dir, "bin", "pip")
        end

        -- ── 4. pip install -e services/infer ────────────────────────
        print("infer-service: installing Python dependencies ...")
        os.vrunv(pip, {"install", "-e", svc_dir})

        cprint("${bright green}infer-service: setup complete${clear}")
    end)

    on_clean(function (target)
        local svc_dir = path.join(os.projectdir(), "services", "infer")

        -- Remove .venv symlink
        local svc_venv = path.join(svc_dir, ".venv")
        if os.islink(svc_venv) then
            os.rm(svc_venv)
            print("infer-service: removed .venv symlink")
        end

        -- Remove egg-info
        local egg = path.join(svc_dir, "llaisys_infer.egg-info")
        if os.isdir(egg) then
            os.rmdir(egg)
        end
    end)

    on_install(function (target) end)
target_end()
