# The vLLM engine core is a spawned process, so the FP8 emulation is installed at interpreter start
# of every process that has this directory on PYTHONPATH. Off unless the serve config sets the flag.
import os

if os.environ.get("OAL_FP8_SM90_EMULATION") == "1":
    import fp8_sm90_emu

    fp8_sm90_emu.install_import_hook()
