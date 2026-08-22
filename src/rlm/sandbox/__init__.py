"""C1 sandbox: Windows Job Object + AppContainer + composite process spawn.

Pure ctypes + stdlib, no pywin32 (spec §5). See `winjob.Job` for the
per-episode resource limits / kill primitive and `winproc` for the
AppContainer profile and the composite `CreateProcessW` spawn.
"""
