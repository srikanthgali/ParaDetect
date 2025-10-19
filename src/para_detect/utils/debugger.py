import os


def setup_debugger(port: int = 5678):
    """Attach VS Code debugger when DEBUG_MODE=1."""
    if os.getenv("DEBUG_MODE") == "1":
        import debugpy

        print(f"[DEBUG] Waiting for VS Code debugger on port {port}...")
        debugpy.listen(("0.0.0.0", port))
        debugpy.wait_for_client()
        print("[DEBUG] Debugger attached!")
