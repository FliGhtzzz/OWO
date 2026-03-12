import subprocess
import os
import time

def kill_processes():
    print("Attempting to force-kill all File Guessr related processes...")
    try:
        # Kill uvicorn and python processes that might be hanging
        # We use taskkill /F to ensure they actually die
        subprocess.run(['taskkill', '/F', '/IM', 'uvicorn.exe', '/T'], capture_output=True)
        subprocess.run(['taskkill', '/F', '/IM', 'python.exe', '/T'], capture_output=True)
        print("Success: Attempted to kill uvicorn and python.")
    except Exception as e:
        print(f"Error during kill: {e}")

if __name__ == "__main__":
    kill_processes()
    print("\nProcesses should be dead now.")
    print("Please go back to the tray icon and click 'Restart Service' one more time.")
    print("Or run run.bat manually.")
