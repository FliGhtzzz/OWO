import os
import sys
import subprocess

def create_shortcut():
    # 1. Get paths
    desktop = os.path.join(os.environ["USERPROFILE"], "Desktop")
    shortcut_path = os.path.join(desktop, "File Guessr.lnk")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(project_root, "launcher_bg.py")
    vbs_path = os.path.join(project_root, "start.vbs")
    
    python_exe = sys.executable
    pythonw_exe = python_exe.replace("python.exe", "pythonw.exe")
    
    if not os.path.exists(pythonw_exe):
        pythonw_exe = python_exe

    # 1.5 Update/Create start.vbs with ABSOLUTE paths for maximum reliability
    vbs_content = f'Set WshShell = CreateObject("WScript.Shell")\n' \
                  f'WshShell.Run """{pythonw_exe}"" ""{script_path}""", 0, False\n'
    with open(vbs_path, "w", encoding="utf-8") as f:
        f.write(vbs_content)

    icon_path = os.path.join(project_root, "static", "favicon.ico")

    print(f"Project Root: {project_root}")
    print(f"Targeting Python: {pythonw_exe}")
    # 2. PowerShell command (Simplified and more standard)
    # We use escaped double quotes for the arguments part specifically
    ps_command = f"""
    $WshShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
    $Shortcut.TargetPath = "{vbs_path}"
    $Shortcut.WorkingDirectory = "{project_root}"
    $Shortcut.Description = "File Guessr - Natural Language Search"
    if (Test-Path "{icon_path}") {{
        $Shortcut.IconLocation = "{icon_path}"
    }}
    $Shortcut.Save()
    """
    
    try:
        # Run PS with -ExecutionPolicy Bypass to be safe
        subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-Command", ps_command], check=True)
        print(f"\n✅ Shortcut UPDATED on your Desktop!")
        print(f"Directory: {project_root}")
        print(f"Target: {pythonw_exe}")
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    create_shortcut()
    input("\nPress Enter to exit...")
