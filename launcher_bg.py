import subprocess
import webbrowser
import sys
import os
import threading
import pystray
from PIL import Image, ImageDraw
import time
import socket
import logging

# Change working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "launcher.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    force=True
)

logging.info("--- Launcher starting ---")

# Configuration
PROC_NAME = "uvicorn"
API_URL = "http://127.0.0.1:8000"

class BackgroundLauncher:
    def __init__(self):
        self.server_process = None
        self.is_running = False
        self.icon = None
        
    def create_icon_image(self):
        # Generate a sleek icon: Purple square with a white "F"
        width = 64
        height = 64
        color_bg = (108, 92, 231)  # Purple
        color_text = (255, 255, 255) # White
        
        image = Image.new('RGB', (width, height), color_bg)
        dc = ImageDraw.Draw(image)
        # Draw a simple search-icon like circle
        dc.ellipse((10, 10, 54, 54), outline=color_text, width=4)
        dc.line((45, 45, 58, 58), fill=color_text, width=6)
        return image

    def wait_for_es_http(self, timeout=90):
        """Wait for Elasticsearch HTTP endpoint to be responsive."""
        logging.info(f"Waiting for Elasticsearch HTTP to be ready at localhost:9200 (timeout={timeout}s)...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Use localhost to match run.bat behavior
                with socket.create_connection(("localhost", 9200), timeout=1):
                    logging.info("Elasticsearch HTTP port is open/responsive.")
                    return True
            except (socket.timeout, ConnectionRefusedError, socket.error):
                time.sleep(2)
        logging.warning("Elasticsearch HTTP wait timed out.")
        return False

    def ensure_es_service(self):
        """Ensure Elasticsearch service is running and ready on Windows."""
        if os.name != 'nt':
            return True

        service_name = 'elasticsearch-service-x64'
        try:
            # Check service status using cmd to avoid PowerShell overhead where possible
            result = subprocess.run(['sc', 'query', service_name], 
                                   capture_output=True, text=True)
            
            if "SERVICE_NAME" not in result.stdout:
                logging.warning(f"Elasticsearch service '{service_name}' not found.")
                return False

            if "RUNNING" not in result.stdout:
                logging.info(f"Service state: NOT RUNNING. Attempting to start {service_name}...")
                # Try starting (might require admin)
                subprocess.run(['sc', 'start', service_name], capture_output=True, text=True)
                
                # Check if it actually started or if we need elevation
                time.sleep(1)
                check = subprocess.run(['sc', 'query', service_name], capture_output=True, text=True)
                if "RUNNING" not in check.stdout:
                    logging.warning("Direct sc start failed. Attempting elevation via PowerShell...")
                    cmd = f"Start-Process cmd -ArgumentList '/c sc start {service_name}' -Verb RunAs"
                    subprocess.run(['powershell', '-Command', cmd], creationflags=subprocess.CREATE_NO_WINDOW)
                
                # Wait for RUNNING state
                logging.info("Waiting for service to reach RUNNING state...")
                for i in range(15): # Wait up to 30 seconds
                    time.sleep(2)
                    check = subprocess.run(['sc', 'query', service_name], capture_output=True, text=True)
                    if "RUNNING" in check.stdout:
                        logging.info(f"Service reached RUNNING state after {i*2+2} seconds.")
                        break
            else:
                logging.info("Elasticsearch service is already RUNNING.")
            
            # Wait for HTTP level readiness
            return self.wait_for_es_http()

        except Exception as e:
            logging.error(f"Error checking/starting ES service: {e}")
            return False

    def start_server(self):
        logging.info("Starting server...")
        if self.is_running:
            logging.info("Server already marked as running.")
            return

        # Ensure ES is up before the app tries to connect
        self.ensure_es_service()

        # Check if port 8000 is already in use
        if self.is_port_in_use(8000):
            logging.info("Port 8000 already in use, assuming server is up.")
            self.is_running = True
            return

        try:
            # Start uvicorn as a subprocess
            # Use python.exe instead of pythonw.exe for uvicorn to ensure standard behavior
            # even when launched from pythonw.exe
            base_dir = os.path.dirname(os.path.abspath(__file__))
            python_exe = os.path.join(base_dir, "venv", "Scripts", "python.exe")
            if not os.path.exists(python_exe):
                python_exe = sys.executable.replace("pythonw.exe", "python.exe")

            cmd = [python_exe, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"]
            logging.info(f"Running command: {' '.join(cmd)}")
            
            # Pipe uvicorn output to a log file for debugging
            server_log = open(os.path.join(base_dir, "server.log"), "a")
            server_log.write(f"\n--- Server Starting at {time.ctime()} ---\n")
            server_log.flush()
            
            # Hide console window on Windows
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 0 # SW_HIDE
            
            self.server_process = subprocess.Popen(
                cmd, 
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                stdout=server_log,
                stderr=server_log,
                cwd=base_dir,
                text=True
            )
            
            logging.info(f"Server process started with PID {self.server_process.pid}")
            self.is_running = True
            # Auto-open browser after a short delay
            threading.Timer(2.0, self.open_browser).start()
            
            # Start monitoring thread
            threading.Thread(target=self.monitor_process, daemon=True).start()
            
        except Exception as e:
            logging.error(f"Failed to start server: {e}", exc_info=True)

    def is_port_in_use(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('127.0.0.1', port)) == 0

    def monitor_process(self):
        while self.is_running and self.server_process:
            if self.server_process.poll() is not None:
                self.is_running = False
                self.server_process = None
                break
            time.sleep(2)

    def stop_server(self):
        if self.server_process:
            self.server_process.terminate()
            self.server_process = None
        self.is_running = False

    def open_browser(self, icon=None, item=None):
        webbrowser.open(API_URL)

    def on_exit(self, icon, item):
        logging.info("Exiting launcher...")
        self.stop_server()
        icon.stop()
        sys.exit(0)

    def run(self):
        try:
            logging.info("Initializing Tray Icon...")
            # Initial server start
            self.start_server()
            
            # Setup Tray Icon
            image = self.create_icon_image()
            menu = pystray.Menu(
                pystray.MenuItem("開啟介面", self.open_browser),
                pystray.MenuItem("重啟服務", self.restart_service),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("退出", self.on_exit)
            )
            self.icon = pystray.Icon("File Guessr", image, "File Guessr (Running)", menu)
            logging.info("Tray icon running.")
            self.icon.run()
        except Exception as e:
            logging.error(f"Runtime error in launcher: {e}", exc_info=True)
            sys.exit(1)

    def restart_service(self, icon=None, item=None):
        self.stop_server()
        time.sleep(1)
        self.start_server()

if __name__ == "__main__":
    launcher = BackgroundLauncher()
    launcher.run()
