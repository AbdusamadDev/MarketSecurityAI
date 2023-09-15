import time
import subprocess
from threading import Thread

# Define the script to be reloaded
script_to_reload = "your_script.py"

def reload_script():
    while True:
        print(f"Reloading {script_to_reload}...")
        subprocess.run(["python", script_to_reload])
        print(f"{script_to_reload} reloaded. Waiting for 1 minute...")
        time.sleep(60)  # Wait for 1 minute before reloading again

if __name__ == "__main__":
    reload_thread = Thread(target=reload_script)
    reload_thread.daemon = True  # This thread will exit when the main program exits
    reload_thread.start()

    # Keep the main program running to allow script reloading
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
