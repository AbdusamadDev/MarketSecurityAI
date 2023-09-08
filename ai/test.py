import psutil
import os


def stop_script(script_path):
    for process in psutil.process_iter():
        try:
            pinfo = process.as_dict(attrs=["pid", "name", "cmdline"])
            if script_path in pinfo["cmdline"]:
                process.terminate()
                print(f"Terminated process with PID: {pinfo['pid']}")
                return
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    print(f"No process found running the script: {script_path}")


current_directory = os.path.abspath(os.getcwd()).split("/")[0:-1]
path = ""
for i in current_directory[1:]:
    path += "/" + i
stop_script(path + "/ai/main.py")
# stop_script("main.py")