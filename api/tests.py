import os

def stop_script(script_path):
    # List processes that match the script path
    processes = os.popen(f'ps aux | grep "{script_path}" | grep -v "grep"').read().strip().split('\n')
    
    if not processes or processes[0] == '':
        print(f"No process found running the script: {script_path}")
        return

    # Display the processes for diagnostic purposes
    for process in processes:
        print(f"Found process: {process}")

    # Attempt to kill each process
    for process in processes:
        pid = process.split()[1]
        exit_code = os.system(f'kill -9 {pid}')
        if exit_code == 0:
            print(f"Terminated process with PID: {pid}")
        else:
            print(f"Error occurred while trying to terminate process with PID: {pid}")

# Example usage:
stop_script("/home/ocean/Projects/MarketPlaceSecurityApp/ai/main.py")
import os

def list_processes(script_name):
    # List processes that match the script name
    processes = os.popen(f'ps aux | grep "{script_name}" | grep -v "grep"').read().strip().split('\n')
    
    if not processes or processes[0] == '':
        print(f"No process found with the name: {script_name}")
        return

    # Display the processes for manual verification
    for process in processes:
        print(f"Found process: {process}")

# Example usage:
list_processes("main.py")
