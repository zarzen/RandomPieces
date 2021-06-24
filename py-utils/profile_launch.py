import subprocess
import os
import sys
import time


def main():
    """"""
    current_env = os.environ.copy()
    nGPUs = 8
    current_env['MASTER_ADDR'] = '127.0.0.1'
    current_env['MASTER_PORT'] = '7890'
    current_env['WORLD_SIZE'] = str(nGPUs)

    profile = True

    cmd = [
        sys.executable,
        './profile_async_allgather.py'
    ]
    
    procs = []
    for i in range(nGPUs):
        current_env['RANK'] = str(i)

        if profile and i == 0:
            profile_cmd = ['nsys', 'profile'] + cmd
            p = subprocess.Popen(profile_cmd, env=current_env)
            procs.append(p)
        else:
            p = subprocess.Popen(cmd, env=current_env)
            procs.append(p)
    alive_procs = set(procs)
    while len(alive_procs):
        finished = []
        
        for p in alive_procs:
            if p.poll() is None:
                pass 
            else:
                finished.append(p)
        alive_procs = set(alive_procs) - set(finished)

        time.sleep(1)


if __name__ == "__main__":
    main()