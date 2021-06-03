""" setup deepspeed environment 

"""

import argparse
from pssh.clients.native.single import SSHClient
import threading


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--nodes', help="ips")
    arg_parser.add_argument('--data-source', default="172.31.14.59")

    return arg_parser.parse_args()

def init_clients(args):
    nodes = args.nodes.split(',')

    clients = {}
    for ip in nodes:
        # assume the password less ssh access
        clients[ip] = SSHClient(ip)

    return clients

def setup(args, clients):

    def _setup_env(cli):
        """"""
        commands = [
            "eval \"$(command conda 'shell.bash' 'hook' 2> /dev/null)\"", # init conda env
            "conda create -n ds -y python=3.7", # create new env
            "conda activate ds", # 
            "pip install regex", 
            # install torch nightly, torch==1.8.1 has a bug AdamW optimizer cannot work
            "pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html",
            "git clone https://github.com/NVIDIA/apex ~/apex", 
            "cd ~/apex", 
            "export CUDA_HOME=/usr/local/cuda-11.1", # 10.2 does not support A100
            "pip install -v --disable-pip-version-check --no-cache-dir --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext\" ./",
            "pip install --no-cache-dir deepspeed",
            # "scp -r ubuntu@{}:~/data ~/data".format(args.data_source),
            # "scp -r ubuntu@{}:~/DeepSpeed ~/DeepSpeed".format(args.data_source),
        ]
        cmd = " ; ".join(commands)
        print(f"{cli.host} :: cmd {cmd}")

        cmd_output = cli.run_command(cmd, use_pty=True, shell="bash -c")
        for line in cmd_output.stdout:
            print(f"{cli.host} :: {line}")
        for line in cmd_output.stderr:
            print(f"{cli.host} :: {line}")
        print(f"{cli.host} :: exit {cmd_output.exit_code}")
    
    client_thds = []
    for ip in clients:
        t = threading.Thread(target=_setup_env, args=(clients[ip], ))
        t.start()
        client_thds.append(t)
    
    for t in client_thds:
        t.join()

def main():
    """"""
    args = get_args()
    clients = init_clients(args)

    setup(args, clients)


if __name__ == "__main__":
    main()