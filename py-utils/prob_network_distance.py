"""
Using fabtests to benchmark the bandwidth between different 
EC2 instances
"""

import argparse
from pssh.clients.native.single import SSHClient
import threading
import os
import time


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--nodes", help="a list of ips",
                        type=str, required=True)
    parser.add_argument(
        "--output-dir", help="directory folder to save bandwidth logs", default="./logs")
    parser.add_argument(
        "--provider", help="network provider e.g., socket/efa", default="sockets", type=str)
    parser.add_argument(
        "--fabtests-path", help="the location of fabtests", default="/opt/fabtests")
    parser.add_argument("--libfabric-path", default="/opt/amazon/efa",
                        help="path of libfabric, used for building fabtests")
    parser.add_argument(
        '--bin', help="the fabtest binary to use", default="fi_rdm_tagged_bw")

    return parser.parse_args()


def print_output(output):
    for line in output.stdout:
        print(f"{output.host} :: {line}")
    for line in output.stderr:
        print(f"{output.host} :: {line}")


def build_fabtests(args, client):
    """"""
    build_cmds = [
        "git clone --recursive https://github.com/ofiwg/libfabric.git /tmp/libfabric",
        "cd /tmp/libfabric/fabtests",
        "git checkout v1.11.1",
        "./autogen.sh --with-libfabric={}".format(args.libfabric_path),
        "./configure --with-libfabric={} --prefix={}".format(
            args.libfabric_path, args.fabtests_path),
        "make -j 4",
        "sudo make install"
    ]

    cmd = " && ".join(build_cmds)
    print('Build fabtests with command: %s' % (cmd, ))
    output = client.run_command(cmd, use_pty=True, shell='bash -c')

    print_output(output)
    print(f"{output.host} :: exit {output.exit_code}")


def check_fabtests(args, client):
    """
    if the args.fabtests_path is not exist
    then build and install the fabtests

    assume the libfabric is available, which is installed by Deep Learning AMI
    """
    cmd = f"ls {args.fabtests_path}"
    output = client.run_command(cmd, use_pty=True, shell='bash -c')
    print_output(output)

    require_build_fabtests = False
    if int(output.exit_code) != 0:
        require_build_fabtests = True

    if require_build_fabtests:
        build_fabtests(args, client)

def save_output(output, filename):
    """"""
    file_folder = os.path.dirname(filename)
    if not os.path.exists(file_folder):
        os.makedirs(file_folder)

    with open(filename, 'w') as out_file:
        for line in output.stdout:
            out_file.write(f"{line}\n")

        for line in output.stderr:
            out_file.write(f"{line}\n")

def cleanup_old_proc(cli, bin_path):
    def proc_check(cli, bin_path):
        status_check_cmd = f"ps aux | grep {bin_path}"
        output = cli.run_command(status_check_cmd, use_pty=True, shell="bash -c")
        cnt_output = 0
        for line in output.stdout:
            if f"grep {bin_path}" in line:
                pass
            else:
                cnt_output += 1
            print(f"{cli.host} :: old proc check {line}")

        if cnt_output >= 1:
            print(f"{cli.host} has old proc {bin_path}")
            return False
        else:
            return True

    kill_proc_cmd = f"pkill -9 -f \"{bin_path}\""
    while not proc_check(cli, bin_path):
        print(f"{cli.host} :: killing old proc with command {kill_proc_cmd} ")
        output = cli.run_command(kill_proc_cmd)
        print_output(output)
        print(f"{cli.host} exit code {output.exit_code}")
    

def benchmark_bandwidth(args, clients, node_ip1, node_ip2):
    """"""
    c1 = clients[node_ip1] # server
    c2 = clients[node_ip2] # client

    bin_path = os.path.join(args.fabtests_path, "bin", args.bin)

    if args.provider == "efa":
        server_cmd = f"{bin_path} -m -w 10 -S all -p {args.provider} -E"
        client_cmd = f"{bin_path} -m -w 10 -S all -p {args.provider} -E {node_ip1}"
    else:
        server_cmd = f"{bin_path} -m -w 10 -S all -p {args.provider} -s 0.0.0.0"
        client_cmd = f"{bin_path} -m -w 10 -S all -p {args.provider} {node_ip1}"

    def exec_cmd(cli, cmd, filename):
        output = cli.run_command(cmd)
        if filename:
            save_output(output, filename)
        else:
            print_output(output)

        print(f"{cli.host} :: cmd :: {cmd} exit code: {output.exit_code}")
    
    print("cleaning existing processes")
    cleanup_old_proc(c1, bin_path)
    cleanup_old_proc(c2, bin_path)

    server_thd = threading.Thread(target=exec_cmd, args=(c1, server_cmd, None))
    server_thd.start()

    # make sure server started
    time.sleep(1)

    log_file = os.path.join(args.output_dir, f"{node_ip2}_{node_ip1}.log")
    client_thd = threading.Thread(target=exec_cmd, args=(c2, client_cmd, log_file))
    client_thd.start()

    server_thd.join()
    client_thd.join()


def init_clients(args):
    nodes = args.nodes.split(',')

    clients = {}
    for ip in nodes:
        # assume the password less ssh access
        clients[ip] = SSHClient(ip)

    checking_thds = []
    for ip in clients:
        c = clients[ip]
        t = threading.Thread(target=check_fabtests, args=(args, c))
        t.start()
        checking_thds.append(t)

    for t in checking_thds:
        t.join()

    return clients


def main():
    """"""

    args = get_args()
    clients = init_clients(args)

    for ip1 in clients:
        for ip2 in clients:
            if ip1 != ip2:
                benchmark_bandwidth(args, clients, ip1, ip2)
                time.sleep(2)


if __name__ == "__main__":
    main()
