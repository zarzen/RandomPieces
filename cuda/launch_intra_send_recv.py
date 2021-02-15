import subprocess
import threading

def launch(n_devs, dev_id, nelem):
    subprocess.run(["./build/intraSendRecv", str(n_devs), str(dev_id), str(nelem)])

# subprocess.run(["ls"])

nranks = 4
nelem = 1024 * 1024 / 4
procs = []
for i in range(nranks):
    procs.append(threading.Thread(target=launch, args=(nranks, i, nelem)))
    procs[i].start()

for p in procs:
    p.join()
