import subprocess
import threading

def launch(n_devs, dev_id, nelem, peer):
    subprocess.run(["./build/intraSendRecv", str(n_devs), str(dev_id), str(nelem), str(peer)])

# subprocess.run(["ls"])

nranks = 8
nelem = 1024 * 1024 
procs = []
ring = [0,1,2,3,7,6,5,4]
assert(len(ring) == nranks)
for i in range(nranks):
    dev = ring[i]
    peer = ring[(i+1)%nranks]
    print("{}->{}".format(dev, peer))
    procs.append(threading.Thread(target=launch, args=(nranks, dev, nelem, peer)))
    procs[i].start()

for p in procs:
    p.join()
