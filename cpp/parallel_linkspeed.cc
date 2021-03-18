#include "common.h"
#include "thd_queue.h"
#include "logger.h"
#include <cstdlib>
#include <vector>
#include <iostream>
#include <cassert>


static int n_socks = 16;
static int min_chunk = 64 * 1024;
static int max_buffer = 4 * 1024 * 1024;

void getEnvVars() {
  if (const char* env_p = std::getenv("N_SOCK")) {
    n_socks = std::stoi(env_p);
    LOG_INFO("change nsock to %d", n_socks);

    assert(n_socks % 2 == 0);
  }

  if (const char* env_p = std::getenv("MIN_CHUNK")) {
    min_chunk = std::stoi(env_p);
    LOG_INFO("change min_chunk to %d", min_chunk);
  }

  if (const char* env_p = std::getenv("MAX_BUFF")) {
    max_buffer = std::stoi(env_p);
    LOG_INFO("change max_buffer size to %d", max_buffer);
  }
}

#define SOCK_SEND 0
#define SOCK_RECV 1
struct SocketTask
{
  int op; // 0: send, 1: recv
  void* ptr;
  int size;
};

void SocketOpLoop(ThdSafeQueue<SocketTask>* queue,
                  ThdSafeQueue<int>* complete_queue,
                  bool* exit,
                  int fd) {
  SocketTask task;

  while(!(*exit)) {
    queue->pop(&task);
    if (*exit) return;

    LOG_DEBUG("op %s, fd %d, size %d", task.op == SOCK_SEND ? "send" : "recv",
              fd, task.size);

    if (task.op == 0) {
      LOG_IF_ERROR(::send(fd, task.ptr, task.size, 0) != task.size, 
        "send data error");
    } else {
      LOG_IF_ERROR(::recv(fd, task.ptr, task.size, MSG_WAITALL) != task.size, "recv data error");
    }

    complete_queue->push(1);
  }
}

int launchSendRecv(std::vector<ThdSafeQueue<SocketTask>*>& task_queues,
                    ThdSafeQueue<int>& complete_queue,
                    int size,
                    int n_sock,
                    char* send_buff,
                    char* recv_buff,
                    int mode) {

  int half_sock = n_sock / 2;
  // launch send recv tasks for local workers
  int offset = 0;
  int task_size = DIVUP(size, half_sock);
  task_size = std::max(min_chunk, task_size);

  int t = 0;
  while (offset < size) {
    int real_size = std::min(task_size, size - offset);
    SocketTask send_task, recv_task;
    send_task.op = SOCK_SEND;
    send_task.ptr = send_buff + offset;
    send_task.size = real_size;

    recv_task.op = SOCK_RECV;
    recv_task.ptr = recv_buff + offset;
    recv_task.size = real_size;

    offset += real_size;

    int qidx = t % half_sock;
    if (mode == 0) {
      // at server size: first n_send socket for send task; rest for recv
      task_queues[qidx]->push(send_task);
      task_queues[qidx + half_sock]->push(recv_task);
    } else {
      task_queues[qidx]->push(recv_task);
      task_queues[qidx + half_sock]->push(send_task);
    }

    ++t;
  }

  return t;
}


void cleanup(std::vector<ThdSafeQueue<SocketTask>*>& task_queues,
  std::vector<std::thread*>& socket_threads, char* send_buff, char* recv_buff) {
    // exit child threads
  SocketTask exit_task;
  for (int i = 0; i < n_socks; ++i) {
    task_queues[i]->push(exit_task);
    socket_threads[i]->join();
    delete task_queues[i];
    delete socket_threads[i];
  }
  free(send_buff);
  free(recv_buff);
}

void waitCompletion(int n, ThdSafeQueue<int>& cq) {
  int tmp;
  for (int i = 0; i < n; ++i) {
    cq.pop(&tmp);
  }
}

void spinWait(int us) {
  double _s = timeMs();
  while ((timeMs() - _s) * 1e3 < us)
    std::this_thread::yield();
}

struct CtrlMessage{
  int size;
  int repeat;
  int latency;
};

void serverMode(int port) {
  int listen_fd;
  LOG_IF_ERROR(createListenSocket(&listen_fd, port) == false, "create listen socket failed");
  LOG_INFO("wait client");

  int ctrl_fd = socketAccept(listen_fd, true);

  std::vector<int> socket_fds;
  std::vector<ThdSafeQueue<SocketTask>*> task_queues;
  ThdSafeQueue<int> completion_queue;

  std::vector<std::thread*> socket_threads;
  bool exit = false;

  for (int i = 0; i < n_socks; ++i) {
    int fd = socketAccept(listen_fd, true);
    socket_fds.push_back(fd);
    ThdSafeQueue<SocketTask>* q = new ThdSafeQueue<SocketTask>();
    std::thread* thd = new std::thread(SocketOpLoop, q, &completion_queue, &exit, fd);
    task_queues.push_back(q);
    socket_threads.push_back(thd);
  }

  char* send_buff = (char*)malloc(max_buffer);
  char* recv_buff = (char*)malloc(max_buffer);

  CtrlMessage c_msg;

  while (!exit) {
    std::cout << "enter 'size repeat latency(us)': ";
    std::cin >> c_msg.size;
    std::cin >> c_msg.repeat;
    std::cin >> c_msg.latency;

    double start = timeMs();
    LOG_IF_ERROR(::send(ctrl_fd, &c_msg, sizeof(c_msg), 0) != sizeof(c_msg), "send ctrl msg failed");

    if (c_msg.size < 0) {
      exit = true;
      LOG_INFO("Exiting");
      goto cleanup;
    }

    int launch_tasks = 0;
    for(int i = 0; i < c_msg.repeat; ++i){
      launch_tasks += launchSendRecv(task_queues, completion_queue, c_msg.size, n_socks, send_buff, recv_buff, 0);

      if (c_msg.latency > 0)
        spinWait(c_msg.latency);
    }

    LOG_DEBUG("launched tasks %d", launch_tasks);
    waitCompletion(launch_tasks * 2, completion_queue);
    double end = timeMs();
    LOG_INFO("size %d, time %f ms, bandwidth %f Gbps, start time %f", 
            c_msg.size * c_msg.repeat, end - start, 
            c_msg.size * c_msg.repeat * 8 / (end - start) / 1e6, start);
  }

cleanup:
  cleanup(task_queues, socket_threads, send_buff, recv_buff);
}

void clientMode(std::string& ip_str, int port) {
  int ip[4];
  ipStrToInts(ip_str, ip);
  int ctrl_fd = createSocketClient(ip, port, true);

  std::vector<int> socket_fds;
  std::vector<ThdSafeQueue<SocketTask>*> task_queues;
  std::vector<std::thread*> socket_threads;
  ThdSafeQueue<int> completion_queue;
  bool exit = false;

  for (int i = 0; i < n_socks; ++i) {
    int fd = createSocketClient(ip, port, true);
    socket_fds.push_back(fd);
    ThdSafeQueue<SocketTask>* q = new ThdSafeQueue<SocketTask>();
    std::thread* thd = new std::thread(SocketOpLoop, q, &completion_queue, &exit, fd);
    task_queues.push_back(q);
    socket_threads.push_back(thd);
  }

  char* send_buff = (char*)malloc(max_buffer);
  char* recv_buff = (char*)malloc(max_buffer);

  CtrlMessage c_msg;
  while (!exit) {
    ::recv(ctrl_fd, &c_msg, sizeof(c_msg), MSG_WAITALL);

    double start = timeMs();
    if (c_msg.size < 0) {
      exit = true;
      LOG_INFO("exiting");
      goto cleanup;
    } 

    int launched_tasks = 0;
    for (int i = 0; i < c_msg.repeat; ++i) {
      launched_tasks += launchSendRecv(task_queues, completion_queue, c_msg.size, n_socks, send_buff, recv_buff, 1);

      if (c_msg.latency > 0)
        spinWait(c_msg.latency);
    }

    LOG_DEBUG("launched tasks %d", launched_tasks);
    waitCompletion(launched_tasks * 2, completion_queue);
    double end = timeMs();
    LOG_INFO("size %d, time %f ms, bandwidth %f Gbps, receive ctrl time %f",
             c_msg.size * c_msg.repeat, end - start, c_msg.size * c_msg.repeat * 8 / (end - start) / 1e6, start);
  }

cleanup:
  cleanup(task_queues, socket_threads, send_buff, recv_buff);
}

int main(int argc, char* argv[]) {
  getEnvVars();

  if (argc < 4) {
    LOG_INFO("require input args: mode, port, peer-ip");
    return -1;
  }
  int mode = std::stoi(argv[1]);
  int port = std::stoi(argv[2]);
  std::string peer_ip = std::string(argv[3]);

  if (mode == 0) {
    serverMode(port);
  } else {
    clientMode(peer_ip, port);
  }

}