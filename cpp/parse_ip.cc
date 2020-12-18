#include <string>
#include <iostream>
#include <vector>
#include <cctype>
#include <algorithm>


void parseIPs(std::string& hosts, std::vector<std::string>& IPs) {
  size_t ip_offset = hosts.find(' ');
  size_t ip_temp_idx = 0;
  while(ip_offset != std::string::npos){
    std::string ip = hosts.substr(ip_temp_idx, ip_offset - ip_temp_idx);
    ip.erase(std::remove_if(ip.begin(), ip.end(), ::isspace), ip.end());
    IPs.push_back(ip);
    ip_temp_idx = ip_offset;
    ip_offset = hosts.find(' ', ip_offset + 1);
  }
  std::string ip = hosts.substr(ip_temp_idx + 1);
  ip.erase(std::remove_if(ip.begin(), ip.end(), ::isspace), ip.end());
  IPs.push_back(ip);
}


int main() {
  std::string test_hosts = "172.31.88.106 172.31.76.23 172.31.79.244";
  std::vector<std::string> IPs;
  parseIPs(test_hosts, IPs);
  for (auto s : IPs) {
    printf("'%s'\n", s.c_str());
  }
  return 0;
}