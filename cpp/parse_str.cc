#include <iostream>
#include <string>
#include <vector>
#include <sstream>

void parse_groups_str(const char* str, std::vector<std::vector<int>>& groups) {
  char buff[10] = {"\0"};
  int i = 0; // scan str
  int j = 0; // record position of buff
  
  std::vector<int> cont;
  while (1) {
    char t = str[i];
    if (t == ',' || t == ';' || t == '\0') {
      if (j != 0) {
        int rank = std::stoi(buff);
        cont.push_back(rank);
        for (int k = 0; k < j; ++k) {
          buff[k] = '\0';
        }
        j = 0;
      }
    }

    if (t == ';' || t == '\0') {
      groups.push_back(std::move(cont));
      cont.clear();
    }
    else if (t != ',') {
      buff[j] = t;
      j++;
    }

    if (t == '\0') 
      break;
    i++;
  }

  std::vector<int> level2_group;
  for (auto& g : groups) {
    level2_group.push_back(g[0]);
  }
  groups.push_back(std::move(level2_group));
}

std::string group_ranks_str(std::vector<int>& group){
  std::stringstream ss;
  for (int i = 0; i < group.size(); ++i) {
    ss << group[i];
    if (i != group.size() - 1) {
      ss << "-";
    }
  }
  return ss.str();
}


int main(int argc, char* argv[]) {
  std::string test1("1,2,3,4;5,6,7,8;9,10,11,12,13");
  std::vector<std::vector<int>> groups;

  parse_groups_str(test1.c_str(), groups);

  // print groups
  for (auto& g : groups) {
    printf("%s\n", group_ranks_str(g).c_str());
  }

  printf("level 2 group %s\n", group_ranks_str(groups[groups.size() - 1]).c_str());
}