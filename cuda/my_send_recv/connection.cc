#include "connection.h"

ConnectionType_t NetConnection::getType() {
    return Net;
}

NetConnection::NetConnection(NetSendConnArgs& args) {}
NetConnection::NetConnection(NetRecvConnArgs& args) {}

void NetConnection::sendCtrl(void* buff, size_t count) {}
void NetConnection::sendData(void* buff, size_t count) {}
