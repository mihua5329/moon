#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#include<cstring>
#include <time.h>
#include<iostream>
#include <WinSock2.h>

#define NUM 1024
#define MAX_BUF_SIZE 1024          /* Max buffer size */
#define DNS_PORT 53                /* DNS port */
#define MAX_ID_TRANS_TABLE_SIZE 16 /* MAX size of transfer table */
#define ID_EXPIRE_TIME 10          /* Expired time is 10s*/
#define MAX_CACHE_SIZE 5           /* Max size of cache */
#define DNS_HEAD_SIZE 12
#define AMOUNT 16//最大ID转换表大小

struct sockaddr_in local_name, out_name,client;//AF_INET地址
int debug_level = 0; /* Debug level */
char DNS_Server_IP[16] = "10.3.9.4"; /* Extern DNS server IP default value */
SOCKET local_sock;
SOCKET out_sock;

typedef struct Item { //存储一个URL_IP映射关系
	int IP[4]; // IPv4 的四个数据段
	char dmName[100]; //URL
	int frequency; // 访问频率
} item;

item localTable[NUM];

typedef struct IDChange
{
	unsigned short oldID;			//原有ID
	BOOL done;						//标记是否完成解析
	SOCKADDR_IN client;				//请求者套接字地址
	int expire_time;                //过期时间
} IDTransform;

IDTransform IDTransTable[AMOUNT];	//ID转换表
int IDcount = 0;					//转换表中的条目个数

int count = 0;//节点数
