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
#define AMOUNT 16//���IDת�����С

struct sockaddr_in local_name, out_name,client;//AF_INET��ַ
int debug_level = 0; /* Debug level */
char DNS_Server_IP[16] = "10.3.9.4"; /* Extern DNS server IP default value */
SOCKET local_sock;
SOCKET out_sock;

typedef struct Item { //�洢һ��URL_IPӳ���ϵ
	int IP[4]; // IPv4 ���ĸ����ݶ�
	char dmName[100]; //URL
	int frequency; // ����Ƶ��
} item;

item localTable[NUM];

typedef struct IDChange
{
	unsigned short oldID;			//ԭ��ID
	BOOL done;						//����Ƿ���ɽ���
	SOCKADDR_IN client;				//�������׽��ֵ�ַ
	int expire_time;                //����ʱ��
} IDTransform;

IDTransform IDTransTable[AMOUNT];	//IDת����
int IDcount = 0;					//ת�����е���Ŀ����

int count = 0;//�ڵ���
