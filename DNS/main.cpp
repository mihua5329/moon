#define _CRT_SECURE_NO_WARNINGS 
#define _WINSOCK_DEPRECATED_NO_WARNINGS 


#include<stdio.h>
#include<string.h>
#include <winsock2.h> 
#include <time.h>

#pragma comment(lib, "wsock32")

#define NUM 1024
#define MAX_BUF_SIZE 1024          //缓冲区大小
#define DNS_PORT 53                //DNS端口号
#define MAX_ID_TRANS_TABLE_SIZE 16 //ID转换表最大大小
#define ID_EXPIRE_TIME 10          //ID过期时间
#define MAX_CACHE_SIZE 5           //cache最大大小
#define DNS_HEAD_SIZE 12           //DNS报头大小
#define AMOUNT 16                  //最大ID转换表大小

struct sockaddr_in local_name, out_name, client;//AF_INET地址
int debug_level = 0; //调试级别
char DNS_Server_IP[16] = "10.3.9.4"; //默认外部dns服务器
SOCKET local_sock;//本地套接字
SOCKET out_sock;//外部套接字
int total;//查询总次数

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

typedef struct cache_Node 
{
	int IP[4]; // IPv4 的四个数据段
	char dmName[100]; //URL
	struct cache_Node* nextptr;
}node;

node* headPtr = NULL;

int total_cache = 0;//cache当前大小

void add_to_cache(char* url, int* ip) {
	node* ptr = headPtr;//指向最后一个节点的指针
	if (total_cache == MAX_CACHE_SIZE)
	{
		int i;
		for (i = 0; i < total_cache - 1; i++)
			ptr = ptr->nextptr;
		free(ptr->nextptr);//释放最后一个节点空间
		total_cache--;
		ptr->nextptr = NULL;
	}
	bool flag2 = false;
	node* temptr = NULL;
	temptr = (node*)malloc(sizeof(node));
	temptr = headPtr->nextptr;

	for (int j = 0; j < total_cache && !flag2; j++)
	{
		//printf("*---------开始查询：%s-------*\n", url);
		if (strcmp(temptr->dmName, url) == 0)
		{
			flag2 = true;
			temptr->IP[0] = ip[0];
			temptr->IP[1] = ip[1];
			temptr->IP[2] = ip[2];
			temptr->IP[3] = ip[3];
		}
		else
		{
			temptr = temptr->nextptr; 
		}
	}
	if (flag2 == false)
	{
		node* temp = (node*)malloc(sizeof(node));
		strcpy(temp->dmName, url);

		temp->IP[0] = ip[0];
		temp->IP[1] = ip[1];
		temp->IP[2] = ip[2];
		temp->IP[3] = ip[3];
		temp->nextptr = headPtr->nextptr;
		headPtr->nextptr = temp;
		total_cache++;
		printf("cache length ++");
	}
}


void Output_Packet(char* buf, int length)
{
	unsigned char unit;
	printf("Packet length = %d\n", length);
	printf("Details of the package:\n");
	for (int i = 0; i < length; i++)
	{
		unit = (unsigned char)buf[i];
		printf("%02x ", unit);
	}
	printf("\n");
}

//转换域名格式
void Convert_to_Url(char* buf, char* dest)
{
	int i = 0, j = 0, k = 0, len = strlen(buf);
	while (i < len)
	{
		if (buf[i] > 0 && buf[i] <= 63) /* Count */
		{
			for (j = buf[i], i++; j > 0; j--, i++, k++) /* Copy the url */
				dest[k] = buf[i];
		}
		if (buf[i] != 0) /* If this is not the end, put a dot into dest */
		{
			dest[k] = '.';
			k++;
		}
	}
	dest[k] = '\0'; /* Set the end */
}

//设置过期时间。参数是要设置的记录指针和生存时间
void set_ID_expire(IDTransform* record, int ttl)
{
	time_t now_time;
	now_time = time(NULL);
	record->expire_time = now_time + ttl;   //过期时间=现在时间+生存时间
}

//检查record是否超时q
int is_ID_expired(IDTransform* record)
{
	time_t now_time;
	now_time = time(NULL);
	if (record->expire_time > 0 && now_time > record->expire_time)  //expire_time>0说明是有效记录
	{
		if(debug_level>=1)
		printf("client ID:%d 请求超时\n", record->oldID);
		return 1;
	}
	return 0;
}

//函数：将请求ID转换为新的ID，并将信息写入ID转换表中
unsigned short RegisterNewID(unsigned short oID, SOCKADDR_IN temp, BOOL ifdone)
{
	int i = 0;
	for (i = 0; i != AMOUNT; ++i)
	{
		//找到已过期或已完成请求的ID位置覆盖
		if (is_ID_expired(&IDTransTable[i]) == 1 || IDTransTable[i].done == TRUE)
		{
			IDTransTable[i].oldID = oID;    //本来的id
			IDTransTable[i].client = temp;  //本来的sockaddr
			IDTransTable[i].done = ifdone;  //是否完成了请求
			set_ID_expire(&IDTransTable[i], ID_EXPIRE_TIME);
			++IDcount;
			if (debug_level >= 1)
				printf("%d id in id buffer\n", IDcount);
			break;
		}
	}
	if (i == AMOUNT)    //没找到可写的地方
		return 0;
	return (unsigned short)i + 1;	//以表中下标作为新的ID
}

void Receive_from_Local()
{
	char buf[MAX_BUF_SIZE], url[65];
	memset(buf, 0, MAX_BUF_SIZE);
	int length = -1;
	int len = sizeof(sockaddr);
	length = recvfrom(local_sock, buf, sizeof(buf), 0, (struct sockaddr*) & client, &len);
	if (length > -1)
	{
		char ori_url[65];
		memcpy(ori_url, &(buf[DNS_HEAD_SIZE]), length);
		Convert_to_Url(ori_url, url); /* Convert original url to normal url */
		if (debug_level)
		{
			printf("\n\n---- Recv : Client [IP:%s]----\n", inet_ntoa(client.sin_addr));

			//输出当前时间
			time_t curtime;
			time(&curtime);
			printf("当前时间 = %s", ctime(&curtime));
			printf("Receive from client [Query : %s]\n", url);
		}
		//netType = buf[i + 1] * 256 + buf[i + 2];
		//netClass = buf[i + 3] * 256 + buf[i + 4];
		int i;
		int ipaddr[4];
		bool flag1 = false;
		bool flag2 = false;
		node* temptr = NULL;
		temptr = (node*)malloc(sizeof(node));
		temptr = headPtr->nextptr;
		node* frontptr = NULL;
		frontptr = (node*)malloc(sizeof(node));
		frontptr = headPtr;

		node* temptr1 = NULL;
		temptr1 = (node*)malloc(sizeof(node));
		temptr1 = headPtr->nextptr;

		for (int k = 0; k < total_cache && temptr1 != NULL; k++)
		{
			printf("%s<=>%d.%d.%d.%d\n", temptr1->dmName, temptr1->IP[0], temptr1->IP[1], temptr1->IP[2], temptr1->IP[3]);
			temptr1 = temptr1->nextptr;
		}
		//free(temptr1);
		for (int j = 0; j < total_cache && !flag2; j++)
		{
			//printf("*---------开始查询：%s-------*\n", url);
			if (strcmp(temptr->dmName, url) == 0)
			{
				flag2 = true;
				ipaddr[0] = temptr->IP[0];
				ipaddr[1] = temptr->IP[1];
				ipaddr[2] = temptr->IP[2];
				ipaddr[3] = temptr->IP[3];
			}
			else
			{
				temptr = temptr->nextptr; frontptr = frontptr->nextptr;
			}
		}
		for (i = 0; i < count && !flag1 && !flag2; i++)
		{ // 与内存中的本地 DNS信息进行一一对比，本地只能查到 IPv4 地址
			if (strcmp(localTable[i].dmName, url) == 0)
			{
				flag1 = true;
				localTable[i].frequency++;
				total++;
				ipaddr[0] = localTable[i].IP[0];
				ipaddr[1] = localTable[i].IP[1];
				ipaddr[2] = localTable[i].IP[2];
				ipaddr[3] = localTable[i].IP[3];
			}
		}


		if (flag1 == false && flag2 == false)
		{ // 本地没有查询到
			unsigned short* pID = (unsigned short*)malloc(sizeof(unsigned short));
			memcpy(pID, buf, sizeof(unsigned short));                 //记录ID
			unsigned short nID = RegisterNewID(*pID, client, FALSE);   //储存ID和该发送方的地址client
			if (nID == 0)
			{
				if (debug_level >= 1)
					puts("Buffer full.");
			}
			else
			{
				if (debug_level >= 1)
					printf("send outside %s\n", url);
				memcpy(buf, &nID, sizeof(unsigned short));
				length = sendto(out_sock, buf, sizeof(buf), 0, (struct sockaddr*) & out_name, sizeof(out_name));  //将该请求发送给外部服务器
			}
			
			free(pID);
		}
		else
		{
			if (debug_level >= 1)
			{
				if (flag1 == true)
					printf("local data: %s -> %d.%d.%d.%d\n", url, ipaddr[0], ipaddr[1], ipaddr[2], ipaddr[3]);
				else
					printf("cache data: %s -> %d.%d.%d.%d\n", url, ipaddr[0], ipaddr[1], ipaddr[2], ipaddr[3]);
			}

			char sendbuf[MAX_BUF_SIZE];
			memcpy(sendbuf, buf, length);						//拷贝请求报文
			unsigned short a = htons(0x8180);//响应	QR，递归查询RD RA为1
			if (ipaddr[0] == 0 && ipaddr[1] == 0 && ipaddr[2] == 0 && ipaddr[3] == 0) //查询的url或ip在黑名单里
			{
				a = htons(0x8183);
			}
			memcpy(&sendbuf[2], &a, sizeof(unsigned short));		//修改标志域

			if (ipaddr[0] == 0 && ipaddr[1] == 0 && ipaddr[2] == 0 && ipaddr[3] == 0) //查询的url或ip在黑名单里
			{

				if (debug_level >= 1)
					printf("%s in blacklist\n", url);
				a = htons(0x0000);	//屏蔽功能：将回答数置为0

			}//判断是否需要屏蔽该域名的回答

			else

				a = htons(0x0001);	//服务器功能：将回答数置为1

			memcpy(&sendbuf[6], &a, sizeof(unsigned short));

			int curLen = 0;
			char answer[16];
			unsigned short Name = htons(0xc00c);//域名指针（偏移量）,11000000 0000 1100,12字节有用
			memcpy(answer, &Name, sizeof(unsigned short));
			curLen += sizeof(unsigned short);

			unsigned short TypeA = htons(0x0001);  //类型
			memcpy(answer + curLen, &TypeA, sizeof(unsigned short));
			curLen += sizeof(unsigned short);

			unsigned short ClassA = htons(0x0001);  //查询类
			memcpy(answer + curLen, &ClassA, sizeof(unsigned short));
			curLen += sizeof(unsigned short);

			unsigned long timeLive = htonl(0x0258);  //生存时间
			memcpy(answer + curLen, &timeLive, sizeof(unsigned long));
			curLen += sizeof(unsigned long);

			unsigned short IPLen = htons(0x0004);  //资源数据长度
			memcpy(answer + curLen, &IPLen, sizeof(unsigned short));
			curLen += sizeof(unsigned short);
			
			unsigned char IP1 = (unsigned char)(ipaddr[0]);
			memcpy(answer + curLen, &IP1, sizeof(unsigned char));
			curLen += sizeof(unsigned char);

			unsigned char IP2 = (unsigned char)(ipaddr[1]);
			memcpy(answer + curLen, &IP2, sizeof(unsigned char));
			curLen += sizeof(unsigned char);

			unsigned char IP3 = (unsigned char)(ipaddr[2]);
			memcpy(answer + curLen, &IP3, sizeof(unsigned char));
			curLen += sizeof(unsigned char);

			unsigned char IP4 = (unsigned char)(ipaddr[3]);
			memcpy(answer + curLen, &IP4, sizeof(unsigned char));
			curLen += sizeof(unsigned char);

			curLen += length;
			memcpy(sendbuf + length, answer, sizeof(answer));
			length = sendto(local_sock, sendbuf, curLen, 0, (SOCKADDR*)& client, sizeof(client));

			if (length < 0)
				perror("recv outside len < 0");

			char* p;
			p = sendbuf + length - 4;
			if (debug_level >= 1)
				printf("send local %s -> %u.%u.%u.%u\n", url, (unsigned char)* p, (unsigned char) * (p + 1), (unsigned char) * (p + 2), (unsigned char) * (p + 3));
		}
		if (flag2 == true&& temptr !=headPtr->nextptr)
		{
			node* temptr2 = NULL;
			temptr2 = (node*)malloc(sizeof(node));
			temptr2 = temptr->nextptr;
			temptr->nextptr = headPtr->nextptr;
			headPtr->nextptr = temptr;
			frontptr->nextptr = temptr2;
		}//指针交换，将最新访问的提到最前面
	}
}

void Receive_from_out() {
	//1、将报文存入buf
	char buf[MAX_BUF_SIZE], url[65];
	memset(buf, 0, MAX_BUF_SIZE);
	int length = -1;
	int len = sizeof(sockaddr);
	length = recvfrom(out_sock, buf, sizeof(buf), 0, (struct sockaddr*) & out_name, &len);
	if (length > -1)
	{
		if (debug_level)
		{
			printf("\n\n---- Recv : Extern [IP:%s]----\n", inet_ntoa(out_name.sin_addr));

			//输出当前时间
			time_t curtime;
			time(&curtime);
			printf("当前时间 = %s", ctime(&curtime));

			if (debug_level == 2)
				Output_Packet(buf, length);
		}
		unsigned short* pID = (unsigned short*)malloc(sizeof(unsigned short));//以进程ID来作为DNS报文的一个随机标示符
		memcpy(pID, buf, sizeof(unsigned short));
		int id_index = (*pID) - 1;
		free(pID);
		
		unsigned short oID = IDTransTable[id_index].oldID;//转换为客户端方向的ID
		memcpy(buf, &oID, sizeof(unsigned short));

		//从ID转换表中获取发出DNS请求者的信息
		--IDcount;
		if (debug_level >= 1)
			printf("%d id in id buffer\n", IDcount);
		IDTransTable[id_index].done = TRUE;
		SOCKADDR_IN client = IDTransTable[id_index].client;//从表中找到此条DNS请求的客户端发送者

		int nquery = ntohs(*((unsigned short*)(buf + 4))), nresponse = ntohs(*((unsigned short*)(buf + 6)));    //问题个数；回答个数
		char* p = buf + 12; //跳过DNS包头的指针
		int ip[4];

		//读取每个问题里的查询url
		for (int i = 0; i < nquery; ++i)
		{
			Convert_to_Url(p, url);    //这么写url里只会记录最后一个问题的url  ???
			while (*p > 0)  //读取标识符前的计数跳过这个url
				p += (*p) + 1;
			p += 5; //跳过url后的信息，指向下一个问题 
		}

		if (nresponse > 0 && debug_level >= 1)
			printf("receive outside query url %s\n", url);

		//分析回复
		//具体参考DNS回复报文格式
		for (int i = 0; i < nresponse; ++i)
		{
			if ((unsigned char)* p == 0xc0) //是指针就跳过
				p += 2;
			else
			{
				//根据计数跳过url
				while (*p > 0)
					p += (*p) + 1;
				++p;    //指向后面的内容
			}
			unsigned short resp_type = ntohs(*(unsigned short*)p);  //回复类型
			p += 2;
			unsigned short resp_class = ntohs(*(unsigned short*)p); //回复类
			p += 2;
			unsigned short high = ntohs(*(unsigned short*)p);   //生存时间高位
			p += 2;
			unsigned short low = ntohs(*(unsigned short*)p);    //生存时间低位
			p += 2;
			int ttl = (((int)high) << 16) | low;    //高低位组合成生存时间
			int datalen = ntohs(*(unsigned short*)p);   //后面数据长度
			p += 2;
			if (debug_level >= 2)
				printf("Type -> %d,  Class -> %d,  TTL -> %d\n", resp_type, resp_class, ttl);
			if (resp_type == 1) //A类型
			{
				//memset(ip, 0, sizeof(ip));
				//读取4个ip部分
				ip[0] = (unsigned char)* p++;
				ip[1] = (unsigned char)* p++;
				ip[2] = (unsigned char)* p++;
				ip[3] = (unsigned char)* p++;

				if (debug_level >= 1)
					printf("ip %d.%d.%d.%d\n", ip[0], ip[1], ip[2], ip[3]);

				// 缓存从外部服务器中接受到的域名对应的IP
				add_to_cache(url, ip);
				break;
			}
			else p += datalen;  //直接跳过
		}

		// 转发回客户端
		if (sendto(local_sock, buf, sizeof(buf), 0, (SOCKADDR*)& client, sizeof(sockaddr)) == SOCKET_ERROR)
			printf("sendto() Error \n");
		else
			if(nresponse > 0)
			printf("send local %s -> ip %d.%d.%d.%d\n", url, ip[0], ip[1], ip[2], ip[3]);
	}
}


int main(int argc, char* argv[])
{
	// 初始化套接字动态库 
	WORD ver = MAKEWORD(2, 2);
	WSADATA dat;//WSADATA 结构被用来保存函数WSAStartup 返回的 Windows Sockets 初始化信息
	if (WSAStartup(ver, &dat) != 0)//初始化Winsock服务 
	{
		//使用 Socket 的程序在使用 Socket 之前必须调用 WSAStartup 函数,当一个应用程序调用 WSAStartup 函数时，
		//操作系统根据请求的 Socket 版本来搜索相应的 Socket 库，
		//然后绑定找到的 Socket 库到该应用程序中。以后应用程序就可以调用所请求的 Socket 库中的其它 Socket 函数了。
		printf("WSAStartup failed !\n"); // 初始化失败
		return 1;
	}


	//1、读取命令行参数，确定调试等级，配置DNS服务器地址,并输出
	int user_set_dns_flag = 0;
	if (argc > 1 && argv[1][0] == '-')
	{
		if (argv[1][1] == 'd') debug_level++; /* Debug level add to 1 */
		if (argv[1][2] == 'd') debug_level++; /* Debug level add to 2 */
		if (argc > 2)
		{
			user_set_dns_flag = 1; /* If user set the dns server ip address */
			strcpy(DNS_Server_IP, argv[2]);
		}
	}
	if (user_set_dns_flag) /* If user set the dns server ip address */
		printf("Set DNS server : %s\n", argv[2]);
	else /* If user do not set the dns server ip address, set it by default */
		printf("Set DNS server : %s by default\n", DNS_Server_IP);
	printf("Debug level : %d\n", debug_level);


	//2、创建外部套接字及本地套接字
	local_sock = socket(AF_INET, SOCK_DGRAM, 0);
	out_sock = socket(AF_INET, SOCK_DGRAM, 0);

	//3、将socket接口改为非阻塞模式
	int non_block = 1;
	ioctlsocket(out_sock, FIONBIO, (u_long FAR*) & non_block);
	ioctlsocket(local_sock, FIONBIO, (u_long FAR*) & non_block);

	//4、设置套接字选项，避免出现本地端口被占用情况
	int reuse = 1;
	setsockopt(local_sock, SOL_SOCKET, SO_REUSEADDR, (const char*)& reuse, sizeof(reuse));

	//5、绑定
	local_name.sin_family = AF_INET;           //Address family AF_INET代表TCP/IP协议族
	local_name.sin_addr.s_addr = INADDR_ANY;    //本地任意 address
	local_name.sin_port = htons(DNS_PORT);       //设定端口为53

	out_name.sin_family = AF_INET;
	out_name.sin_addr.s_addr = inet_addr(DNS_Server_IP);
	out_name.sin_port = htons(DNS_PORT);

	if (bind(local_sock, (struct sockaddr*) & local_name, sizeof(local_name)) < 0)
	{
		if (debug_level >= 1)
			printf("Bind socket port failed.\n");
		WSACleanup();//绑定错误，释放资源
		exit(1);
	}
	printf("Bind socket port successfully.\n");

	//6、读取本地URL_IP映射关系表并输出
	FILE* fptr;
	if ((fptr = fopen("dnsrelay.txt", "r")) == NULL)
		return false;

	//int i;
	for (count = 0; !feof(fptr); count++)
	{
		fscanf(fptr, "%d.%d.%d.%d %s",
			&localTable[count].IP[0], &localTable[count].IP[1], &localTable[count].IP[2],
			&localTable[count].IP[3], localTable[count].dmName);
		localTable[count].frequency = 0; // 初始化访问次数
		if (debug_level >= 1)
			printf("Read from 'dnsrelay.txt' -> [Url : %s, IP : %d.%d.%d.%d]\n", localTable[count].dmName, localTable[count].IP[0], localTable[count].IP[1], localTable[count].IP[2], localTable[count].IP[3]);
	}
	fclose(fptr);

	//7.初始化ID转换表
	for (int i = 0; i < AMOUNT; i++)
	{
		IDTransTable[i].oldID = 0;
		IDTransTable[i].done = TRUE;
		IDTransTable[i].expire_time = 0;
		memset(&(IDTransTable[i].client), 0, sizeof(SOCKADDR_IN));
	}

	//8、
	headPtr = (node*)malloc(sizeof(node));
	while (true)
	{
		Receive_from_out();
		Receive_from_Local();
	}
}