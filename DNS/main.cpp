#define _CRT_SECURE_NO_WARNINGS 
#define _WINSOCK_DEPRECATED_NO_WARNINGS 


#include<stdio.h>
#include<string.h>
#include <winsock2.h> 
#include <time.h>

#pragma comment(lib, "wsock32")

#define NUM 1024
#define MAX_BUF_SIZE 1024          //��������С
#define DNS_PORT 53                //DNS�˿ں�
#define MAX_ID_TRANS_TABLE_SIZE 16 //IDת��������С
#define ID_EXPIRE_TIME 10          //ID����ʱ��
#define MAX_CACHE_SIZE 5           //cache����С
#define DNS_HEAD_SIZE 12           //DNS��ͷ��С
#define AMOUNT 16                  //���IDת�����С

struct sockaddr_in local_name, out_name, client;//AF_INET��ַ
int debug_level = 0; //���Լ���
char DNS_Server_IP[16] = "10.3.9.4"; //Ĭ���ⲿdns������
SOCKET local_sock;//�����׽���
SOCKET out_sock;//�ⲿ�׽���
int total;//��ѯ�ܴ���

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

typedef struct cache_Node 
{
	int IP[4]; // IPv4 ���ĸ����ݶ�
	char dmName[100]; //URL
	struct cache_Node* nextptr;
}node;

node* headPtr = NULL;

int total_cache = 0;//cache��ǰ��С

void add_to_cache(char* url, int* ip) {
	node* ptr = headPtr;//ָ�����һ���ڵ��ָ��
	if (total_cache == MAX_CACHE_SIZE)
	{
		int i;
		for (i = 0; i < total_cache - 1; i++)
			ptr = ptr->nextptr;
		free(ptr->nextptr);//�ͷ����һ���ڵ�ռ�
		total_cache--;
		ptr->nextptr = NULL;
	}
	bool flag2 = false;
	node* temptr = NULL;
	temptr = (node*)malloc(sizeof(node));
	temptr = headPtr->nextptr;

	for (int j = 0; j < total_cache && !flag2; j++)
	{
		//printf("*---------��ʼ��ѯ��%s-------*\n", url);
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

//ת��������ʽ
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

//���ù���ʱ�䡣������Ҫ���õļ�¼ָ�������ʱ��
void set_ID_expire(IDTransform* record, int ttl)
{
	time_t now_time;
	now_time = time(NULL);
	record->expire_time = now_time + ttl;   //����ʱ��=����ʱ��+����ʱ��
}

//���record�Ƿ�ʱq
int is_ID_expired(IDTransform* record)
{
	time_t now_time;
	now_time = time(NULL);
	if (record->expire_time > 0 && now_time > record->expire_time)  //expire_time>0˵������Ч��¼
	{
		if(debug_level>=1)
		printf("client ID:%d ����ʱ\n", record->oldID);
		return 1;
	}
	return 0;
}

//������������IDת��Ϊ�µ�ID��������Ϣд��IDת������
unsigned short RegisterNewID(unsigned short oID, SOCKADDR_IN temp, BOOL ifdone)
{
	int i = 0;
	for (i = 0; i != AMOUNT; ++i)
	{
		//�ҵ��ѹ��ڻ�����������IDλ�ø���
		if (is_ID_expired(&IDTransTable[i]) == 1 || IDTransTable[i].done == TRUE)
		{
			IDTransTable[i].oldID = oID;    //������id
			IDTransTable[i].client = temp;  //������sockaddr
			IDTransTable[i].done = ifdone;  //�Ƿ����������
			set_ID_expire(&IDTransTable[i], ID_EXPIRE_TIME);
			++IDcount;
			if (debug_level >= 1)
				printf("%d id in id buffer\n", IDcount);
			break;
		}
	}
	if (i == AMOUNT)    //û�ҵ���д�ĵط�
		return 0;
	return (unsigned short)i + 1;	//�Ա����±���Ϊ�µ�ID
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

			//�����ǰʱ��
			time_t curtime;
			time(&curtime);
			printf("��ǰʱ�� = %s", ctime(&curtime));
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
			//printf("*---------��ʼ��ѯ��%s-------*\n", url);
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
		{ // ���ڴ��еı��� DNS��Ϣ����һһ�Աȣ�����ֻ�ܲ鵽 IPv4 ��ַ
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
		{ // ����û�в�ѯ��
			unsigned short* pID = (unsigned short*)malloc(sizeof(unsigned short));
			memcpy(pID, buf, sizeof(unsigned short));                 //��¼ID
			unsigned short nID = RegisterNewID(*pID, client, FALSE);   //����ID�͸÷��ͷ��ĵ�ַclient
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
				length = sendto(out_sock, buf, sizeof(buf), 0, (struct sockaddr*) & out_name, sizeof(out_name));  //���������͸��ⲿ������
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
			memcpy(sendbuf, buf, length);						//����������
			unsigned short a = htons(0x8180);//��Ӧ	QR���ݹ��ѯRD RAΪ1
			if (ipaddr[0] == 0 && ipaddr[1] == 0 && ipaddr[2] == 0 && ipaddr[3] == 0) //��ѯ��url��ip�ں�������
			{
				a = htons(0x8183);
			}
			memcpy(&sendbuf[2], &a, sizeof(unsigned short));		//�޸ı�־��

			if (ipaddr[0] == 0 && ipaddr[1] == 0 && ipaddr[2] == 0 && ipaddr[3] == 0) //��ѯ��url��ip�ں�������
			{

				if (debug_level >= 1)
					printf("%s in blacklist\n", url);
				a = htons(0x0000);	//���ι��ܣ����ش�����Ϊ0

			}//�ж��Ƿ���Ҫ���θ������Ļش�

			else

				a = htons(0x0001);	//���������ܣ����ش�����Ϊ1

			memcpy(&sendbuf[6], &a, sizeof(unsigned short));

			int curLen = 0;
			char answer[16];
			unsigned short Name = htons(0xc00c);//����ָ�루ƫ������,11000000 0000 1100,12�ֽ�����
			memcpy(answer, &Name, sizeof(unsigned short));
			curLen += sizeof(unsigned short);

			unsigned short TypeA = htons(0x0001);  //����
			memcpy(answer + curLen, &TypeA, sizeof(unsigned short));
			curLen += sizeof(unsigned short);

			unsigned short ClassA = htons(0x0001);  //��ѯ��
			memcpy(answer + curLen, &ClassA, sizeof(unsigned short));
			curLen += sizeof(unsigned short);

			unsigned long timeLive = htonl(0x0258);  //����ʱ��
			memcpy(answer + curLen, &timeLive, sizeof(unsigned long));
			curLen += sizeof(unsigned long);

			unsigned short IPLen = htons(0x0004);  //��Դ���ݳ���
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
		}//ָ�뽻���������·��ʵ��ᵽ��ǰ��
	}
}

void Receive_from_out() {
	//1�������Ĵ���buf
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

			//�����ǰʱ��
			time_t curtime;
			time(&curtime);
			printf("��ǰʱ�� = %s", ctime(&curtime));

			if (debug_level == 2)
				Output_Packet(buf, length);
		}
		unsigned short* pID = (unsigned short*)malloc(sizeof(unsigned short));//�Խ���ID����ΪDNS���ĵ�һ�������ʾ��
		memcpy(pID, buf, sizeof(unsigned short));
		int id_index = (*pID) - 1;
		free(pID);
		
		unsigned short oID = IDTransTable[id_index].oldID;//ת��Ϊ�ͻ��˷����ID
		memcpy(buf, &oID, sizeof(unsigned short));

		//��IDת�����л�ȡ����DNS�����ߵ���Ϣ
		--IDcount;
		if (debug_level >= 1)
			printf("%d id in id buffer\n", IDcount);
		IDTransTable[id_index].done = TRUE;
		SOCKADDR_IN client = IDTransTable[id_index].client;//�ӱ����ҵ�����DNS����Ŀͻ��˷�����

		int nquery = ntohs(*((unsigned short*)(buf + 4))), nresponse = ntohs(*((unsigned short*)(buf + 6)));    //����������ش����
		char* p = buf + 12; //����DNS��ͷ��ָ��
		int ip[4];

		//��ȡÿ��������Ĳ�ѯurl
		for (int i = 0; i < nquery; ++i)
		{
			Convert_to_Url(p, url);    //��ôдurl��ֻ���¼���һ�������url  ???
			while (*p > 0)  //��ȡ��ʶ��ǰ�ļ����������url
				p += (*p) + 1;
			p += 5; //����url�����Ϣ��ָ����һ������ 
		}

		if (nresponse > 0 && debug_level >= 1)
			printf("receive outside query url %s\n", url);

		//�����ظ�
		//����ο�DNS�ظ����ĸ�ʽ
		for (int i = 0; i < nresponse; ++i)
		{
			if ((unsigned char)* p == 0xc0) //��ָ�������
				p += 2;
			else
			{
				//���ݼ�������url
				while (*p > 0)
					p += (*p) + 1;
				++p;    //ָ����������
			}
			unsigned short resp_type = ntohs(*(unsigned short*)p);  //�ظ�����
			p += 2;
			unsigned short resp_class = ntohs(*(unsigned short*)p); //�ظ���
			p += 2;
			unsigned short high = ntohs(*(unsigned short*)p);   //����ʱ���λ
			p += 2;
			unsigned short low = ntohs(*(unsigned short*)p);    //����ʱ���λ
			p += 2;
			int ttl = (((int)high) << 16) | low;    //�ߵ�λ��ϳ�����ʱ��
			int datalen = ntohs(*(unsigned short*)p);   //�������ݳ���
			p += 2;
			if (debug_level >= 2)
				printf("Type -> %d,  Class -> %d,  TTL -> %d\n", resp_type, resp_class, ttl);
			if (resp_type == 1) //A����
			{
				//memset(ip, 0, sizeof(ip));
				//��ȡ4��ip����
				ip[0] = (unsigned char)* p++;
				ip[1] = (unsigned char)* p++;
				ip[2] = (unsigned char)* p++;
				ip[3] = (unsigned char)* p++;

				if (debug_level >= 1)
					printf("ip %d.%d.%d.%d\n", ip[0], ip[1], ip[2], ip[3]);

				// ������ⲿ�������н��ܵ���������Ӧ��IP
				add_to_cache(url, ip);
				break;
			}
			else p += datalen;  //ֱ������
		}

		// ת���ؿͻ���
		if (sendto(local_sock, buf, sizeof(buf), 0, (SOCKADDR*)& client, sizeof(sockaddr)) == SOCKET_ERROR)
			printf("sendto() Error \n");
		else
			if(nresponse > 0)
			printf("send local %s -> ip %d.%d.%d.%d\n", url, ip[0], ip[1], ip[2], ip[3]);
	}
}


int main(int argc, char* argv[])
{
	// ��ʼ���׽��ֶ�̬�� 
	WORD ver = MAKEWORD(2, 2);
	WSADATA dat;//WSADATA �ṹ���������溯��WSAStartup ���ص� Windows Sockets ��ʼ����Ϣ
	if (WSAStartup(ver, &dat) != 0)//��ʼ��Winsock���� 
	{
		//ʹ�� Socket �ĳ�����ʹ�� Socket ֮ǰ������� WSAStartup ����,��һ��Ӧ�ó������ WSAStartup ����ʱ��
		//����ϵͳ��������� Socket �汾��������Ӧ�� Socket �⣬
		//Ȼ����ҵ��� Socket �⵽��Ӧ�ó����С��Ժ�Ӧ�ó���Ϳ��Ե���������� Socket ���е����� Socket �����ˡ�
		printf("WSAStartup failed !\n"); // ��ʼ��ʧ��
		return 1;
	}


	//1����ȡ�����в�����ȷ�����Եȼ�������DNS��������ַ,�����
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


	//2�������ⲿ�׽��ּ������׽���
	local_sock = socket(AF_INET, SOCK_DGRAM, 0);
	out_sock = socket(AF_INET, SOCK_DGRAM, 0);

	//3����socket�ӿڸ�Ϊ������ģʽ
	int non_block = 1;
	ioctlsocket(out_sock, FIONBIO, (u_long FAR*) & non_block);
	ioctlsocket(local_sock, FIONBIO, (u_long FAR*) & non_block);

	//4�������׽���ѡ�������ֱ��ض˿ڱ�ռ�����
	int reuse = 1;
	setsockopt(local_sock, SOL_SOCKET, SO_REUSEADDR, (const char*)& reuse, sizeof(reuse));

	//5����
	local_name.sin_family = AF_INET;           //Address family AF_INET����TCP/IPЭ����
	local_name.sin_addr.s_addr = INADDR_ANY;    //�������� address
	local_name.sin_port = htons(DNS_PORT);       //�趨�˿�Ϊ53

	out_name.sin_family = AF_INET;
	out_name.sin_addr.s_addr = inet_addr(DNS_Server_IP);
	out_name.sin_port = htons(DNS_PORT);

	if (bind(local_sock, (struct sockaddr*) & local_name, sizeof(local_name)) < 0)
	{
		if (debug_level >= 1)
			printf("Bind socket port failed.\n");
		WSACleanup();//�󶨴����ͷ���Դ
		exit(1);
	}
	printf("Bind socket port successfully.\n");

	//6����ȡ����URL_IPӳ���ϵ�����
	FILE* fptr;
	if ((fptr = fopen("dnsrelay.txt", "r")) == NULL)
		return false;

	//int i;
	for (count = 0; !feof(fptr); count++)
	{
		fscanf(fptr, "%d.%d.%d.%d %s",
			&localTable[count].IP[0], &localTable[count].IP[1], &localTable[count].IP[2],
			&localTable[count].IP[3], localTable[count].dmName);
		localTable[count].frequency = 0; // ��ʼ�����ʴ���
		if (debug_level >= 1)
			printf("Read from 'dnsrelay.txt' -> [Url : %s, IP : %d.%d.%d.%d]\n", localTable[count].dmName, localTable[count].IP[0], localTable[count].IP[1], localTable[count].IP[2], localTable[count].IP[3]);
	}
	fclose(fptr);

	//7.��ʼ��IDת����
	for (int i = 0; i < AMOUNT; i++)
	{
		IDTransTable[i].oldID = 0;
		IDTransTable[i].done = TRUE;
		IDTransTable[i].expire_time = 0;
		memset(&(IDTransTable[i].client), 0, sizeof(SOCKADDR_IN));
	}

	//8��
	headPtr = (node*)malloc(sizeof(node));
	while (true)
	{
		Receive_from_out();
		Receive_from_Local();
	}
}