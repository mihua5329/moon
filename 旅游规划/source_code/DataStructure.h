#pragma once

struct passenger
{
	int start;
	int finish;
	char tool;
};  //�˿���Ϣ

struct order
{
	//char start[20];
	int finish;
	int  strategy;//0Ϊ������С��1Ϊʱ�������·�����С 
	int TimeLimit;
	struct order *nextorder;
}; 
 

// ʱ�̱�Ľṹ��

typedef struct _EdgeData

{

   //char finish[30];
	char tool;
	int DeTime;
	int ArTime;

}EData; 

//��ͼ����
struct city
{

    char vexs[10][30];       // ���㼯��

    int vexnum;           // ������

    int edgnum[10][10];           // ����

    EData matrix[10][10][6]; // �ڽӾ���
    float danger[10];//����ֵ 

};



struct route
{
	int start;//��� 
    int finish;//�յ� 
	char tool;//��ͨ����
	int Detime;//����ʱ�� 
	int Artime;//����ʱ�䣬��֪����û����	
};
