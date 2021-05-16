#pragma once

struct passenger
{
	int start;
	int finish;
	char tool;
};  //乘客信息

struct order
{
	//char start[20];
	int finish;
	int  strategy;//0为风险最小，1为时间限制下风险最小 
	int TimeLimit;
	struct order *nextorder;
}; 
 

// 时刻表的结构体

typedef struct _EdgeData

{

   //char finish[30];
	char tool;
	int DeTime;
	int ArTime;

}EData; 

//地图矩阵
struct city
{

    char vexs[10][30];       // 顶点集合

    int vexnum;           // 顶点数

    int edgnum[10][10];           // 边数

    EData matrix[10][10][6]; // 邻接矩阵
    float danger[10];//风险值 

};



struct route
{
	int start;//起点 
    int finish;//终点 
	char tool;//交通工具
	int Detime;//出发时间 
	int Artime;//到达时间，不知道有没有用	
};
