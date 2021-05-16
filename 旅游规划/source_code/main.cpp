#include "travel.h"
#include <thread>
#include "DataStructure.h"
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h> 
#include<process.h>
#include<windows.h>

struct passenger passer;
struct city graph;
int currenttime;
struct order *command=NULL,*end=NULL;
int over=0;
int flag=0;//设计路线信号 
int fflag=1;//启动时间信号 
int st=0;
struct route line1[10];
struct route line2[10];
struct route temp;
int ll=0,middle=0;
int ffflag=0;//一个计划已结束 
time_t start,tfinish;
int differtime;
int fffflag=0;//同步锁 
int state=0;//旅客的状态
int ptime=1; 
int main(void) 
{
	Init ("input.txt");			//读入航班信息 
	HANDLE hThread1;
    hThread1 =(HANDLE)_beginthreadex(NULL,0,fnInput,NULL,0,NULL);
    while(over==0)
    	fnTimeTick();
    CloseHandle(hThread1);
	//getch();
	return 0;
}
