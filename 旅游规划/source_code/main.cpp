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
int flag=0;//���·���ź� 
int fflag=1;//����ʱ���ź� 
int st=0;
struct route line1[10];
struct route line2[10];
struct route temp;
int ll=0,middle=0;
int ffflag=0;//һ���ƻ��ѽ��� 
time_t start,tfinish;
int differtime;
int fffflag=0;//ͬ���� 
int state=0;//�ÿ͵�״̬
int ptime=1; 
int main(void) 
{
	Init ("input.txt");			//���뺽����Ϣ 
	HANDLE hThread1;
    hThread1 =(HANDLE)_beginthreadex(NULL,0,fnInput,NULL,0,NULL);
    while(over==0)
    	fnTimeTick();
    CloseHandle(hThread1);
	//getch();
	return 0;
}
