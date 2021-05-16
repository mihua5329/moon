#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h> 
#include<windows.h>
#include <process.h>
#include<graphics.h>  
#include <conio.h>    
#include"data_structure.h"
#include"const.h"
#include"put.h"
#include"buyriders.h"
#include"delete.h"
#include"orderput.h"
#include"orderdivide.h"
#include"routedesign.h"
#include"ridermove.h"
#include"output.h"
#include"bankrupt.h"
#include"mouseput.h" 
LISTNODE* allheadptr = NULL, * headptr = NULL;
struct rider riders[5];
struct point* rheadptr[5];
LISTNODE *neworder1; //待派送订单数组
LISTNODE neworder2;
int over = 0;//是否破产 
int currenttime = 0;//当前时刻 
int ridersnum = 0; 
int len=0;
int money = 1000, takeorders = 0, finish = 0, timeout = 0; int init = 1;
int chrom[sizemost][NUM];
struct position lastpos[7]; 
int finishnum[7] = { 0,0,0,0,0,0,0 };
int finenum[7]= {0,0,0,0,0,0,0 };
struct position ridersstop[7];
int flag[7] = { 0,0,0,0,0 ,0,0};
int judge[7]={0,0,0,0,0,0,0};
struct spot* lastaspot[7];
struct spot * lastbspot[7];
int fflag[7]={0};
int ffflag=0;
int both[2];
HANDLE hPenMutex;
int main()
{
	HANDLE hThread1;
    hThread1 =(HANDLE)_beginthreadex(NULL,0,mouseput,NULL,0,NULL);
	while (over==0)
	{ if(ffflag==1) {
	int sb=0;
	for(;sb<7;sb++){
		flag[sb]=0;
		judge[sb]=0;
	}
        buy_riders();
		order_Divide();
		//接到新订单的时候重新规划路线
		int a; 
			for (a = 0; a < ridersnum; a++)
			{
			if(((fflag[a]==0&&riders[a].rflag==1)||fflag[a]==1)&&(riders[a].ctake_orders!=0))
		    designroute(a);
		    fflag[a]=0;
		    if(riders[a].ctake_orders!=0)
				rider_move(a);
				if (over == 1)
					break;
				riders[a].rflag=0;
			}
		out_put();//包括图像输出 
		ffflag=0;
		len=0;
}
}	gobankrupt(); 
	CloseHandle(hThread1);
	getch();
	return 0;
}
//吕文秀、陈绍银、黄若妍 

