#include<stdio.h>
#include<windows.h>
#include<graphics.h>  
#include <conio.h> 
#include<time.h>
#include"data_structure.h"
#include<process.h>
#include<math.h>
extern int ffflag;//同步锁
extern HANDLE hPenMutex;//互斥对象
extern LISTNODE *neworder1; //待派送订单数组
extern LISTNODE neworder2;
extern int takeorders,currenttime,money,ridersnum,finish,timeout;//当前时间，接单数，钱数，骑手数，完成数，超时数
extern int len;//待派送订单的长度
extern struct rider riders[];//骑手数组结构
extern int both[]; 
unsigned __stdcall mouseput(void* pArguments)
{
	time_t start, tfinish;//程序的时间，用来计算时间单位
	initgraph(1000, 770); 
	mouse_msg msg={0};//鼠标信息结构体
	int x=0,y=0;//存储鼠标位置 
	PIMAGE img;//存储图片 
    img=newimage();// 调用动画函数
    getimage(img,"背景六.jpg",0,0);
   char ridernum[10],current[10],mon[10],take[350],fini[350],timeover[10],qishou[30];
 		setfont(40, 0, "宋体");
        //写文字 
        sprintf(current, "%d", currenttime);//将整形a转换成字符串
        outtextxy(730, 200, "时间：");
        outtextxy(850, 200, current);
        
        sprintf(mon, "%d", money);
        outtextxy(730, 240, "钱：");
         outtextxy(810, 240, mon);
         
         sprintf(qishou, "%d", ridersnum);
        outtextxy(730, 280, "骑手：");
         outtextxy(850, 280, qishou);
        sprintf(take, "%d", takeorders);
        outtextxy(730, 320, "接单数：");
         outtextxy(890, 320, take);
         
        sprintf(fini, "%d", finish);
        outtextxy(730, 360, "完成数：");
         outtextxy(890, 360, fini);
         
        sprintf(timeover, "%d", timeout);
        outtextxy(730, 400,	"超时数：");
        outtextxy(890, 400, timeover);
    	LISTNODE *lastptr=NULL;
        start = clock(); // 开始计时
        LISTNODE *currentt=NULL;
  	    currentt=(LISTNODE *)malloc(sizeof(LISTNODE));
    while(!kbhit())
    {
		if(ffflag==0){
    	//WaitForSingleObject(hPenMutex,INFINITE);
        putimage(0, 0, img);
		if(mousemsg())
	    { 
		msg = getmouse();
		if(msg.is_down())
		{
		 	both[0]++;
			mousepos(&x, &y);
	    	currentt->x=x/70*2;currentt->y=y/70*2;currentt->flag=0;
			if(neworder1==NULL)
			{
				neworder1=currentt;
			}
			else
			{	neworder2.x=currentt->x;
			    neworder2.flag=currentt->flag;
			}
			 	getimage(img,"餐馆.jpg",0,0);
				putimage(x/70*70, y/70*70, img);
			}
		 else if(msg.is_up())
		 {
		 	if(neworder1->flag==0)
			 {
			 	
		 	neworder1->time=currenttime;
			mousepos(&x, &y);
			neworder1->a=x/70*2;
			neworder1->b=y/70*2;
		    getimage(img,"食客.jpg",0,0);
			putimage(x/70*70, y/70*70, img);
			both[1]++;
			takeorders++;
			neworder1->number=takeorders;
			neworder1->flag=1;
			len++;
		}
			else{
				neworder2.time=currenttime;
			mousepos(&x, &y);
			neworder2.a=x/70*2;
			neworder2.b=y/70*2;
			getimage(img,"食客.jpg",0,0);
			putimage(x/70*70, y/70*70, img);
			both[1]++;
			takeorders++;
			neworder2.number=takeorders;
			neworder2.flag=1;
			len++;
			}
		 }  
     	}
     	tfinish = clock();
     	double duration = (double)(tfinish - start) ;
     	if(((int)duration-2000*((int)duration/2000))<=50)
     	{
     		ffflag=1;
     		currenttime++;
     		printf("%d ",currenttime);
		}
	}
	}
	    _endthreadex(0);	
	return 0;
}//ffflag=1的时候分配订单neworder，然后变为空，ffflag=0;
//吕文秀 
