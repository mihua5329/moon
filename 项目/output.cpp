#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h> 
#include<windows.h>
#include <process.h>
#include<graphics.h>  
#include <conio.h>
#include"data_structure.h" 
extern struct rider riders[5]; 
extern int init,ridersnum, currenttime, money, takeorders, finish, timeout;
extern int finishnum[5], finenum[5], flag[5];//依次存储的是骑手结单号，罚款号，以及骑手到的时候是完成（1)还是超时（2）还是没到（0）  
extern struct position ridersstop[7];//存储骑手所到点的坐标 
extern int judge[7];//骑手到的是餐馆（1）还是食客（0）
void out_put() 
{
	FILE *fptr;
	if ((fptr =fopen("output.txt", "a")) == NULL)
		printf("can't open file output!\n");
	else 
	{
	int p;
	fprintf(fptr, "时间:%d\n", currenttime);
	fprintf(fptr, "钱:%d\n", money);
	fprintf(fptr, "接单数:%d\n", takeorders);
	fprintf(fptr, "完成数:%d;", finish);
	fprintf(fptr, "结单：");
	for (p = 0; p <= ridersnum - 1; p++)
		if (flag[p] == 1)
			fprintf(fptr, "%d ", finishnum[p]);
	fprintf(fptr, ";\n" );
	fprintf(fptr, "超时数:%d;", timeout);
	fprintf(fptr, "罚单：");
	for (p = 0; p <= ridersnum - 1; p++)
		if (flag[p] == 2)
			fprintf(fptr, "%d ", finenum[p]);
	fprintf(fptr, ";\n");
	int num;
	for (num = 0; num <= ridersnum - 1; num++) 
	{
		fprintf(fptr, "骑手%d位置：%d,%d;", num, riders[num].x, riders[num].y);
		fprintf(fptr, "停靠：");
		if(flag[num]!=0)
			if(judge[num]==0)
			fprintf(fptr,"食客 %d %d",ridersstop[num].x,ridersstop[num].y);
			else fprintf(fptr,"餐馆 %d %d",ridersstop[num].x,ridersstop[num].y);
			fprintf(fptr,";\n");
	}
	fclose(fptr);
	}
     //将上一时间单位的背景及输出全部清屏 
    cleardevice(); 
	PIMAGE img;
	img=newimage();
	//贴背景 
    getimage(img,"背景六.jpg",0,0);
	putimage(0, 0, img);
	
    //声明数组，用来存放字符串
	char ridernum[10],current[10],mon[10],take[350],fini[350],timeover[10],qishou[30];
 		//指定字体高度宽度 
		setfont(40, 0, "宋体");
        sprintf(current, "%d", currenttime);//将整形currenttime转换成字符串
        outtextxy(730, 200, "时间：");
        outtextxy(850, 200, current);//将字符串输出到界面上 
        
        sprintf(mon, "%d", money);//将整形money转换成字符串
        outtextxy(730, 240, "钱：");
        outtextxy(810, 240, mon);
         
         sprintf(qishou, "%d", ridersnum);//将整形money转换成字符串
        outtextxy(730, 280, "骑手：");
         outtextxy(850, 280, qishou);
         
        sprintf(take, "%d", takeorders);//将整形takeorders转换成字符串
        outtextxy(730, 320, "接单数：");
         outtextxy(890, 320, take);
         
        sprintf(fini, "%d", finish);//将整形finish转换成字符串
        outtextxy(730, 360, "完成数：");
         outtextxy(890, 360, fini);
         
        sprintf(timeover, "%d", timeout);//将整形timeout转换成字符串
        outtextxy(730, 400,	"超时数：");
        outtextxy(890, 400, timeover);
        
       for(int i=0;i<ridersnum;i++)
	   {
		//贴骑手 
		if(riders[i].x%2!=0&&riders[i].y%2==0)
		{
			//如果骑手横坐标为奇数，纵坐标为偶数 
			getimage(img,"骑手.jpg",0,0);
			putimage(((riders[i].x+1)/2)*40+(riders[i].x/2)*30,(riders[i].y/2)*70+5,img);
		}
	    else
	    {
		   //在其他位置 
	    	getimage(img,"骑手.jpg",0,0);
	    	putimage((riders[i].x/2)*70+5,((riders[i].y+1)/2)*40+(riders[i].y/2)*30,img);
	    }
	    //判断骑手是否到达餐馆或食客
		if(flag[i]!=0) 
		    //如果到达的是餐馆 
			if(judge[i]==0)
			{
				//贴餐馆的图，动画显示，表示骑手到达餐馆 
			 	getimage(img,"食客.jpg",0,0);
				putimage(ridersstop[i].x*35,ridersstop[i].y*35, img);
				Sleep(500); 
			}
			//如果到达的是食客 
			else 
			{
				//贴食客的图，动画显示，表示骑手到达食客 
				getimage(img,"餐馆.jpg",0,0);
				putimage(ridersstop[i].x*35,ridersstop[i].y*35, img);
				Sleep(500); 
			}
	}
}
//陈绍银 
