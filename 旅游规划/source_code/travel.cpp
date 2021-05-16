#include "DataStructure.h"
#include <iostream>
#include <string>
#include "travel.h"
#include "droute.h"
#include <time.h>
#include<process.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

extern struct passenger passer;
extern struct city graph;
extern int currenttime;
extern struct order *command,*end;
extern int over;
extern struct passenger passer;
extern int flag;//设计路线信号 
extern int fflag;//启动时间信号 
extern struct route line1[10];
extern struct route line2[10];
extern struct route temp;
extern int ll,middle;
extern int ffflag;//一个计划已结束 
extern time_t start;
extern int differtime;
extern int fffflag;//同步锁
extern int state; 
extern int st;
extern int ptime;
void FileInput(const char* file_name)
{
	FILE *fp;
	fp=fopen(file_name,"r");
	if(fp==NULL)
	printf("文件打开失败!"); 
		char command[30]; 
		int j,i=0;
		int num=0;
		char city_name[30];
		char mode;
		float danger;
		while (!feof(fp))
	{
		fscanf(fp,"%s",command);
		if (strcmp(command,"city")==0)
		{
			fscanf(fp,"%s%f",city_name,&danger);
				while (strcmp(city_name,"-n")!=0)
				{
					//graph.vexs[i]=city_name;
					strcpy(graph.vexs[i],city_name);
					graph.danger[i]=danger;
					fscanf(fp,"%s%f",city_name,&danger);
					i++;					
				}
		}
		else if (strcmp(command,"trp")==0)
		{
			fscanf(fp,"%c",&mode);
			fscanf(fp,"%c",&mode);
				int  tDeparture, tArrival,a,b;
				char cityA[30], cityB[30];
			fscanf(fp,"%s",cityA);
			fscanf(fp,"%s",cityB);
			fscanf(fp,"%d",&tDeparture);
			fscanf(fp,"%d",&tArrival);
			for(j=0;j<=9;j++)
			{
			if(strcmp(graph.vexs[j],cityA)==0)
			a=j;
			if(strcmp(graph.vexs[j],cityB)==0)
			b=j;
			}
			num=graph.edgnum[a][b];
			graph.matrix[a][b][num].tool=mode;
			graph.matrix[a][b][num].DeTime=tDeparture;
			graph.matrix[a][b][num].ArTime=tArrival;
			graph.edgnum[a][b]++;
			//graph.matrix[a][b].finish	
		}
}
	fclose(fp);
}

void FileOutput(int kind)
{
	FILE *fp;
	fp=fopen("log.txt","a");
	if(fp==NULL)
	printf("日志文件打开失败!"); 
	if(kind==0)
	{
		fprintf(fp,"旅客状态：时间：%d ",(currenttime%24));
		if(passer.start==passer.finish)
		fprintf(fp,"当前所在城市：%d\n",passer.start);
		else
		fprintf(fp,"旅客乘坐%c从%d到%d\n",passer.tool,passer.start,passer.finish);
	}
	if(kind==1)
	{
		fprintf(fp,"旅客想要到城市%d旅行,选择策略为%d,时间限制为%d\n",command->finish,command->strategy,command->TimeLimit);// 
	}
	fclose(fp);
}
void Init(const char* file_name)
{
		// Input data
		FileInput(file_name);//读取文件，建好邻接矩阵 
		int a,b,j;
		for(int i=0;i<10;i++)
		printf("序号%d代表城市%s\n",i,graph.vexs[i]);
		printf("交通工具：c表示汽车，t表示火车，p表示飞机\n");
		printf("低风险策略0，限制时间内低风险策略1\n");
//		for(a=0;a<10;a++)
//		for(b=0;b<10;b++)
//		if(a!=b)
//		for(j=0;j<graph.edgnum[a][b];j++)
//		printf("%s到%s的出发时间：%d,到达时间：%d,交通工具：%c\n",graph.vexs[a],graph.vexs[b],graph.matrix[a][b][j].DeTime,graph.matrix[a][b][j].ArTime,graph.matrix[a][b][j].tool);
		printf("请输入旅客所在城市序号：") ;
		scanf("%d",&passer.start);
		passer.finish=passer.start;
		passer.tool='0';
		FileOutput(0);
		 
} 

void change()//时间推进有问题，日志输出有问题，唉， 
{

	if(state==1)
	{
		if(differtime!=1)
	{
	differtime--;
	passer.start=line2[middle].finish;
	passer.finish=line2[middle+1].finish;
	passer.tool=line2[middle+1].tool;
	FileOutput(0);
	printf("旅客状态：时间：%d 旅客乘坐%c从%d到%d\n",currenttime%24,passer.tool,passer.start,passer.finish);
	}//旅行状态 
	else
	{
		state=0;
		middle++;
	}
	
	}
		if(state==0)
	{
		passer.finish=line2[middle].finish;
		passer.start=line2[middle].finish;
		FileOutput(0);
		printf("旅客状态：时间：%d 当前所在城市：%d\n",currenttime%24,passer.start);
		if(middle==ll)
		{
		ll=0;
		middle=0;
		ffflag=1;
		flag=1; 	
		}//这次旅行结束 
		else 
		if((currenttime%24)==line2[middle+1].Detime)
		{
			state=1;
			if(line2[middle+1].Detime<line2[middle+1].Artime)
			differtime=line2[middle+1].Artime-line2[middle+1].Detime;
			else
			differtime=24+line2[middle+1].Artime-line2[middle+1].Detime;
		}
	}//等待状态
	fffflag=0;
}
unsigned __stdcall fnInput(void* pArguments)
{
	char ch;
	
	printf("P:输入旅行计划，Q：查询旅客当前状态，E：结束旅行\n请输入当前操作");
	while(1)
	{
	scanf("%c",&ch);
	if(ch=='P')
	{
		int finish,str,time=0;
		struct order *node;
		printf("输入城市：");
	  	scanf("%d",&finish);
		printf("输入策略：") ;
		scanf("%d",&str);
		if(str==1)
		{
		printf("输入限制时间：");
		scanf("%d",&time);
		}
		node=(order*)malloc(sizeof(order));
		node->finish=finish;node->strategy=str;node->TimeLimit=time;
	  if(command==NULL)
	  {
	  command=node;
	  end=command;
	  }
	  else 
	  {
	  end->nextorder=node;
	  end=node;
	  }
	  end->nextorder=NULL;
	} 
	if (ch=='Q')
	{
		printf("旅客状态：时间：%d",(currenttime%24));
		if(passer.start==passer.finish)
		printf("当前所在城市：%d\n",passer.start);
		else
		printf("旅客乘坐%c从%d到%d\n",passer.tool,passer.start,passer.finish);
	}
	if(ch=='E')
	over=1; 
	}
	 _endthreadex(0);
	 return 0;
}
void _time()
{
	time_t tfinish = clock();
     	double duration = (double)(tfinish - start) ;
     	if(((int)duration-2000*((int)duration/2000))<=2)
     	{
     	ptime++;
     	if(ptime==300)
     	{
     	currenttime++;
     	//printf("%d ",currenttime);
     	fffflag=1;
     	ptime=1;
		 }
		 }//1s加一 
	
}//计时模块 

void ordersolve() 
{
	if(ffflag==1)
	{
		struct order *node;
		node =command;
		command=command->nextorder;
		delete(node);
		ffflag=0;
	}//删除头节点 
}
void fnTimeTick()
{
	ordersolve();//命令链表处理函数 
	if(command!=NULL&&fflag==1)
	{
	start=clock();
	st=1;
	fflag=0;
	flag=1;
	}
	if(st==1)
	_time(); 
	if(command!=NULL&&flag==1)
	{
	FileOutput(1);
	design();//设计路线
	flag=0;
	printf("旅客从%d到%d的路线为：\n",passer.start,command->finish);
	printf("起始站：%d 出发时间：%d 交通工具：%c\n",line2[0].finish,line2[1].Detime,line2[1].tool);
	for(int i=1;i<ll;i++)
	printf("中间站：%d 交通工具：%c 到达时间：%d 再出发时间：%d\n",line2[i].finish,line2[i+1].tool,line2[i].Artime,line2[i+1].Detime);//再次出发时间：没有赋值 
	printf("终点站：%d 到达时间：%d\n",line2[ll].finish,line2[ll].Artime);
}
	if(command!=NULL&&fffflag==1)
	change();//时间推进，状态变化 

}
