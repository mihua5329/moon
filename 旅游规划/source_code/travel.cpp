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
extern int flag;//���·���ź� 
extern int fflag;//����ʱ���ź� 
extern struct route line1[10];
extern struct route line2[10];
extern struct route temp;
extern int ll,middle;
extern int ffflag;//һ���ƻ��ѽ��� 
extern time_t start;
extern int differtime;
extern int fffflag;//ͬ����
extern int state; 
extern int st;
extern int ptime;
void FileInput(const char* file_name)
{
	FILE *fp;
	fp=fopen(file_name,"r");
	if(fp==NULL)
	printf("�ļ���ʧ��!"); 
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
	printf("��־�ļ���ʧ��!"); 
	if(kind==0)
	{
		fprintf(fp,"�ÿ�״̬��ʱ�䣺%d ",(currenttime%24));
		if(passer.start==passer.finish)
		fprintf(fp,"��ǰ���ڳ��У�%d\n",passer.start);
		else
		fprintf(fp,"�ÿͳ���%c��%d��%d\n",passer.tool,passer.start,passer.finish);
	}
	if(kind==1)
	{
		fprintf(fp,"�ÿ���Ҫ������%d����,ѡ�����Ϊ%d,ʱ������Ϊ%d\n",command->finish,command->strategy,command->TimeLimit);// 
	}
	fclose(fp);
}
void Init(const char* file_name)
{
		// Input data
		FileInput(file_name);//��ȡ�ļ��������ڽӾ��� 
		int a,b,j;
		for(int i=0;i<10;i++)
		printf("���%d�������%s\n",i,graph.vexs[i]);
		printf("��ͨ���ߣ�c��ʾ������t��ʾ�𳵣�p��ʾ�ɻ�\n");
		printf("�ͷ��ղ���0������ʱ���ڵͷ��ղ���1\n");
//		for(a=0;a<10;a++)
//		for(b=0;b<10;b++)
//		if(a!=b)
//		for(j=0;j<graph.edgnum[a][b];j++)
//		printf("%s��%s�ĳ���ʱ�䣺%d,����ʱ�䣺%d,��ͨ���ߣ�%c\n",graph.vexs[a],graph.vexs[b],graph.matrix[a][b][j].DeTime,graph.matrix[a][b][j].ArTime,graph.matrix[a][b][j].tool);
		printf("�������ÿ����ڳ�����ţ�") ;
		scanf("%d",&passer.start);
		passer.finish=passer.start;
		passer.tool='0';
		FileOutput(0);
		 
} 

void change()//ʱ���ƽ������⣬��־��������⣬���� 
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
	printf("�ÿ�״̬��ʱ�䣺%d �ÿͳ���%c��%d��%d\n",currenttime%24,passer.tool,passer.start,passer.finish);
	}//����״̬ 
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
		printf("�ÿ�״̬��ʱ�䣺%d ��ǰ���ڳ��У�%d\n",currenttime%24,passer.start);
		if(middle==ll)
		{
		ll=0;
		middle=0;
		ffflag=1;
		flag=1; 	
		}//������н��� 
		else 
		if((currenttime%24)==line2[middle+1].Detime)
		{
			state=1;
			if(line2[middle+1].Detime<line2[middle+1].Artime)
			differtime=line2[middle+1].Artime-line2[middle+1].Detime;
			else
			differtime=24+line2[middle+1].Artime-line2[middle+1].Detime;
		}
	}//�ȴ�״̬
	fffflag=0;
}
unsigned __stdcall fnInput(void* pArguments)
{
	char ch;
	
	printf("P:�������мƻ���Q����ѯ�ÿ͵�ǰ״̬��E����������\n�����뵱ǰ����");
	while(1)
	{
	scanf("%c",&ch);
	if(ch=='P')
	{
		int finish,str,time=0;
		struct order *node;
		printf("������У�");
	  	scanf("%d",&finish);
		printf("������ԣ�") ;
		scanf("%d",&str);
		if(str==1)
		{
		printf("��������ʱ�䣺");
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
		printf("�ÿ�״̬��ʱ�䣺%d",(currenttime%24));
		if(passer.start==passer.finish)
		printf("��ǰ���ڳ��У�%d\n",passer.start);
		else
		printf("�ÿͳ���%c��%d��%d\n",passer.tool,passer.start,passer.finish);
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
		 }//1s��һ 
	
}//��ʱģ�� 

void ordersolve() 
{
	if(ffflag==1)
	{
		struct order *node;
		node =command;
		command=command->nextorder;
		delete(node);
		ffflag=0;
	}//ɾ��ͷ�ڵ� 
}
void fnTimeTick()
{
	ordersolve();//������������ 
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
	design();//���·��
	flag=0;
	printf("�ÿʹ�%d��%d��·��Ϊ��\n",passer.start,command->finish);
	printf("��ʼվ��%d ����ʱ�䣺%d ��ͨ���ߣ�%c\n",line2[0].finish,line2[1].Detime,line2[1].tool);
	for(int i=1;i<ll;i++)
	printf("�м�վ��%d ��ͨ���ߣ�%c ����ʱ�䣺%d �ٳ���ʱ�䣺%d\n",line2[i].finish,line2[i+1].tool,line2[i].Artime,line2[i+1].Detime);//�ٴγ���ʱ�䣺û�и�ֵ 
	printf("�յ�վ��%d ����ʱ�䣺%d\n",line2[ll].finish,line2[ll].Artime);
}
	if(command!=NULL&&fffflag==1)
	change();//ʱ���ƽ���״̬�仯 

}
