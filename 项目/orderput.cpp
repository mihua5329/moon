#include<stdio.h>
#include<stdlib.h>
#include"data_structure.h"
#include"delete.h"
extern int currenttime;
extern LISTNODE * allheadptr;
extern LISTNODE * headptr;
int checkTime(int n)//��鵱ǰʱ���붩���µ�ʱ����ͬ 
{
	int flag;
	if(currenttime==n)
	flag=1;
	if(currenttime!=n)
	flag=0;
	return flag;
}

void timeorder_Put()
{
	LISTNODE * currentptr=NULL,* lastptr=NULL;
	int aim,flag=1;
	while(flag)
	{
		if(allheadptr!=NULL)//�������δ������ 
		{
	    aim=checkTime(allheadptr->time);//��鵱ǰʱ�䵥λ���Ƿ��ж��� 
		if(aim==1)
		{
			//�ж��� 
		currentptr=(LISTNODE *)malloc(sizeof(LISTNODE));//�������ڴ� 
		if(currentptr!=NULL)
		{
		currentptr->number=allheadptr->number;
		currentptr->time=allheadptr->time;
		currentptr->x=allheadptr->x;
		currentptr->y=allheadptr->y;
		currentptr->a=allheadptr->a;
		currentptr->b=allheadptr->b;
		if(headptr==NULL)//����������ͷ��� 
		{
		headptr=currentptr;
		lastptr=currentptr;
		deleteNodes(&allheadptr);//ɾ���ܶ�����ͷ��㣬ʹͷ������ 
	    }
	   else
	   {
	   // �������������β���lastptr����ʹlastptrָ��ǰ��������һ�����
    	lastptr->nextptr=currentptr;
    	lastptr=currentptr;
    	deleteNodes(&allheadptr);
		}
	    }
        }
	else flag=0;// ���ܶ���ͷ���Ϊ�գ�˵���Ѿ����������ж�������flag��ֵΪ0
    }
    else  flag=0;//����ʱ�䵥λ��û�ж�������flag��ֵ��Ϊ0 
  }
	if(headptr!=NULL)
	lastptr->nextptr=NULL;//���ý�����־ 
}
//������ 
