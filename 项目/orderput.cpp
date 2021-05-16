#include<stdio.h>
#include<stdlib.h>
#include"data_structure.h"
#include"delete.h"
extern int currenttime;
extern LISTNODE * allheadptr;
extern LISTNODE * headptr;
int checkTime(int n)//检查当前时间与订单下单时间相同 
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
		if(allheadptr!=NULL)//如果订单未派送完 
		{
	    aim=checkTime(allheadptr->time);//检查当前时间单位内是否有订单 
		if(aim==1)
		{
			//有订单 
		currentptr=(LISTNODE *)malloc(sizeof(LISTNODE));//分配结点内存 
		if(currentptr!=NULL)
		{
		currentptr->number=allheadptr->number;
		currentptr->time=allheadptr->time;
		currentptr->x=allheadptr->x;
		currentptr->y=allheadptr->y;
		currentptr->a=allheadptr->a;
		currentptr->b=allheadptr->b;
		if(headptr==NULL)//若创建的是头结点 
		{
		headptr=currentptr;
		lastptr=currentptr;
		deleteNodes(&allheadptr);//删除总订单的头结点，使头结点后移 
	    }
	   else
	   {
	   // 将结点连上链表尾结点lastptr，并使lastptr指向当前链表的最后一个结点
    	lastptr->nextptr=currentptr;
    	lastptr=currentptr;
    	deleteNodes(&allheadptr);
		}
	    }
        }
	else flag=0;// 若总订单头结点为空，说明已经分配完所有订单，将flag赋值为0
    }
    else  flag=0;//若该时间单位内没有订单，将flag的值改为0 
  }
	if(headptr!=NULL)
	lastptr->nextptr=NULL;//设置结束标志 
}
//陈绍银 
