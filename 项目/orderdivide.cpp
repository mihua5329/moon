#include<stdlib.h>
#include<stdio.h>
#include"data_structure.h"
#include"delete.h"
extern int ridersnum,takeorders;
extern LISTNODE *neworder1;
extern LISTNODE neworder2;
extern struct rider riders[];
extern struct spot * lastaspot[7];
extern struct spot * lastbspot[7];
extern int both[];
static int FindMinNum(int a[], int n)
{
	int min = a[0];//将a[0]设为最小值 
	int i=0;
	for (i=1; i < n; i++)
	{
		if (a[i]<min)//当前的数比最小值小 
		{
			min = a[i];
		}
	}
	return min;
}

static int FindIndex(int a[], int n, int min)
{
	int index = 0;//记录最小值的下标，初始值是0
	int i;
	for (i = 0; i < n; i++)
	{
		if (a[i] == min)//当前的数和最小值相等 
		{
			index = i;
			break;
		}
	}
	return index;
}

void  order_Divide()
{
	int i,a,t,j,temp,temp1,flag,symbol,m,flag1;
	struct spot * currentaspot=NULL;
    struct spot * currentbspot=NULL;
	static int Long1[3]={0},Long2[3]={0},Long[3]={0};
	if(neworder1!=NULL&&neworder1->flag==1)
	{
	currentaspot=(struct spot *)malloc(sizeof(struct spot));//分配结点内存
	currentbspot=(struct spot *)malloc(sizeof(struct spot));
	     //将订单分为A,B任务 
	     printf(" %d ",neworder1->x) ;
    	currentaspot->time=neworder1->time;//A任务
    	currentaspot->x=neworder1->x;
    	currentaspot->y=neworder1->y;
    	currentaspot->number=neworder1->number;
    	currentaspot->arrival=0;
    	currentaspot->ifarrival=0;
    	currentaspot->correspondptr=NULL; 
    	//B任务 
    	currentbspot->time=neworder1->time;
    	currentbspot->x=neworder1->a;
    	currentbspot->y=neworder1->b;
    	currentbspot->number=neworder1->number;
    	currentbspot->arrival=0;
    	currentbspot->ifarrival=0;
    	currentbspot->correspondptr=currentaspot;//存储订单中食客对应的餐馆的信息 
    	for(a=0,m=0;a<=ridersnum-1;a++) 
		{
			if(riders[a].ctake_orders<3)//找出未派送订单数小于3的骑手 
			{
			Long[m]=a;
			m++;
		}
		} 
	    for(a=0;a<m;a++) 
	    {
		Long2[a]=riders[Long[a]].ctake_orders;	
		}
        temp=FindMinNum(Long2,m);//找出最小待派送订单 
		symbol=FindIndex(Long2,m,temp);//返回下标 
		i=Long[symbol];//待派送订单最少的骑手 
    	riders[i].mark=i;
		//给骑手分配订单 
    	if(riders[i].headaspot==NULL)//若插入的是头结点
		{
    	currentaspot->lptr=NULL;//lptr存储该骑手上一个订单的餐馆的信息 
        riders[i].headaspot=currentaspot;
        lastaspot[i]=currentaspot;
    }
     else
	    {
		//将结点连上链表尾结点，并使lastaspot指向当前链表的最后一个结点
	    	currentaspot->lptr=lastaspot[i];
	    	lastaspot[i]->nextspot=currentaspot;
	    	lastaspot[i]=currentaspot;
		}
		if(riders[i].headbspot==NULL)//若插入的是头结点 
		{
		currentbspot->lptr=NULL;
		riders[i].headbspot=currentbspot;	
		lastbspot[i]=currentbspot;
	}
	  else
	  {
	  	//将结点连上链表尾结点，并使lastbspot指向当前链表的最后一个结点
	  	currentbspot->lptr=lastbspot[i];
	  	lastbspot[i]->nextspot=currentbspot;
	    	lastbspot[i]=currentbspot;
		  }		
		
	riders[i].take_orders++;//骑手总接单数加一 
	riders[i].rflag=1;
	riders[i].ctake_orders++;//骑手待派送订单数加一 
	neworder1=NULL;//使头结点后移 
	} 
if(neworder2.flag==1)
	{
	currentaspot=(struct spot *)malloc(sizeof(struct spot));//分配结点内存
	currentbspot=(struct spot *)malloc(sizeof(struct spot));
	     //将订单分为A,B任务 
	     printf(" %d ",neworder2.x) ;
    	currentaspot->time=neworder2.time;//A任务
    	currentaspot->x=neworder2.x;
    	currentaspot->y=neworder2.y;
    	currentaspot->number=neworder2.number;
    	currentaspot->arrival=0;
    	currentaspot->ifarrival=0;
    	currentaspot->correspondptr=NULL; 
    	//B任务 
    	currentbspot->time=neworder2.time;
    	currentbspot->x=neworder2.a;
    	currentbspot->y=neworder2.b;
    	currentbspot->number=neworder2.number;
    	currentbspot->arrival=0;
    	currentbspot->ifarrival=0;
    	currentbspot->correspondptr=currentaspot;//存储订单中食客对应的餐馆的信息 
    	for(a=0,m=0;a<=ridersnum-1;a++) 
		{
			if(riders[a].ctake_orders<3)//找出未派送订单数小于3的骑手 
			{
			Long[m]=a;
			m++;
		}
		} 
	    for(a=0;a<m;a++) 
	    {
		Long2[a]=riders[Long[a]].ctake_orders;	
		}
        temp=FindMinNum(Long2,m);//找出最小待派送订单 
		symbol=FindIndex(Long2,m,temp);//返回下标 
		i=Long[symbol];//待派送订单最少的骑手 
    	riders[i].mark=i;
		//给骑手分配订单 
    	if(riders[i].headaspot==NULL)//若插入的是头结点
		{
    	currentaspot->lptr=NULL;//lptr存储该骑手上一个订单的餐馆的信息 
        riders[i].headaspot=currentaspot;
        lastaspot[i]=currentaspot;
    }
     else
	    {
		//将结点连上链表尾结点，并使lastaspot指向当前链表的最后一个结点
	    	currentaspot->lptr=lastaspot[i];
	    	lastaspot[i]->nextspot=currentaspot;
	    	lastaspot[i]=currentaspot;
		}
		if(riders[i].headbspot==NULL)//若插入的是头结点 
		{
		currentbspot->lptr=NULL;
		riders[i].headbspot=currentbspot;	
		lastbspot[i]=currentbspot;
	}
	  else
	  {
	  	//将结点连上链表尾结点，并使lastbspot指向当前链表的最后一个结点
	  	currentbspot->lptr=lastbspot[i];
	  	lastbspot[i]->nextspot=currentbspot;
	    	lastbspot[i]=currentbspot;
		  }		
		
	riders[i].take_orders++;//骑手总接单数加一 
	riders[i].rflag=1;
	riders[i].ctake_orders++;//骑手待派送订单数加一 
	
}
    for(i=0;i<ridersnum;i++)
    {
    	if(riders[i].headaspot!=NULL&&riders[i].headbspot!=NULL)//设置链表结束标志
    	{
    		lastaspot[i]->nextspot=NULL;
    		lastbspot[i]->nextspot=NULL;
		}
	}
	neworder2.a=0;
	neworder2.b=0;
	neworder2.flag=0;
	neworder2.number=0;
	neworder2.time=0;
	neworder2.x=0;
	neworder2.y=0;
	neworder2.nextptr=NULL;
}
//陈绍银 
