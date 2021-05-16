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
	int min = a[0];//��a[0]��Ϊ��Сֵ 
	int i=0;
	for (i=1; i < n; i++)
	{
		if (a[i]<min)//��ǰ��������СֵС 
		{
			min = a[i];
		}
	}
	return min;
}

static int FindIndex(int a[], int n, int min)
{
	int index = 0;//��¼��Сֵ���±꣬��ʼֵ��0
	int i;
	for (i = 0; i < n; i++)
	{
		if (a[i] == min)//��ǰ��������Сֵ��� 
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
	currentaspot=(struct spot *)malloc(sizeof(struct spot));//�������ڴ�
	currentbspot=(struct spot *)malloc(sizeof(struct spot));
	     //��������ΪA,B���� 
	     printf(" %d ",neworder1->x) ;
    	currentaspot->time=neworder1->time;//A����
    	currentaspot->x=neworder1->x;
    	currentaspot->y=neworder1->y;
    	currentaspot->number=neworder1->number;
    	currentaspot->arrival=0;
    	currentaspot->ifarrival=0;
    	currentaspot->correspondptr=NULL; 
    	//B���� 
    	currentbspot->time=neworder1->time;
    	currentbspot->x=neworder1->a;
    	currentbspot->y=neworder1->b;
    	currentbspot->number=neworder1->number;
    	currentbspot->arrival=0;
    	currentbspot->ifarrival=0;
    	currentbspot->correspondptr=currentaspot;//�洢������ʳ�Ͷ�Ӧ�Ĳ͹ݵ���Ϣ 
    	for(a=0,m=0;a<=ridersnum-1;a++) 
		{
			if(riders[a].ctake_orders<3)//�ҳ�δ���Ͷ�����С��3������ 
			{
			Long[m]=a;
			m++;
		}
		} 
	    for(a=0;a<m;a++) 
	    {
		Long2[a]=riders[Long[a]].ctake_orders;	
		}
        temp=FindMinNum(Long2,m);//�ҳ���С�����Ͷ��� 
		symbol=FindIndex(Long2,m,temp);//�����±� 
		i=Long[symbol];//�����Ͷ������ٵ����� 
    	riders[i].mark=i;
		//�����ַ��䶩�� 
    	if(riders[i].headaspot==NULL)//���������ͷ���
		{
    	currentaspot->lptr=NULL;//lptr�洢��������һ�������Ĳ͹ݵ���Ϣ 
        riders[i].headaspot=currentaspot;
        lastaspot[i]=currentaspot;
    }
     else
	    {
		//�������������β��㣬��ʹlastaspotָ��ǰ��������һ�����
	    	currentaspot->lptr=lastaspot[i];
	    	lastaspot[i]->nextspot=currentaspot;
	    	lastaspot[i]=currentaspot;
		}
		if(riders[i].headbspot==NULL)//���������ͷ��� 
		{
		currentbspot->lptr=NULL;
		riders[i].headbspot=currentbspot;	
		lastbspot[i]=currentbspot;
	}
	  else
	  {
	  	//�������������β��㣬��ʹlastbspotָ��ǰ��������һ�����
	  	currentbspot->lptr=lastbspot[i];
	  	lastbspot[i]->nextspot=currentbspot;
	    	lastbspot[i]=currentbspot;
		  }		
		
	riders[i].take_orders++;//�����ܽӵ�����һ 
	riders[i].rflag=1;
	riders[i].ctake_orders++;//���ִ����Ͷ�������һ 
	neworder1=NULL;//ʹͷ������ 
	} 
if(neworder2.flag==1)
	{
	currentaspot=(struct spot *)malloc(sizeof(struct spot));//�������ڴ�
	currentbspot=(struct spot *)malloc(sizeof(struct spot));
	     //��������ΪA,B���� 
	     printf(" %d ",neworder2.x) ;
    	currentaspot->time=neworder2.time;//A����
    	currentaspot->x=neworder2.x;
    	currentaspot->y=neworder2.y;
    	currentaspot->number=neworder2.number;
    	currentaspot->arrival=0;
    	currentaspot->ifarrival=0;
    	currentaspot->correspondptr=NULL; 
    	//B���� 
    	currentbspot->time=neworder2.time;
    	currentbspot->x=neworder2.a;
    	currentbspot->y=neworder2.b;
    	currentbspot->number=neworder2.number;
    	currentbspot->arrival=0;
    	currentbspot->ifarrival=0;
    	currentbspot->correspondptr=currentaspot;//�洢������ʳ�Ͷ�Ӧ�Ĳ͹ݵ���Ϣ 
    	for(a=0,m=0;a<=ridersnum-1;a++) 
		{
			if(riders[a].ctake_orders<3)//�ҳ�δ���Ͷ�����С��3������ 
			{
			Long[m]=a;
			m++;
		}
		} 
	    for(a=0;a<m;a++) 
	    {
		Long2[a]=riders[Long[a]].ctake_orders;	
		}
        temp=FindMinNum(Long2,m);//�ҳ���С�����Ͷ��� 
		symbol=FindIndex(Long2,m,temp);//�����±� 
		i=Long[symbol];//�����Ͷ������ٵ����� 
    	riders[i].mark=i;
		//�����ַ��䶩�� 
    	if(riders[i].headaspot==NULL)//���������ͷ���
		{
    	currentaspot->lptr=NULL;//lptr�洢��������һ�������Ĳ͹ݵ���Ϣ 
        riders[i].headaspot=currentaspot;
        lastaspot[i]=currentaspot;
    }
     else
	    {
		//�������������β��㣬��ʹlastaspotָ��ǰ��������һ�����
	    	currentaspot->lptr=lastaspot[i];
	    	lastaspot[i]->nextspot=currentaspot;
	    	lastaspot[i]=currentaspot;
		}
		if(riders[i].headbspot==NULL)//���������ͷ��� 
		{
		currentbspot->lptr=NULL;
		riders[i].headbspot=currentbspot;	
		lastbspot[i]=currentbspot;
	}
	  else
	  {
	  	//�������������β��㣬��ʹlastbspotָ��ǰ��������һ�����
	  	currentbspot->lptr=lastbspot[i];
	  	lastbspot[i]->nextspot=currentbspot;
	    	lastbspot[i]=currentbspot;
		  }		
		
	riders[i].take_orders++;//�����ܽӵ�����һ 
	riders[i].rflag=1;
	riders[i].ctake_orders++;//���ִ����Ͷ�������һ 
	
}
    for(i=0;i<ridersnum;i++)
    {
    	if(riders[i].headaspot!=NULL&&riders[i].headbspot!=NULL)//�������������־
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
//������ 
