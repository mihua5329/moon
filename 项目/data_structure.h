#pragma once
//#ifndef __DATA_STRUCTURE_H__
//#define __DATA_STRUCTURE_H__
//#define _CRT_INSECURE_DEPRECATE
struct spot
{
	int number;//���� 
	int x;
	int y;//����λ�õ� 
	int time;//�µ�ʱ�� 
	int arrival;//�Ƿ�ɴ� 
	struct spot* nextspot;
	struct spot* correspondptr;
	struct spot* lptr;
	int ifarrival;
};  //��������ڵ�
struct point
{
	int flag;//A��B��־ 
	int x;
	int y;
	int time;//�µ�ʱ��
	struct spot* preptr;
	struct point* nextptr;
	int number;//���� 
};//·������ڵ� 
struct rider {
	int x, y;
	int mark;
	int rflag;
	int take_orders,ctake_orders;
	int finish_, over_time;
	struct spot* headaspot;
	struct spot* headbspot;
};
typedef struct node {
	int number;
	int time;
	int x, y, a, b;
	struct node* nextptr;
	int flag;
}LISTNODE;
struct position {
	int x, y;
};
//#endif

