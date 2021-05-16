#pragma once
//#ifndef __DATA_STRUCTURE_H__
//#define __DATA_STRUCTURE_H__
//#define _CRT_INSECURE_DEPRECATE
struct spot
{
	int number;//单号 
	int x;
	int y;//坐标位置点 
	int time;//下单时间 
	int arrival;//是否可达 
	struct spot* nextspot;
	struct spot* correspondptr;
	struct spot* lptr;
	int ifarrival;
};  //点列链表节点
struct point
{
	int flag;//A或B标志 
	int x;
	int y;
	int time;//下单时间
	struct spot* preptr;
	struct point* nextptr;
	int number;//单号 
};//路径链表节点 
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

