#include<stdio.h>
#include"const.h"
#include"data_structure.h"
extern int money, ridersnum;
extern struct rider riders[7];
void buy_riders() 
{
	while (money >= 400&&ridersnum<7) 
	{
		ridersnum++; money -= PRICE;
		riders[ridersnum - 1].x = initialx;
		riders[ridersnum - 1].y = initialy;
		riders[ridersnum - 1].finish_ = 0; riders[ridersnum - 1].over_time = 0; riders[ridersnum - 1].take_orders = 0;
		riders[ridersnum - 1].headaspot = NULL; riders[ridersnum - 1].headbspot = NULL;riders[ridersnum-1].ctake_orders=0;
	}//购买骑手并进行初始化 
}
//吕文秀 
