#include <stdio.h>
#include"data_structure.h"
extern int ridersnum;
extern struct rider riders[7];
extern int over; 
void gobankrupt()
{
    if(over==1)
	{
		FILE *fptr; 
		if ((fptr =fopen("output.txt", "a")) == NULL)
			printf("can't open file output!\n");
		else
		{
			int c;
			for (c = 0; c <= ridersnum - 1; c++)
			fprintf(fptr, "����%d:�ӵ���:%d �����:%d ��ʱ��:%d\n", c, riders[c].take_orders, riders[c].finish_, riders[c].over_time);
			fprintf(fptr, "game over!");
			fclose(fptr);
		}
	}//ִ���Ʋ���������ֹͣ��������� 
}
//������ 
