#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h> 
#include<windows.h>
#include <process.h>
#include<graphics.h>  
#include <conio.h>
#include"data_structure.h" 
extern struct rider riders[5]; 
extern int init,ridersnum, currenttime, money, takeorders, finish, timeout;
extern int finishnum[5], finenum[5], flag[5];//���δ洢�������ֽᵥ�ţ�����ţ��Լ����ֵ���ʱ������ɣ�1)���ǳ�ʱ��2������û����0��  
extern struct position ridersstop[7];//�洢��������������� 
extern int judge[7];//���ֵ����ǲ͹ݣ�1������ʳ�ͣ�0��
void out_put() 
{
	FILE *fptr;
	if ((fptr =fopen("output.txt", "a")) == NULL)
		printf("can't open file output!\n");
	else 
	{
	int p;
	fprintf(fptr, "ʱ��:%d\n", currenttime);
	fprintf(fptr, "Ǯ:%d\n", money);
	fprintf(fptr, "�ӵ���:%d\n", takeorders);
	fprintf(fptr, "�����:%d;", finish);
	fprintf(fptr, "�ᵥ��");
	for (p = 0; p <= ridersnum - 1; p++)
		if (flag[p] == 1)
			fprintf(fptr, "%d ", finishnum[p]);
	fprintf(fptr, ";\n" );
	fprintf(fptr, "��ʱ��:%d;", timeout);
	fprintf(fptr, "������");
	for (p = 0; p <= ridersnum - 1; p++)
		if (flag[p] == 2)
			fprintf(fptr, "%d ", finenum[p]);
	fprintf(fptr, ";\n");
	int num;
	for (num = 0; num <= ridersnum - 1; num++) 
	{
		fprintf(fptr, "����%dλ�ã�%d,%d;", num, riders[num].x, riders[num].y);
		fprintf(fptr, "ͣ����");
		if(flag[num]!=0)
			if(judge[num]==0)
			fprintf(fptr,"ʳ�� %d %d",ridersstop[num].x,ridersstop[num].y);
			else fprintf(fptr,"�͹� %d %d",ridersstop[num].x,ridersstop[num].y);
			fprintf(fptr,";\n");
	}
	fclose(fptr);
	}
     //����һʱ�䵥λ�ı��������ȫ������ 
    cleardevice(); 
	PIMAGE img;
	img=newimage();
	//������ 
    getimage(img,"������.jpg",0,0);
	putimage(0, 0, img);
	
    //�������飬��������ַ���
	char ridernum[10],current[10],mon[10],take[350],fini[350],timeover[10],qishou[30];
 		//ָ������߶ȿ�� 
		setfont(40, 0, "����");
        sprintf(current, "%d", currenttime);//������currenttimeת�����ַ���
        outtextxy(730, 200, "ʱ�䣺");
        outtextxy(850, 200, current);//���ַ�������������� 
        
        sprintf(mon, "%d", money);//������moneyת�����ַ���
        outtextxy(730, 240, "Ǯ��");
        outtextxy(810, 240, mon);
         
         sprintf(qishou, "%d", ridersnum);//������moneyת�����ַ���
        outtextxy(730, 280, "���֣�");
         outtextxy(850, 280, qishou);
         
        sprintf(take, "%d", takeorders);//������takeordersת�����ַ���
        outtextxy(730, 320, "�ӵ�����");
         outtextxy(890, 320, take);
         
        sprintf(fini, "%d", finish);//������finishת�����ַ���
        outtextxy(730, 360, "�������");
         outtextxy(890, 360, fini);
         
        sprintf(timeover, "%d", timeout);//������timeoutת�����ַ���
        outtextxy(730, 400,	"��ʱ����");
        outtextxy(890, 400, timeover);
        
       for(int i=0;i<ridersnum;i++)
	   {
		//������ 
		if(riders[i].x%2!=0&&riders[i].y%2==0)
		{
			//������ֺ�����Ϊ������������Ϊż�� 
			getimage(img,"����.jpg",0,0);
			putimage(((riders[i].x+1)/2)*40+(riders[i].x/2)*30,(riders[i].y/2)*70+5,img);
		}
	    else
	    {
		   //������λ�� 
	    	getimage(img,"����.jpg",0,0);
	    	putimage((riders[i].x/2)*70+5,((riders[i].y+1)/2)*40+(riders[i].y/2)*30,img);
	    }
	    //�ж������Ƿ񵽴�͹ݻ�ʳ��
		if(flag[i]!=0) 
		    //���������ǲ͹� 
			if(judge[i]==0)
			{
				//���͹ݵ�ͼ��������ʾ����ʾ���ֵ���͹� 
			 	getimage(img,"ʳ��.jpg",0,0);
				putimage(ridersstop[i].x*35,ridersstop[i].y*35, img);
				Sleep(500); 
			}
			//����������ʳ�� 
			else 
			{
				//��ʳ�͵�ͼ��������ʾ����ʾ���ֵ���ʳ�� 
				getimage(img,"�͹�.jpg",0,0);
				putimage(ridersstop[i].x*35,ridersstop[i].y*35, img);
				Sleep(500); 
			}
	}
}
//������ 
