#include<stdio.h>
#include<windows.h>
#include<graphics.h>  
#include <conio.h> 
#include<time.h>
#include"data_structure.h"
#include<process.h>
#include<math.h>
extern int ffflag;//ͬ����
extern HANDLE hPenMutex;//�������
extern LISTNODE *neworder1; //�����Ͷ�������
extern LISTNODE neworder2;
extern int takeorders,currenttime,money,ridersnum,finish,timeout;//��ǰʱ�䣬�ӵ�����Ǯ���������������������ʱ��
extern int len;//�����Ͷ����ĳ���
extern struct rider riders[];//��������ṹ
extern int both[]; 
unsigned __stdcall mouseput(void* pArguments)
{
	time_t start, tfinish;//�����ʱ�䣬��������ʱ�䵥λ
	initgraph(1000, 770); 
	mouse_msg msg={0};//�����Ϣ�ṹ��
	int x=0,y=0;//�洢���λ�� 
	PIMAGE img;//�洢ͼƬ 
    img=newimage();// ���ö�������
    getimage(img,"������.jpg",0,0);
   char ridernum[10],current[10],mon[10],take[350],fini[350],timeover[10],qishou[30];
 		setfont(40, 0, "����");
        //д���� 
        sprintf(current, "%d", currenttime);//������aת�����ַ���
        outtextxy(730, 200, "ʱ�䣺");
        outtextxy(850, 200, current);
        
        sprintf(mon, "%d", money);
        outtextxy(730, 240, "Ǯ��");
         outtextxy(810, 240, mon);
         
         sprintf(qishou, "%d", ridersnum);
        outtextxy(730, 280, "���֣�");
         outtextxy(850, 280, qishou);
        sprintf(take, "%d", takeorders);
        outtextxy(730, 320, "�ӵ�����");
         outtextxy(890, 320, take);
         
        sprintf(fini, "%d", finish);
        outtextxy(730, 360, "�������");
         outtextxy(890, 360, fini);
         
        sprintf(timeover, "%d", timeout);
        outtextxy(730, 400,	"��ʱ����");
        outtextxy(890, 400, timeover);
    	LISTNODE *lastptr=NULL;
        start = clock(); // ��ʼ��ʱ
        LISTNODE *currentt=NULL;
  	    currentt=(LISTNODE *)malloc(sizeof(LISTNODE));
    while(!kbhit())
    {
		if(ffflag==0){
    	//WaitForSingleObject(hPenMutex,INFINITE);
        putimage(0, 0, img);
		if(mousemsg())
	    { 
		msg = getmouse();
		if(msg.is_down())
		{
		 	both[0]++;
			mousepos(&x, &y);
	    	currentt->x=x/70*2;currentt->y=y/70*2;currentt->flag=0;
			if(neworder1==NULL)
			{
				neworder1=currentt;
			}
			else
			{	neworder2.x=currentt->x;
			    neworder2.flag=currentt->flag;
			}
			 	getimage(img,"�͹�.jpg",0,0);
				putimage(x/70*70, y/70*70, img);
			}
		 else if(msg.is_up())
		 {
		 	if(neworder1->flag==0)
			 {
			 	
		 	neworder1->time=currenttime;
			mousepos(&x, &y);
			neworder1->a=x/70*2;
			neworder1->b=y/70*2;
		    getimage(img,"ʳ��.jpg",0,0);
			putimage(x/70*70, y/70*70, img);
			both[1]++;
			takeorders++;
			neworder1->number=takeorders;
			neworder1->flag=1;
			len++;
		}
			else{
				neworder2.time=currenttime;
			mousepos(&x, &y);
			neworder2.a=x/70*2;
			neworder2.b=y/70*2;
			getimage(img,"ʳ��.jpg",0,0);
			putimage(x/70*70, y/70*70, img);
			both[1]++;
			takeorders++;
			neworder2.number=takeorders;
			neworder2.flag=1;
			len++;
			}
		 }  
     	}
     	tfinish = clock();
     	double duration = (double)(tfinish - start) ;
     	if(((int)duration-2000*((int)duration/2000))<=50)
     	{
     		ffflag=1;
     		currenttime++;
     		printf("%d ",currenttime);
		}
	}
	}
	    _endthreadex(0);	
	return 0;
}//ffflag=1��ʱ����䶩��neworder��Ȼ���Ϊ�գ�ffflag=0;
//������ 
