#include "droute.h"
#include <fstream>
#include <iostream>
#include "DataStructure.h"

#define INF 9999
extern struct city graph;
extern struct route line1[10];
extern struct route line2[10];
extern struct route temp;
extern struct passenger passer;
extern struct order *command,*end;
extern int ll;

int _min(int a,int b,int c)
{
	int mini=INF;
	int k;
	for(int i=0;i<graph.edgnum[a][b];i++)//�����ж�����ȷ�� 
	if (graph.matrix[a][b][i].DeTime>=c&&(graph.matrix[a][b][i].DeTime-c)<mini)

            {

                mini = graph.matrix[a][b][i].DeTime-c;

                k = i;

            }
    else
	if(graph.matrix[a][b][i].DeTime<c&&(graph.matrix[a][b][i].DeTime+24-c)<mini)
	{

                mini = graph.matrix[a][b][i].DeTime+24-c;

                k = i;

    }
    if(mini!=INF)
    {
    temp.Artime=graph.matrix[a][b][k].ArTime;
    temp.tool=graph.matrix[a][b][k].tool;
    temp.finish=a;
    temp.Detime=graph.matrix[a][b][k].DeTime;
	}
    return mini==INF?INF:mini*(int)(graph.danger[a]*10);
}

void plan(int fi)
{
	if(fi==passer.start)
	{
	line2[ll].finish=passer.start;
	line2[ll].Artime=0;
	}
	else
	{
		plan(line1[fi].finish);
		ll++; 
		line2[ll].finish=fi;
		line2[ll].Artime=line1[fi].Artime;
		line2[ll].tool=line1[fi].tool;//��fi�Ľ�ͨ������line1��tool�Լ�����ʱ��͵���ʱ�� 
		line2[ll].Detime=line1[fi].Detime;
	}//�߼��ǶԵ� 
	
}
void design()
{
	if(command->strategy==0)
	{
	int i,j,k;

    int min;

    int tmp;

    int flag[10];      // flag[i]=1��ʾ"����vs"��"����i"�����·���ѳɹ���ȡ��

	int vs=passer.start;

    // ��ʼ��
   // int prev[10];
	int dist[10];
    for (i = 0; i < 10; i++)

    {

        flag[i] = 0;              // ����i�����·����û��ȡ����

        //prev[i] = 0;              // ����i��ǰ������Ϊ0��

        dist[i] = _min(vs,i,0);// ����i�����·��Ϊ"����vs"��"����i"��Ȩ��
        
        line1[i].finish=vs;//�����ǰ���ڵ�
		
		if(dist[i]!=INF) 
		{
		line1[i].Artime=temp.Artime;
		line1[i].tool=temp.tool;
		line1[i].Detime=temp.Detime; 
	}

    }

 

    // ��"����vs"������г�ʼ��

    flag[vs] = 1;

    dist[vs] = 0;

 

    // ����G.vexnum-1�Σ�ÿ���ҳ�һ����������·����

    for (i = 1; i <10; i++)

    {

        // Ѱ�ҵ�ǰ��С��·����

        // ������δ��ȡ���·���Ķ����У��ҵ���vs����Ķ���(k)��

        min = INF;

        for (j = 0; j <10; j++)

        {

            if (flag[j]==0 && dist[j]<min)

            {

                min = dist[j];

                k = j;

            }

        }

        // ���"����k"Ϊ�Ѿ���ȡ�����·��

        flag[k] = 1;

 

        // ������ǰ���·����ǰ������

        // �������Ѿ�"����k�����·��"֮�󣬸���"δ��ȡ���·���Ķ�������·����ǰ������"��

        for (j = 0; j < 10; j++)

        {

            if(graph.edgnum[k][j]==0)
            tmp=INF;
            else
            tmp=min+_min(k,j,line1[k].Artime);
            //��ֹ��� 
            
            if (flag[j] == 0 && (tmp  < dist[j]) )

            {

                dist[j] = tmp;

                //prev[j] = k;
                line1[j].finish=k;
                line1[j].Artime= temp.Artime;
                line1[j].tool=temp.tool;
                line1[j].Detime=temp.Detime;

            }

        }

    }
	int fi=command->finish;
	plan(fi);  //û��д������ֵ 
}
	else
	{
		//����ʱ����� 
	}
}
