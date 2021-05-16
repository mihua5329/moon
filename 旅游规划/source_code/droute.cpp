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
	for(int i=0;i<graph.edgnum[a][b];i++)//假设判断是正确的 
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
		line2[ll].tool=line1[fi].tool;//到fi的交通工具是line1的tool以及出发时间和到达时间 
		line2[ll].Detime=line1[fi].Detime;
	}//逻辑是对的 
	
}
void design()
{
	if(command->strategy==0)
	{
	int i,j,k;

    int min;

    int tmp;

    int flag[10];      // flag[i]=1表示"顶点vs"到"顶点i"的最短路径已成功获取。

	int vs=passer.start;

    // 初始化
   // int prev[10];
	int dist[10];
    for (i = 0; i < 10; i++)

    {

        flag[i] = 0;              // 顶点i的最短路径还没获取到。

        //prev[i] = 0;              // 顶点i的前驱顶点为0。

        dist[i] = _min(vs,i,0);// 顶点i的最短路径为"顶点vs"到"顶点i"的权。
        
        line1[i].finish=vs;//顶点的前驱节点
		
		if(dist[i]!=INF) 
		{
		line1[i].Artime=temp.Artime;
		line1[i].tool=temp.tool;
		line1[i].Detime=temp.Detime; 
	}

    }

 

    // 对"顶点vs"自身进行初始化

    flag[vs] = 1;

    dist[vs] = 0;

 

    // 遍历G.vexnum-1次；每次找出一个顶点的最短路径。

    for (i = 1; i <10; i++)

    {

        // 寻找当前最小的路径；

        // 即，在未获取最短路径的顶点中，找到离vs最近的顶点(k)。

        min = INF;

        for (j = 0; j <10; j++)

        {

            if (flag[j]==0 && dist[j]<min)

            {

                min = dist[j];

                k = j;

            }

        }

        // 标记"顶点k"为已经获取到最短路径

        flag[k] = 1;

 

        // 修正当前最短路径和前驱顶点

        // 即，当已经"顶点k的最短路径"之后，更新"未获取最短路径的顶点的最短路径和前驱顶点"。

        for (j = 0; j < 10; j++)

        {

            if(graph.edgnum[k][j]==0)
            tmp=INF;
            else
            tmp=min+_min(k,j,line1[k].Artime);
            //防止溢出 
            
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
	plan(fi);  //没有写出风险值 
}
	else
	{
		//限制时间策略 
	}
}
