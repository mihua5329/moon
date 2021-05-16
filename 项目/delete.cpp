#include<stdio.h>
#include<stdlib.h>
#include"data_structure.h"
void deleteNodes(LISTNODE * *sptr)
{
    LISTNODE * currentptr;
    currentptr=*sptr;/*将头接点地址赋给currentPtr*/
        /*查找待删除结点，若找到，则由currentPt指向该结点*/
         if (currentptr!=NULL)
		 { /*如果找到要删除的结点*/
            	 *sptr=currentptr->nextptr;/*更新头结点*/
            	 free(currentptr); 
            	 currentptr = *sptr; /* currentPtr指向头结点*/
         }
}//陈绍银 
