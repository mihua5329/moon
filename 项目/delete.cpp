#include<stdio.h>
#include<stdlib.h>
#include"data_structure.h"
void deleteNodes(LISTNODE * *sptr)
{
    LISTNODE * currentptr;
    currentptr=*sptr;/*��ͷ�ӵ��ַ����currentPtr*/
        /*���Ҵ�ɾ����㣬���ҵ�������currentPtָ��ý��*/
         if (currentptr!=NULL)
		 { /*����ҵ�Ҫɾ���Ľ��*/
            	 *sptr=currentptr->nextptr;/*����ͷ���*/
            	 free(currentptr); 
            	 currentptr = *sptr; /* currentPtrָ��ͷ���*/
         }
}//������ 
