/****************************************************************************
Floyd - Warshall Algorithm developed using CUDA. A 2011-2012 assignement for
Parallel Programming Course of Electrical and Computer Engineering Department
in the Aristotle Faculty of Enginnering - Thessaloniki.

*****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>

#define INF 100000000
#define BLOCKSIZE 128
#define BITSFT 7 //log2(BLOCKSIZE)


/*****************************************
Array Generator - filling weight matrices
according to Floyd-Warshall theory.
******************************************/
void generate(float *A,float *D,float *Q,int vertices)
{
   int i,j;	
   srand ( time(NULL) );
   for(i=0;i<vertices;i++)
     {
        for(j=0;j<vertices;j++)
	  {
	     A[i*vertices+j]=(float)(rand()%10000);// Insert edge-weight randomly limited to 10000
	     Q[i*vertices+j]=0;			
	     if(i==j)
	       {
		  A[i*vertices+j]=0;
		  D[i*vertices+j]=INF;
	       }
	     else if(A[i*vertices+j]==0)
	       {
		  A[i*vertices+j]=INF;
		  D[i*vertices+j]=INF;
               }
		  if(A[i*vertices+j]!=INF&&D[i*vertices+j]!=INF)
		  D[i*vertices+j]=A[i*vertices+j];
	  }
     }
}

/************************************************************
Serial function of Floyd Warshall Algorithm. Find pseudocode 
at http://en.wikipedia.org/wiki/Floyd_Warshall_algorithm and
translated into C.
*************************************************************/
void sFloyd(float *D,float *Q,int vertices)
{
   int i,j,k;
   for(k=0;k<vertices;k++)
     {
        for(i=0;i<vertices;i++)
	  {
	     for(j=0;j<vertices;j++)
	       {
		  if((D[i*vertices+k]+D[k*vertices+j])<D[i*vertices+j])
		    {
		       D[i*vertices+j]=D[i*vertices+k]+D[k*vertices+j];
		       Q[i*vertices+j]=k;
		    }
	       }
	  }
     }
}

/*****************************************
Parallel Version of Floyd Warshall using
Cuda global.
******************************************/
__global__ void pFloyd(float *D,float *Q,int vertices,int k,int k2)
{
   int i,j,index;
   i= blockIdx.x;
   j=(blockIdx.y << BITSFT) + threadIdx.x;
   index=(i << vertices)+j; 				//vertices equals log2(vertices).
     if((D[(i << vertices)+k]+D[(k2)+j])<D[index])
       {
	  D[index]=D[(i << vertices)+k]+D[(k2)+j];
	  Q[index]=k;
       }
}

/***************************************
Check Function. Check if matrices
D and Q from serial Floyd match the
parallel ones.
****************************************/
void check(float *parallelD,float *D,float *parallelQ,float *Q,int vertices)
{
   int i,j;
   int err=0; 
   for(i=0;i<vertices;i++)
     {
        for(j=0;j<vertices;j++)
	  {
	     printf("parallelD:%f = realD:%f\n",parallelD[i*vertices+j],D[i*vertices+j]);	//Nice old print CHECK
	     if((parallelD[i*vertices+j]!=D[i*vertices+j])||(parallelQ[i*vertices+j]!=Q[i*vertices+j]))
	     err++;
	  }
     }
     printf("ERRORS:%d\n",err);
}	

/*************************************************************/
int main ( int argc, char *argv[] )
{
   float *A, *D, *Q, *parallelD, *parallelQ, *dev_D, *dev_Q;
   int i,vertices,n,k2;	
   if(argc!=2)
   {
      printf("You forgot to ENTER vertices argument.\n./<program name> <number  of vertices>=\n");
      return 1;
   }
   /*Variable init*/	
   vertices=atoi(argv[1]);
   n=(int)log2((float)vertices);
   const int size = vertices*vertices*sizeof(float);
   dim3 dimBlock(BLOCKSIZE,1);
   dim3 dimGrid(vertices,vertices/BLOCKSIZE);
   cudaMalloc( (void**)&dev_D, size);
   cudaMalloc( (void**)&dev_Q, size);
   A=(float*)malloc(size);
   D=(float*)malloc(size);	
   Q=(float*)malloc(size);
   parallelD=(float*)malloc(size);
   parallelQ=(float*)malloc(size);
   generate(A,D,Q,vertices);

   struct timeval first, second, lapsed;
   struct timezone tzp;
   gettimeofday(&first, &tzp); //Calculation time plus GPU memory transfer time.
   cudaMemcpy(dev_D,D,size,cudaMemcpyHostToDevice);
   cudaMemcpy(dev_Q,Q,size,cudaMemcpyHostToDevice);
//	gettimeofday(&first,&tzp);  //calculation's duration time ONLY.
   for(i=0;i<vertices;i++)
     {	
        k2=i*vertices;	
	pFloyd<<<dimGrid,dimBlock>>>(dev_D,dev_Q,n,i,k2);
     }
   cudaThreadSynchronize();
//	gettimeofday(&second,&tzp);  //calculation's duration time ONLY.
   cudaMemcpy(parallelD,dev_D,size,cudaMemcpyDeviceToHost);
   cudaMemcpy(parallelQ,dev_Q,size,cudaMemcpyDeviceToHost);
   sFloyd(D,Q,vertices);
   gettimeofday(&second, &tzp);  //Calculation time plus GPU memory transfer time.
   if(first.tv_usec>second.tv_usec)
     {
        second.tv_usec += 1000000;
        second.tv_sec--;
     }
   lapsed.tv_usec = second.tv_usec - first.tv_usec;
   lapsed.tv_sec = second.tv_sec - first.tv_sec;
   check(parallelD,D,parallelQ,Q,vertices);
   printf("Time elapsed: %lu, %lu s\n", lapsed.tv_sec,lapsed.tv_usec);
   cudaFree(dev_D);
   cudaFree(dev_Q);
   free(A);
   free(D);
   free(Q);
   free(parallelD);
   free(parallelQ);
   return 0;
}
