#include <petscts.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

int im,jm,N,Nl,ms,me;
int rank, Ncpu;
double Flops;
MPI_Comm comm;

void Get_local_dims(int N,int Ncpu,int rank,int* ms,int* me);
void Form_matrix(double** Al);
void Form_RHS(double* Bl);
void Equal(double* Xl,double* Yl);
void Dot(double* Xl,double* Yl,double* sum);
void MultVecPlus(double* Zl,double* Xl,double alpha,double* Yl);
void MultMatVec(double* Sl,double **Al,double* Pl,double* P);

int main(int argc, char* argv[]) {
  PetscInitialize(&argc,&argv,NULL,help);

  double **Al,*Bl,*Xl;
  im=321, jm=im; 
  N=(im+1)*(jm+1);

  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_size(comm, &Ncpu);
  MPI_Comm_rank(comm, &rank);
  
  Get_local_dims(N,Ncpu,rank,&ms,&me);
  
  //  printf("N: %d, Ncpu:%d, rank:%d , ms:%d , me:%d \n",N,Ncpu,rank,ms,me);

  // Allocate
  int i,m;
  m=me-ms+1;
  Al=(double**)calloc(m,sizeof(double *));
  for(i=0;i<m;i++){
    Al[i]=(double*)calloc(N,sizeof(double));
  }
  Xl=(double *)calloc(N,sizeof(double));
  Bl=(double *)calloc(N,sizeof(double));


  Form_matrix(Al);
  Form_RHS(Bl);
  

  //Conjugate gradient
  int k=0,kmax=10;
  double tol=1.e-12,res=1.,Beta,Alpha;
  double *rl,*r,*Pl,*P,*Sl,*Zl;
  m=me-ms+1;
   rl=(double *)calloc(m,sizeof(double));
   r=(double *)calloc(N,sizeof(double));
   Pl=(double *)calloc(m,sizeof(double));
   P=(double *)calloc(N,sizeof(double));
   Sl=(double *)calloc(m,sizeof(double));
   Zl=(double *)calloc(m,sizeof(double));
  
   Equal(rl,Bl);
   Equal(Pl,rl);

   Flops=0.;
   clock_t start, end;
   double cpu_time_used;
   start = clock();
      
   do{
     k+=1;

    

     MultMatVec(Sl,Al,Pl,P);
      
    


     double PDr=0.0,PDS=0.0;
     Dot(Pl,rl,&PDr);
     Dot(Pl,Sl,&PDS);
    
     Alpha=PDr/PDS;
         
     MultVecPlus(Xl,Xl,Alpha,Pl);
     MultVecPlus(rl,rl,-Alpha,Sl);

     MultMatVec(Zl,Al,rl,r);
  
     double PDZ=0.0;
     Dot(Pl,Zl,&PDZ);
     Beta=-PDZ/PDS;

     MultVecPlus(Pl,rl,Beta,Pl);

     double res=0.0;
     Dot(rl,rl,&res);
     res=sqrt(res);
     
     if(rank==0){
       //  printf("K: %d, res:%le , Alpha: %le , Beta : %le \n",k,res,Alpha,Beta);
       printf("%d, %le\n",k,res);
     }
     
   } while (res>tol && k<kmax);

   if(rank==0)
     //  printf("Flops:%d, Data Size: %d \n",Flops,N*N+N);

   end = clock();
   cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

   if(rank==0)
   printf("CPUtime: %le  \n",cpu_time_used);



   free(rl);
   free(r);
   free(Pl);
   free(P);
   free(Al);
   free(Xl);
   free(Bl);

 
   MPI_Finalize();
   PetscFinalize();
   return 0;
} 
//-------------------------------------------------------------
void Get_local_dims(int N,int Ncpu,int rank,int* ms,int* me) {
  //Based on the row 
 *ms=(N/Ncpu)*(rank);
 *me=(N/Ncpu)*(rank+1)-1;
}  
//-------------------------------------------------------------
void Form_matrix(double** Al){
 
  double dx2=1./im,dy2=1./jm;
  int m,ml,i,j,n;
  
  for (m = ms; m < me+1; m++){
    j=(int)(m/(im+1)); // For bounday condition
    i=m-(j*(im+1));
    ml=m-ms;

    if(i==0 || j==0 || i==im || j==jm){
      n=m;
      Al[ml][n]=1.0;
    }else{
   
      n=m;
      Al[ml][n] =-2*((1/dx2)+(1/dy2));

      n=m+1;
      Al[ml][n]=1./dx2;
      
      n=m-1;
      Al[ml][n]=1./dx2;
    
      n=m+im+1;
      Al[ml][n]=1./dy2;
  
      n=m-(im+1);
      Al[ml][n]=1./dy2;
    }
  }//if not on boundary

}
//---------------------------------------------------------------
void Form_RHS(double* Bl){

  double XL=100.,XR=200.,XD=50.,XU=100.;
  int m,ml,i,j;

   for (m = ms; m < me+1; m++){
    j=(int)(m/(im+1)); 
    i=m-(j*(im+1));
    ml=m-ms;

    if(i==0){
      Bl[ml]=XL;
    }
    if(i==im){
      Bl[ml]=XR;
    }
    if(j==0){
      Bl[ml]=XD;
    }
    if(j==jm){
      Bl[ml]=XU;
    }
    if(i==0 && j==0){
      Bl[ml]=(XL+XD)/2;
    }
    if(i==0 && j==jm){
      Bl[ml]=(XL+XU)/2;
    }
    if(i==im && j==0){
      Bl[ml]=(XR+XD)/2;
    }
    if(i==im && j==jm){
      Bl[ml]=(XR+XU)/2;
    }
  }

}
//----------------------------------------------------------------
void Equal(double* Xl,double* Yl){
  int m,ml;
   for (m = ms; m < me+1; m++){
     ml=m-ms;
    Xl[ml]=Yl[ml];
  }
}
//----------------------------------------------------------------
void Dot(double* Xl,double* Yl,double* sum){
  int m,ml;
  double suml=0.0,sumg=0.0;
   for (m = ms; m < me+1; m++){
    ml=m-ms;
    suml+=Yl[ml]*Xl[ml];
   
  }

 MPI_Allreduce(&suml,&sumg,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

  *sum=sumg;
  
}

//----------------------------------------------------------------
  void MultVecPlus(double* Zl,double* Xl,double alpha,double* Yl){
    int m,ml;
     for (m = ms; m < me+1; m++){
       ml=m-ms;
      Zl[ml]=Xl[ml]+alpha*Yl[ml];
     }
 
  }
//----------------------------------------------------------------
void MultMatVec(double* Sl,double **Al,double* Pl,double* P) {
  int m,ml,n;
  ml=me-ms;
  // MPI_Barrier(comm);
  MPI_Allgather(Pl,ml+1, MPI_DOUBLE,P,ml+1, MPI_DOUBLE, comm); 
  for (m = ms; m < me+1; m++) {
    ml=m-ms;
      Sl[ml]= 0.0;
      for (n = 0; n < N; n++){
         Sl[ml] += Al[ml][n]*P[n];
	 //Flops+=2;
      }
   }
}
//----------------------------------------------------------------
