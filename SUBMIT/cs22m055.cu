#include <iostream>
#include <stdio.h>
#include <cuda.h>

#define max_N 100000
#define max_P 30
#define BLOCKSIZE 1024

using namespace std;

//*******************************************

// Write down the kernels here

//It calculate the number of requests for a facility of a center.
__global__ void req_per_facility(int *d_req_cen, int *d_req_fac, int *d_start_cen, int *d_req_per_fac, int size){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < size){
    int index = d_start_cen[d_req_cen[id]] + d_req_fac[id];
    atomicAdd(&d_req_per_fac[index],1);

  }
}


//Find number of success request per center.

__global__ void success_request(int *d_start_fac, int *d_fac_id_req, int *d_req_start, int *d_req_slots, int *d_capacity, int *d_success, int *d_req_cen, int size, int *total_succ){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < size){
    int time[24]; // It is storing the number less than capacity of request at a time.
    for(int i=0;i<24;i++){
      time[i]=0;
    }
    
    int i = d_start_fac[id];
    while(i<d_start_fac[id+1]){
      int st_time = d_req_start[d_fac_id_req[i]]-1; // starting time of request.
      int end_time = st_time +d_req_slots[d_fac_id_req[i]]; // end time of request.
      int flag = true;
      for(int j=st_time ; j< end_time; j++){
        if(d_capacity[id] <= time[j]){ //if time value of j is greater than capacity of request than set flag equal to false.
          flag=false;
        }
      }
      if(flag){
        for(int j=st_time; j< end_time ; j++){
          time[j]++;
        }
        atomicAdd(&d_success[d_req_cen[d_fac_id_req[i]]],1); //Increase the given success of center.
        atomicAdd(&total_succ[0],1);
      }
      i++;
    }
  }
}

//***********************************************


int main(int argc,char **argv)
{
	// variable declarations...
    int N,*centre,*facility,*capacity,*fac_ids, *succ_reqs, *tot_reqs;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &N ); // N is number of centres
	
    // Allocate memory on cpu
    centre=(int*)malloc(N * sizeof (int));  // Computer  centre numbers
    facility=(int*)malloc(N * sizeof (int));  // Number of facilities in each computer centre
    fac_ids=(int*)malloc(max_P * N  * sizeof (int));  // Facility room numbers of each computer centre
    capacity=(int*)malloc(max_P * N * sizeof (int));  // stores capacities of each facility for every computer centre 


    int success=0;  // total successful requests
    int fail = 0;   // total failed requests
    tot_reqs = (int *)malloc(N*sizeof(int));   // total requests for each centre
    succ_reqs = (int *)malloc(N*sizeof(int)); // total successful requests for each centre

    // Input the computer centres data
    int k1=0 , k2 = 0;
    for(int i=0;i<N;i++)
    {
      fscanf( inputfilepointer, "%d", &centre[i] );
      fscanf( inputfilepointer, "%d", &facility[i] );
      
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &fac_ids[k1] );
        k1++;
      }
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &capacity[k2]);
        k2++;     
      }
    }

    // variable declarations
    int *req_id, *req_cen, *req_fac, *req_start, *req_slots;   // Number of slots requested for every request
    
    // Allocate memory on CPU 
	int R;
	fscanf( inputfilepointer, "%d", &R); // Total requests
    req_id = (int *) malloc ( (R) * sizeof (int) );  // Request ids
    req_cen = (int *) malloc ( (R) * sizeof (int) );  // Requested computer centre
    req_fac = (int *) malloc ( (R) * sizeof (int) );  // Requested facility
    req_start = (int *) malloc ( (R) * sizeof (int) );  // Start slot of every request
    req_slots = (int *) malloc ( (R) * sizeof (int) );   // Number of slots requested for every request
    
    // Input the user request data
    for(int j = 0; j < R; j++)
    {
       fscanf( inputfilepointer, "%d", &req_id[j]);
       fscanf( inputfilepointer, "%d", &req_cen[j]);
       fscanf( inputfilepointer, "%d", &req_fac[j]);
       fscanf( inputfilepointer, "%d", &req_start[j]);
       fscanf( inputfilepointer, "%d", &req_slots[j]);
       tot_reqs[req_cen[j]]+=1;  
    }
		
    //------------------------cpu code-------------------------------//
    
    int *start_cen, *req_per_fac, *start_fac, *fac_id_req, *id_inc;
    int *d_capacity, *d_req_id, *d_req_cen, *d_req_fac, *d_req_per_fac, *d_start_cen, *d_start_fac, *d_id_inc, *d_fac_id_req, *d_success;
    int *d_req_start, *d_req_slots;

    
    start_cen = (int *)malloc(N*sizeof(int));//It is use to store the starting index of each center.

    //calculate the starting index of facility.
    int sum=0;
    for(int i=0;i<N;i++){
      start_cen[i]=sum;
      sum+=facility[i];
    }

    id_inc = (int *)malloc((sum+1)*sizeof(int));
    req_per_fac = (int *)malloc(sum*sizeof(int)); //it use to store the number of requests per facility.
    start_fac = (int *)malloc((sum+1)*sizeof(int)); //Store staring index of each facility request.
    fac_id_req = (int *)malloc(R*sizeof(int));  //Store the req_id of the facility request.

    //Memory allocation on GPU.
    cudaMalloc(&d_start_cen,N*sizeof(int));
    cudaMalloc(&d_start_fac,(sum+1)*sizeof(int));
    cudaMalloc(&d_req_per_fac,sum*sizeof(int));
    cudaMalloc(&d_capacity,sum*sizeof(int));
    cudaMalloc(&d_req_id, R*sizeof(int));
    cudaMalloc(&d_req_cen,R*sizeof(int));
    cudaMalloc(&d_req_fac,R*sizeof(int));
    cudaMalloc(&d_req_start,R*sizeof(int));
    cudaMalloc(&d_req_slots,R*sizeof(int));
    cudaMalloc(&d_start_fac,(sum + 1)*sizeof(int));
    cudaMalloc(&d_success,N*sizeof(int));
    cudaMalloc(&d_id_inc,(sum+1)*sizeof(int));
    
    //Set value to 0.
    cudaMemset(d_req_per_fac , 0 ,sum*sizeof(int));
    cudaMemset(d_success, 0 , N*sum*sizeof(int));

    //Memory copy from CPU to GPU.
    cudaMemcpy(d_start_cen, start_cen, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_capacity, capacity, sum*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_id, req_id, R*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_cen, req_cen, R*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_fac, req_fac, R*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_start, req_start, R*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_slots, req_slots, R*sizeof(int), cudaMemcpyHostToDevice);
    
    //Kernel call to calculate the no. of request per facility.
    int no_of_blocks = ceil((float)R/1024);
    req_per_facility<<< no_of_blocks, 1024>>>(d_req_cen, d_req_fac, d_start_cen, d_req_per_fac, R);
    cudaMemcpy(req_per_fac, d_req_per_fac, sum*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    //calculate the starting index of each facility.
    int fac_sum=0;
    for(int i=0; i<sum+1; i++){
      start_fac[i]=fac_sum;
      id_inc[i]=fac_sum;
      fac_sum += req_per_fac[i];
    }


    cudaMalloc(&d_fac_id_req, R*sizeof(int));
    cudaMemcpy(d_start_fac, start_fac, (sum+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_id_inc, start_fac, (sum+1)*sizeof(int), cudaMemcpyHostToDevice);
    
    //index maping to facility.
    for(int i=0; i<R; i++){
      int index = start_cen[req_cen[i]] + req_fac[i];
      fac_id_req[id_inc[index]] = req_id[i];
      id_inc[index]++;
    }


    cudaMemcpy(d_fac_id_req, fac_id_req , R*sizeof(int), cudaMemcpyHostToDevice);

  
    //*********************************
    // Call the kernels here

    //Calculate the success request at centers.
    int *total_succ;
    cudaMalloc(&total_succ,sizeof(int));
    cudaMemset(total_succ, 0, sizeof(int));
    no_of_blocks = ceil((float)sum/1024);
    success_request<<<no_of_blocks, 1024>>>(d_start_fac, d_fac_id_req, d_req_start, d_req_slots, d_capacity, d_success, d_req_cen, sum, total_succ);
    cudaMemcpy(succ_reqs, d_success, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&success, total_succ, sizeof(int), cudaMemcpyDeviceToHost);
    //********************************

    fail = R-success;



    // Output
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    fprintf( outputfilepointer, "%d %d\n", success, fail);
    for(int j = 0; j < N; j++)
    {
        fprintf( outputfilepointer, "%d %d\n", succ_reqs[j], tot_reqs[j]-succ_reqs[j]);
    }
    fclose( inputfilepointer );
    fclose( outputfilepointer );
    cudaDeviceSynchronize();
	return 0;
}