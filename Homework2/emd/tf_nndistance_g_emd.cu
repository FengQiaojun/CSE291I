#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#define EPSILON 0.01
//NUM_THREADS is equal to N must be 2^k where k<=10
#define NUM_THREADS 1024
//#define DEBUG 0

#include <cub/cub.cuh>




__global__
void constructadjmat(const int n,const  float *src, const float *tgt, float* adj, float* prices, int* src2tgt, int*tgt2src, float* average, int* d_offsets, int* d_offsets_end)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int idx = index; idx < n*n; idx += stride)
	{
		int i = idx/n;
		int j = idx-n*i;
		float x1=src[i*3];
		float y1=src[i*3+1];
		float z1=src[i*3+2];
		float x2=tgt[j*3];
		float y2=tgt[j*3+1];
		float z2=tgt[j*3+2];
		double d1=x1-x2;
		double d2=y1-y2;
		double d3=z1-z2;
		double d =d1*d1+d2*d2+d3*d3;
		adj[i*n+j]=d;
		atomicAdd(average,  d*1.0/(n*n));
	}
	for (int idx = index; idx < n; idx += stride)
	{
		prices[idx]=0;
		src2tgt[idx]=-1;
		tgt2src[idx]=-1;
		d_offsets[idx]=idx*n;
		d_offsets_end[idx]=idx*n+n;
	}

}


__global__
void  calcbidvalues(int n, int* src2tgt, float* adj, float* prices,  bool* complete, float* values, float* bids)
{
	int INDEX = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int idx = INDEX; idx < n*n; idx += stride)
	{
		int i=idx/n;
		int j= idx-i*n;
		bids[i*n+j]=-1;
		if(src2tgt[i]!=-1)
		{
			continue;
		}
		complete[0]=false;
		values[i*n+j]= -adj[i*n+j]-prices[j];
	}
}


__global__
void submitbids3(const int n, int* src2tgt, float* prices, float* bids, float best[], int index[], cub::KeyValuePair<int,float>* d_max, double eps)
{
	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	for(int i= idx; i < n; i+=stride)
	{
		if(src2tgt[i]==-1)
		{
			float best2 =d_max[i].value;
			bids[index[i]*n+i] = prices[index[i]]+best[i]-best2+eps;
		}
	}
}


__global__
void submitbids2(const int n,int* src2tgt, float best[], int index[], float* values, cub::KeyValuePair<int,float>* d_max)
{
	int stride = blockDim.x * gridDim.x;
	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	for(int i= idx; i < n; i+=stride)
	{
		if(src2tgt[i]==-1)
		{
			best[i]=d_max[i].value;
			index[i]=d_max[i].key;
			values[i*n+index[i]]=INT_MIN;
		}
	}
}

void submitbids(const int n, int* src2tgt, float* prices, float* bids,  float* values, double eps, void* d_temp_storage, size_t temp_storage_bytes, cub::KeyValuePair<int,float>* d_max, int* d_offsets, int* d_offsets_end, float* best, int* index)
{
	int count=0;
	int check[n];
	cub::DeviceSegmentedReduce::ArgMax(d_temp_storage, temp_storage_bytes, values, d_max,
    n, d_offsets, d_offsets_end);
	cudaDeviceSynchronize();
	int num_blocks=std::ceil((n)/NUM_THREADS);
	submitbids2<<<num_blocks, NUM_THREADS>>>(n, src2tgt, best, index, values, d_max);
	cub::DeviceSegmentedReduce::ArgMax(d_temp_storage, temp_storage_bytes, values, d_max,
    n, d_offsets, d_offsets_end);
	cudaDeviceSynchronize();
//	for(int i=0; i < n; i++)
//		std::cout<<prices[index[i]]<<" "<<best[i]<<" "<<d_max[i].value<<" "<<eps<<" "<<d_offsets[i]<<" "<<d_offsets_end[i]<<std::endl;
	submitbids3<<<num_blocks, NUM_THREADS>>>(n, src2tgt, prices, bids, best, index, d_max,eps);
}


__global__
void processbids2(const int n, float* bids, int* src2tgt, int* tgt2src, float* prices, float*adj, cub::KeyValuePair<int,float> *d_max, int* d_offsets, int* d_offsets_end, float* result, float* result2)
{
	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	for(int j= idx; j < n; j+=stride)
	{
		float best=d_max[j].value;
		int index=d_max[j].key;
		if( best != -1)
		{
			int prev=tgt2src[j];
			if(prev!=-1)
			{
				src2tgt[prev]=-1;
				d_offsets[prev]=prev*n;
				d_offsets_end[prev]=prev*n+n;
			}
			src2tgt[index]=j;
			result[index]=adj[index*n+j];
			d_offsets[index]=index*n;
			d_offsets_end[index]=index*n;
			tgt2src[j]=index;
			result2[j]=adj[index*n+j];
			prices[j]=bids[j*n+index];
		}
	}
}

void processbids(const int n, float* bids, int* src2tgt, int* tgt2src, float* prices, float*adj, void* d_temp_storage, size_t temp_storage_bytes, cub::KeyValuePair<int,float>* d_max, int* d_offsets, int* d_offsets_end, float* result, float* result2)
{
	cub::DeviceSegmentedReduce::ArgMax(d_temp_storage, temp_storage_bytes, bids, d_max,
    n, d_offsets, d_offsets+1);
	cudaDeviceSynchronize();
	int num_blocks=std::ceil((n)/NUM_THREADS);
	processbids2<<<num_blocks,NUM_THREADS>>>(n,bids,src2tgt,tgt2src,prices,adj, d_max, d_offsets, d_offsets_end, result, result2);
}



void NnEMDDistanceKernelLauncher(int b,const int N,const float * src,const float * tgt,float * result,int * src2tgt,float * result2,int * tgt2src){
	//N=NUM_THREADS;
	//float* src, *tgt;
	float *adj, *prices,*bids, *values;
	//int  *src2tgt, *tgt2src;
	bool* complete;
	float* average;
	float* best;
	int *index;
	int* d_offsets, *d_offsets_end;

	// Allocate Unified Memory – accessible from CPU or GPU
	//cudaMallocManaged(&src, 3*N*sizeof(float));
	//cudaMallocManaged(&tgt, 3*N*sizeof(float));
	cudaMallocManaged(&adj, N*N*sizeof(float));
	cudaMallocManaged(&prices, N*sizeof(float)); //init to 0
	cudaMallocManaged(&bids, N*N*sizeof(float)); 
	cudaMallocManaged(&values, N*N*sizeof(float));


	cudaMallocManaged(&complete, sizeof(bool)); 
	cudaMallocManaged(&average, sizeof(float)); 

	cudaMallocManaged(&d_offsets, (N+1)*sizeof(int)); 
	cudaMallocManaged(&d_offsets_end, N*sizeof(int)); 
	cudaMallocManaged(&best, N*sizeof(float)); 
	cudaMallocManaged(&index, N*sizeof(int)); 
	d_offsets[N]=N*N;
	void* d_temp_storage=NULL;
	size_t temp_storage_bytes = 0;
	cub::KeyValuePair<int,float>* d_max;//(float**) malloc(N*sizeof(float*))
	cudaMallocManaged(&d_max, N*sizeof(cub::KeyValuePair<int,float>));
	cub::DeviceSegmentedReduce::ArgMax(d_temp_storage, temp_storage_bytes, values, d_max,
			N, d_offsets, d_offsets_end);

	cudaMallocManaged(&d_temp_storage, temp_storage_bytes*N);






	#if DEBUG
	std::cout<<"SOURCE TGT: "<<std::endl;
	for(int i=0; i < N; i++)
	{
		int idx=i*3;
		printf("%f %f %f        %f %f %f\n",src[idx], src[idx+1], src[idx+2], tgt[idx], tgt[idx+1], tgt[idx+2]);
	}
	#endif
	int num_blocks=std::min(std::ceil((N*N)/NUM_THREADS), 65535.0);
	constructadjmat<<<num_blocks, NUM_THREADS>>>(N, src, tgt, adj, prices, src2tgt, tgt2src, average, d_offsets, d_offsets_end);
	#ifdef DEBUG
	cudaDeviceSynchronize();
	printf("\n  ");
	for(int i=0; i <8;i++)
		printf("%3d ",i);
	printf("\n");
	for(int i= 0; i < 8; i++)
	{
		printf("%3d ",i);
		for(int j=0; j < 8; j++)
		{
			printf("%03d ", (int)adj[i*N+j]);
		}
		std::cout<<std::endl;
	}
	#endif
	complete[0]=false;
	int iter=0;	
	while(1)
	{
		
		iter++;
		num_blocks=std::min(std::ceil((N*N)/NUM_THREADS), 65535.0);
		calcbidvalues<<<num_blocks, NUM_THREADS>>>(N, src2tgt, adj, prices, complete, values, bids);
		#ifdef DEBUG
		cudaDeviceSynchronize();
		printf("VALS: \n  ");
		for(int i=0; i <N;i++)
			printf("%4d ",i);
		printf("\n");
		for(int i= 0; i < N; i++)
		{
			printf("%4d ",i);
			for(int j=0; j < N; j++)
			{
				printf("%04d ", (int)values[i*N+j]);
			}
			std::cout<<std::endl;
		}
		#endif
		cudaDeviceSynchronize();
		if(complete[0])
			break;
		complete[0]=true;
		submitbids(N, src2tgt, prices, bids,  values, EPSILON*average[0]*pow(2,iter/200), d_temp_storage, temp_storage_bytes, d_max, d_offsets, d_offsets_end, best, index);
		#if DEBUG
		cudaDeviceSynchronize();
		printf("BIDS: \n  ");
		for(int i=0; i <N;i++)
			printf("%3d ",i);
		printf("\n");
		for(int i= 0; i < N; i++)
		{
			printf("%3d ",i);
			for(int j=0; j < N; j++)
			{
				printf("%03d ", (int)bids[i*N+j]);
			}
			std::cout<<std::endl;
		}
		#endif

		//find max bid for each project
		processbids(N, bids,  src2tgt, tgt2src, prices,  adj, d_temp_storage, temp_storage_bytes, d_max, d_offsets, d_offsets_end,result,result2);
		#if DEBUG
		cudaDeviceSynchronize();
		printf("Src2tgt: \n  ");
		for(int i=0; i <N;i++)
			printf("%3d ",src2tgt[i]);
		printf("\n");
		printf("Prices: \n  ");
		for(int i=0; i <N;i++)
			printf("%3d ", (int)prices[i]);
		printf("\n");
		char c;
		get(c);
		#endif
		
	}
	//std::cout<<"ITERS: "<<N<<" "<<iter<<std::endl;
	
	cudaFree(values);
	cudaFree(adj);
	cudaFree(prices);
	cudaFree(bids);
	cudaFree(complete);
	cudaFree(average);
	cudaFree(d_offsets);
	cudaFree(d_offsets_end);
	cudaFree(best);
	cudaFree(index);
	cudaFree(d_temp_storage);
	cudaFree(d_max);


}

#endif
