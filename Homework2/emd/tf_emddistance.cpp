#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
REGISTER_OP("EmdDistance")
	.Input("xyz1: float32")
	.Input("xyz2: float32")
	.Output("dist1: float32")
	.Output("idx1: int32")
	.Output("idx2: int32");
REGISTER_OP("EmdDistanceGrad")
	.Input("xyz1: float32")
	.Input("xyz2: float32")
	.Input("grad_dist1: float32")
	.Input("idx1: int32")
	.Input("idx2: int32")
	.Output("grad_xyz1: float32")
	.Output("grad_xyz2: float32");
using namespace tensorflow;

#define UPPERBOUND  1000
//maximum number of points supported
#define EPSILON 1


static void emdsearch(int b,int n,const float * xyz1,const float * xyz2,float * dist,int * idx1, int* idx2){
    for(int B=0; B < b; B++)
    {
        const float* src=xyz1+B*n*3;
        const float* tgt=xyz2+B*n*3;
        float* res = dist+B*n;
        int* src2tgt = idx1+B*n;
        int* tgt2src = idx2+B*n;
        //bipartite(src, tgt, n, res, src2tgt, tgt2src);
        double adj[UPPERBOUND][UPPERBOUND];
        double prices[UPPERBOUND];
        double bids[UPPERBOUND];
        int bidtargets[UPPERBOUND];
        int bidsreceived[UPPERBOUND];
        //int src2tgt[UPPERBOUND];
        //int tgt2src[UPPERBOUND];
        for(int i=0; i < n; i++)
        {
            float x1=src[i*3];
            float y1=src[i*3+1];
            float z1=src[i*3+2];
            for(int j=0; j < n; j++)
            {
                float x2=tgt[j*3];
                float y2=tgt[j*3+1];
                float z2=tgt[j*3+2];
                double d1=x1-x2;
                double d2=y1-y2;
                double d3=z1-z2;
                adj[i][j]=d1*d1+d2*d2+d3*d3;

            }
            prices[i]=0;
            src2tgt[i]=-1;
            tgt2src[i]=-1;
        }


        bool complete=false;
        int iter=0;

        while(!complete)
        {
            iter++;
            //bid
            complete=true;
            for(int i=0; i < n ; i++)
            {
                if(src2tgt[i]!=-1)
                    continue;
                complete=false;
                double best=INT_MIN;
                double secondbest=INT_MIN;
                int index=-1;
                for(int j=0; j < n; j++)
                {
                    //want to maxmimize value
                    double value = -adj[i][j]-prices[j];
                    if(value>=best)
                    {
                        secondbest=best;
                        best=value;
                        index=j;
                    }
                }
                bids[i] = prices[index]+best-secondbest+EPSILON;
                bidtargets[i]= index;
            }

            for(int j=0;j<n;j++)
            {
                bidsreceived[j]=-1;
            }

            //assign
            //find max for each object
            for(int i=0; i < n; i++)
            {
                int index=bidtargets[i];
                double bid=bids[i];
                int cur= bidsreceived[index];
                if(cur==-1 || bids[cur]<bid)
                    bidsreceived[index]=i;
            }


            for(int j=0; j <n ;j++)
            {
                int cur=bidsreceived[j];	
                if(cur==-1)
                    continue;
                int prev=tgt2src[j];
                src2tgt[prev]=-1;
                src2tgt[cur]=j;
                tgt2src[j]=cur;
                res[cur]=adj[cur][j];
                prices[j]=bids[cur];
            }
        }
    std::cout<<"Iters: "<<iter<<std::endl;
    }

}

class EmdDistanceOp : public OpKernel{
    public:
        explicit EmdDistanceOp(OpKernelConstruction* context):OpKernel(context){}
        void Compute(OpKernelContext * context)override{
            const Tensor& xyz1_tensor=context->input(0);
            const Tensor& xyz2_tensor=context->input(1);
			OP_REQUIRES(context,xyz1_tensor.dims()==3,errors::InvalidArgument("EmdDistance requires xyz1 be of shape (batch,#points,3)"));
			OP_REQUIRES(context,xyz1_tensor.shape().dim_size(2)==3,errors::InvalidArgument("EmdDistance only accepts 3d point set xyz1"));
			int b=xyz1_tensor.shape().dim_size(0);
			int n=xyz1_tensor.shape().dim_size(1);
			OP_REQUIRES(context,xyz2_tensor.dims()==3,errors::InvalidArgument("EmdDistance requires xyz2 be of shape (batch,#points,3)"));
			OP_REQUIRES(context,xyz2_tensor.shape().dim_size(2)==3,errors::InvalidArgument("EmdDistance only accepts 3d point set xyz2"));
			OP_REQUIRES(context,xyz2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("EmdDistance expects xyz1 and xyz2 have same batch size"));
			OP_REQUIRES(context,xyz2_tensor.shape().dim_size(1)==n,errors::InvalidArgument("EmdDistance expects xyz1 and xyz2 have same number of points"));
			OP_REQUIRES(context,n<=1000,errors::InvalidArgument("EmdDistance cannot process more than 4096 points (can be changed in tf_emddistance.cpp)"));
			auto xyz1_flat=xyz1_tensor.flat<float>();
			const float * xyz1=&xyz1_flat(0);
			auto xyz2_flat=xyz2_tensor.flat<float>();
			const float * xyz2=&xyz2_flat(0);
			Tensor * dist1_tensor=NULL;
			Tensor * idx1_tensor=NULL;
			Tensor * idx2_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n},&dist1_tensor));
			OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,n},&idx1_tensor));
			auto dist1_flat=dist1_tensor->flat<float>();
			auto idx1_flat=idx1_tensor->flat<int>();
			OP_REQUIRES_OK(context,context->allocate_output(2,TensorShape{b,n},&idx2_tensor));
			auto idx2_flat=idx2_tensor->flat<int>();
			float * dist1=&(dist1_flat(0));
			int * idx1=&(idx1_flat(0));
			int * idx2=&(idx2_flat(0));
			emdsearch(b,n,xyz1,xyz2,dist1,idx1, idx2);
		}
};
REGISTER_KERNEL_BUILDER(Name("EmdDistance").Device(DEVICE_CPU), EmdDistanceOp);
class EmdDistanceGradOp : public OpKernel{
	public:
		explicit EmdDistanceGradOp(OpKernelConstruction* context):OpKernel(context){}
		void Compute(OpKernelContext * context)override{
			const Tensor& xyz1_tensor=context->input(0);
			const Tensor& xyz2_tensor=context->input(1);
			const Tensor& grad_dist1_tensor=context->input(2);
			const Tensor& idx1_tensor=context->input(3);
			const Tensor& idx2_tensor=context->input(4);
			OP_REQUIRES(context,xyz1_tensor.dims()==3,errors::InvalidArgument("EmdDistanceGrad requires xyz1 be of shape (batch,#points,3)"));
			OP_REQUIRES(context,xyz1_tensor.shape().dim_size(2)==3,errors::InvalidArgument("EmdDistanceGrad only accepts 3d point set xyz1"));
			int b=xyz1_tensor.shape().dim_size(0);
			int n=xyz1_tensor.shape().dim_size(1);
			OP_REQUIRES(context,xyz2_tensor.dims()==3,errors::InvalidArgument("EmdDistanceGrad requires xyz2 be of shape (batch,#points,3)"));
			OP_REQUIRES(context,xyz2_tensor.shape().dim_size(2)==3,errors::InvalidArgument("EmdDistanceGrad only accepts 3d point set xyz2"));
			OP_REQUIRES(context,xyz2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("EmdDistanceGrad expects xyz1 and xyz2 have same batch size"));
			OP_REQUIRES(context,xyz2_tensor.shape().dim_size(1)==n,errors::InvalidArgument("EmdDistanceGrad expects xyz1 and xyz2 have same batch size"));
			OP_REQUIRES(context,grad_dist1_tensor.shape()==(TensorShape{b,n}),errors::InvalidArgument("EmdDistanceGrad requires grad_dist1 be of shape(batch,#points)"));
			OP_REQUIRES(context,idx1_tensor.shape()==(TensorShape{b,n}),errors::InvalidArgument("EmdDistanceGrad requires idx1 be of shape(batch,#points)"));
			OP_REQUIRES(context,idx2_tensor.shape()==(TensorShape{b,n}),errors::InvalidArgument("EmdDistanceGrad requires idx2 be of shape(batch,#points)"));
			auto xyz1_flat=xyz1_tensor.flat<float>();
			const float * xyz1=&xyz1_flat(0);
			auto xyz2_flat=xyz2_tensor.flat<float>();
			const float * xyz2=&xyz2_flat(0);
			auto idx1_flat=idx1_tensor.flat<int>();
			const int * idx1=&idx1_flat(0);
			auto idx2_flat=idx2_tensor.flat<int>();
			const int * idx2=&idx2_flat(0);
			auto grad_dist1_flat=grad_dist1_tensor.flat<float>();
			const float * grad_dist1=&grad_dist1_flat(0);
			Tensor * grad_xyz1_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n,3},&grad_xyz1_tensor));
			Tensor * grad_xyz2_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,n,3},&grad_xyz2_tensor));
			auto grad_xyz1_flat=grad_xyz1_tensor->flat<float>();
			float * grad_xyz1=&grad_xyz1_flat(0);
			auto grad_xyz2_flat=grad_xyz2_tensor->flat<float>();
			float * grad_xyz2=&grad_xyz2_flat(0);
			for (int i=0;i<b*n*3;i++)
            {
				grad_xyz1[i]=0;
				grad_xyz2[i]=0;
            }
			for (int i=0;i<b;i++){
				for (int j=0;j<n;j++){
					float x1=xyz1[(i*n+j)*3+0];
					float y1=xyz1[(i*n+j)*3+1];
					float z1=xyz1[(i*n+j)*3+2];
					int j2=idx1[i*n+j];
					float x2=xyz2[(i*n+j2)*3+0];
					float y2=xyz2[(i*n+j2)*3+1];
					float z2=xyz2[(i*n+j2)*3+2];
					float g=grad_dist1[i*n+j]*2;
					grad_xyz1[(i*n+j)*3+0]+=g*(x1-x2);
					grad_xyz1[(i*n+j)*3+1]+=g*(y1-y2);
					grad_xyz1[(i*n+j)*3+2]+=g*(z1-z2);
					grad_xyz2[(i*n+j2)*3+0]-=(g*(x1-x2));
					grad_xyz2[(i*n+j2)*3+1]-=(g*(y1-y2));
					grad_xyz2[(i*n+j2)*3+2]-=(g*(z1-z2));
				}
			}
		}
};
REGISTER_KERNEL_BUILDER(Name("EmdDistanceGrad").Device(DEVICE_CPU), EmdDistanceGradOp);

