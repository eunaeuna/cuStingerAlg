#pragma once

#include "algs.cuh"

#define PR_UPDATE 1

#if PR_UPDATE
#include "update.hpp"
#endif

namespace cuStingerAlgs {

typedef float prType;
class pageRankUpdate{
public:
	prType* prevPR;
	prType* currPR;
	prType* absDiff;
	// void* reduction;
	prType* reductionOut;
	prType* contri;

	length_t iteration;
	length_t iterationMax;
	length_t nv;
	prType threshhold;
	prType damp;
	prType normalizedDamp;
#if 1 //queue	
	prType epsilon; //determinant for enqueuing 
	vertexQueue queue;
	vertexQueue queueDlt; //delta queue
	length_t *visited;
    prType* delta;
//#else //array
	vertexId_t* vArraySrc;
	vertexId_t* vArrayDst;
#endif
	
};

// Label propogation is based on the values from the previous iteration.
class StreamingPageRank:public StaticAlgorithm{
public:
	virtual void Init(cuStinger& custing);
	virtual void Reset();
	virtual void Run(cuStinger& custing);
	virtual void Release();
#if PR_UPDATE	
	void UpdateDiff(cuStinger& custing, BatchUpdateData& bud);
	void Run2(cuStinger& custing);
#endif
	void SyncHostWithDevice(){
		copyArrayDeviceToHost(devicePRData,&hostPRData,1, sizeof(pageRankUpdate));
	}
	void SyncDeviceWithHost(){
		copyArrayHostToDevice(&hostPRData,devicePRData,1, sizeof(pageRankUpdate));
	}
	void setInputParameters(length_t iterationMax = 20, prType threshhold = 0.001 ,prType damp=0.85);

	length_t getIterationCount();

	// User is responsible for de-allocating memory.
	prType* getPageRankScoresHost(){
		prType* hostArr = (prType*)allocHostArray(hostPRData.nv, sizeof(prType));
		copyArrayDeviceToHost(hostPRData.currPR, hostArr, hostPRData.nv, sizeof(prType) );
		return hostArr;
	}

	// User sends pre-allocated array.
	void getPageRankScoresHost(vertexId_t* hostArr){
		copyArrayDeviceToHost(hostPRData.currPR, hostArr, hostPRData.nv, sizeof(prType) );
	}

	void printRankings(cuStinger& custing);

protected: 
	pageRankUpdate hostPRData, *devicePRData;
	length_t reductionBytes;
private: 
	cusLoadBalance* cusLB;	
};


class StreamingPageRankOperator{
public:
static __device__ void init(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankUpdate* pr = (pageRankUpdate*)metadata;
	pr->absDiff[src]=pr->currPR[src]=0.0;
	pr->prevPR[src]=1/float(pr->nv);
	// printf("%f, ", pr->prevPR[src]);
	*(pr->reductionOut)=0;
}

static __device__ void resetCurr(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankUpdate* pr = (pageRankUpdate*)metadata;
	pr->currPR[src]=0.0;
	*(pr->reductionOut)=0;
}

static __device__ void computeContribuitionPerVertex(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankUpdate* pr = (pageRankUpdate*)metadata;
	length_t sizeSrc = custing->dVD->getUsed()[src];
	if(sizeSrc==0)
		pr->contri[src]=0.0;
	else
		pr->contri[src]=pr->prevPR[src]/sizeSrc;
}


static __device__ void addContribuitions(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	pageRankUpdate* pr = (pageRankUpdate*)metadata;
	atomicAdd(pr->currPR+dst,pr->contri[src]);
}

static __device__ void addContribuitionsUndirected(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	pageRankUpdate* pr = (pageRankUpdate*)metadata;
	atomicAdd(pr->currPR+src,pr->contri[dst]);

}

static __device__ void dampAndDiffAndCopy(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankUpdate* pr = (pageRankUpdate*)metadata;
	// pr->currPR[src]=(1-pr->damp)/float(pr->nv)+pr->damp*pr->currPR[src];
	pr->currPR[src]=pr->normalizedDamp+pr->damp*pr->currPR[src];

	pr->absDiff[src]= fabsf(pr->currPR[src]-pr->prevPR[src]);
	pr->prevPR[src]=pr->currPR[src];
}

static __device__ void sum(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankUpdate* pr = (pageRankUpdate*)metadata;
	atomicAdd(pr->reductionOut,pr->absDiff[src] );
}

static __device__ void sumPr(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankUpdate* pr = (pageRankUpdate*)metadata;
	atomicAdd(pr->reductionOut,pr->prevPR[src] );
}

//update
#define PR_UPDATE 1
#if PR_UPDATE 
static __device__ void clearVisited(cuStinger* custing,vertexId_t src, void* metadata){
        pageRankUpdate* pr = (pageRankUpdate*)metadata;
        pr->visited[src] = 0;
}

static __device__ void markVisited(cuStinger* custing,vertexId_t src, void* metadata){
        pageRankUpdate* pr = (pageRankUpdate*)metadata;
        //pr->visited[src]++;
        atomicAdd(pr->visited+src,1);
}

static __device__ void recomputeContributionUndirected(cuStinger* custing, vertexId_t src, vertexId_t dst, void* metadata){
        pageRankUpdate* pr = (pageRankUpdate*)metadata;
//src
        length_t sizeDst = custing->dVD->getUsed()[dst];
#if 0
        prType updateDiff = pr->damp*(pr->prevPR[dst]/(sizeDst+1));
#else  //custingTest
        prType updateDiff = pr->damp*(pr->prevPR[dst]/(sizeDst));
#endif
//        printf("\n---damp=%f, (prevPR[%d:dst]:%e)/(n:%d)",pr->damp,dst,(pr->prevPR[dst]),sizeDst+1);
//        printf("\n---------------(pr->prevPR[%d]:%e)+=(updateDiff=%e)",src,pr->prevPR[src],updateDiff);
//        printf("\n!!----------------------------------(pr->prevPR[%d]=%e), (pr->currPR[%d]=%e)",src,pr->prevPR[src],src,pr->currPR[src]);
        atomicAdd(pr->currPR+src,updateDiff);
//        printf("\n!!----------------------------------(pr->prevPR[%d]=%e), (pr->currPR[%d]=%e)\n",src,pr->prevPR[src],src,pr->currPR[src]);
}

static __device__ void updateContributionsUndirected(cuStinger* custing, vertexId_t src, vertexId_t dst, void* metadata){
	    pageRankUpdate* pr = (pageRankUpdate*)metadata;
        length_t sizeSrc = custing->dVD->getUsed()[src];

//			printf("\n ++pr->prevPR[%d]:%e,pr->currPR[%d]:%e,sizeSrc+1:%d",src,pr->prevPR[src],src,pr->currPR[src],sizeSrc+1);
#if 1
            prType updateDiff = pr->damp*((pr->currPR[src]/(sizeSrc+1))-(pr->prevPR[src]/sizeSrc));
#else //custingTest
            prType updateDiff = pr->damp*((pr->currPR[src]/(sizeSrc))-(pr->prevPR[src]/(sizeSrc-1)));
#endif            
//			printf("\n ++(propagation:[%d])---------------(pr->prevPR[%d]:%e)+=(updateDiff=%e),size:%d",src,dst,pr->prevPR[dst],updateDiff,sizeSrc+1);
#if 1 //setting pr epsilon for update
//            printf("\n fabs(updateDiff) = %e, epsilon = %e\n",fabs(updateDiff), pr->epsilon);
            if (fabs(updateDiff) < pr->epsilon) {
            	pr->queueDlt.enqueue(src);
                atomicAdd(pr->delta+src,updateDiff);
        		printf("&& DeltaQ: enqueue(%d)!!\n",src);            	
                printf("++return!!\n");
            	return;
            }
#endif            
            atomicAdd(pr->currPR+dst,updateDiff);
			//pr->prevPR[dst] = pr->currPR[dst]; //do not update prev pr value
//			printf("\n ++(propagation:[%d])---------------(pr->currPR[%d]) = %e, size:%d\n",src,dst,pr->currPR[dst],sizeSrc+1);

			length_t sizeDst = custing->dVD->getUsed()[dst];

			
	        if ((sizeDst > 0) && (pr->visited[dst] == 0)) {
#if 0 
	        	pr->visited[dst]++;
	        	pr->queue.enqueue(dst);
				printf("&& enqueue(%d)!!\n",dst);
#else //prevent the race condition
	        	//CAS: old == compare ? val : old
	        	length_t temp = pr->visited[dst] + 1;
	        	length_t old = atomicCAS(pr->visited+dst,0,temp);
	        	if (old == 0) {
	        		pr->queue.enqueue(dst);
	        		printf("&& enqueue(%d)!!\n",dst);
	        	}
#endif	        	
            }
}

static __device__ void updateContributionsUndirected2(cuStinger* custing, vertexId_t src, void* metadata){
	    pageRankUpdate* pr = (pageRankUpdate*)metadata;
        
        if(pr->delta[src] > pr->epsilon)
        {
        		length_t temp = pr->visited[src] + 1;
	        	length_t old = atomicCAS(pr->visited+src,0,temp);
	        	if (old == 0) {
	        		pr->queue.enqueue(src);
	        		printf("&& enqueue(%d)!!\n",src);
	        	}
        }
}

static __device__ void updateDiffAndCopy(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankUpdate* pr = (pageRankUpdate*)metadata;
	//pr->currPR[src]=pr->normalizedDamp+pr->damp*pr->currPR[src];

	pr->absDiff[src]= fabsf(pr->currPR[src]-pr->prevPR[src]); //adsDiff --> delta[nv]
	pr->prevPR[src]=pr->currPR[src];
}

static __device__ void updateSum(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankUpdate* pr = (pageRankUpdate*)metadata;
	atomicAdd(pr->reductionOut,pr->absDiff[src] );
}
#endif

static __device__ void setIds(cuStinger* custing,vertexId_t src, void* metadata){
	vertexId_t* ids = (vertexId_t*)metadata;
	ids[src]=src;
}

static __device__ void print(cuStinger* custing,vertexId_t src, void* metadata){
	int* ids = (int*)metadata;
	if(threadIdx.x==0 & blockIdx.x==0){
		// printf("The wheels on the bus go round and round and round and round %d\n",*ids);
	}
}



// static __device__ void addDampening(cuStinger* custing,vertexId_t src, void* metadata){
// 	pageRankUpdate* pr = (pageRankUpdate*)metadata;
// 	pr->currPR[src]=(1-pr->damp)/float(pr->nv)+pr->damp*pr->currPR[src];
// }

// static __device__ void absDiff(cuStinger* custing,vertexId_t src, void* metadata){
// 	pageRankUpdate* pr = (pageRankUpdate*)metadata;
// 	pr->absDiff[src]= abs(pr->currPR[src]-pr->prevPR[src]);
// }

// static __device__ void prevEqualCurr(cuStinger* custing,vertexId_t src, void* metadata){
// 	pageRankUpdate* pr = (pageRankUpdate*)metadata;
// 	pr->prevPR[src]=pr->currPR[src];
// }



};



} // cuStingerAlgs namespace
