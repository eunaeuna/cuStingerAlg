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
#if 0 //queue	
	vertexQueue queueSrc;
	vertexQueue queueDst;
#else //array
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
/*
static __device__ void recomputeContribution(cuStinger* custing, vertexId_t src, vertexId_t dst, void* metadata){
        pageRankUpdate* pr = (pageRankUpdate*)metadata;
        length_t sizeSrc = custing->dVD->getUsed()[src];
        prType updateDiff = pr->damp*(pr->prevPR[src]/(sizeSrc+1));
        updateDiff = updateDiff - (pr->damp*(((float)(pr->prevPR[src])/sizeSrc)*(1/(sizeSrc+1))));
        atomicAdd(pr->currPR+src,pr->contri[dst]);
}
static __device__ void updateContributionsUndirected(cuStinger* custing, vertexId_t src, vertexId_t dst, void* metadata){
	streaming_page_rank   pageRankUpdate* pr = (pageRankUpdate*)metadata;
        length_t sizeSrc = custing->dVD->getUsed()[src];
        prType updateDiff = pr->damp*(pr->prevPR[src]/(sizeSrc+1));
        updateDiff = updateDiffhostDst,devSrc,elements*eleSize,cudaMemcpyDeviceToHost) - (pr->damp*(((float)(pr->prevPR[src])/sizeSrc)*(1/(sizeSrc+1))));
        atomicAdd(pr->currPR+src,pr->contri[dst]);
}
*/
static __device__ void recomputeContributionUndirected(cuStinger* custing, vertexId_t src, vertexId_t dst, void* metadata){
        pageRankUpdate* pr = (pageRankUpdate*)metadata;
//src
        length_t sizeDst = custing->dVD->getUsed()[dst];
        prType updateDiff = pr->damp*(pr->prevPR[dst]/(sizeDst+1));
        printf("\n---damp=%f, (prevPR[%d:dst]:%f)/(n:%d)",pr->damp,dst,(pr->prevPR[dst]),sizeDst+1);
        printf("\n---------------(pr->prevPR[%d]:%f)+=(updateDiff=%f)",src,pr->prevPR[src],updateDiff);
        //updateDiff += pr->prevPR[src]; //new = old + pr_ver_in_new_edge
        //atomicAdd(pr->currPR+src,updateDiff);
        printf("\n!!----------------------------------(pr->prevPR[%d]=%f), (pr->currPR[%d]=%f)",src,pr->prevPR[src],src,pr->currPR[src]);
        pr->currPR[src] = pr->prevPR[src] + updateDiff;
        //pr->prevPR[src]=pr->currPR[src]; //preserve old pr values for propagation
        printf("\n!!----------------------------------(pr->prevPR[%d]=%f), (pr->currPR[%d]=%f)",src,pr->prevPR[src],src,pr->currPR[src]);
}

static __device__ void updateContributionsUndirected(cuStinger* custing, vertexId_t src, vertexId_t dst, void* metadata){
	    pageRankUpdate* pr = (pageRankUpdate*)metadata;
        length_t sizeSrc = custing->dVD->getUsed()[src];

        printf("\n ++pr->prevPR[%d]:%f,pr->currPR[%d]:%f,sizeSrc+1:%d",src,pr->prevPR[src],src,pr->currPR[src],sizeSrc+1);
        
        prType updateDiff = pr->damp*((pr->currPR[src]/(sizeSrc+1))-(pr->prevPR[src]/sizeSrc));// = pr->damp*((pr->prevPR[dst]/(sizeSrc*sizeDst)) - (pr->prevPR[src]/(sizeSrc*(sizeSrc-1))));
        updateDiff += pr->prevPR[dst];

        printf("\n ++(propagation:[%d])---------------(pr->prevPR[%d]:%f)+=(updateDiff=%f),size:%d",src,dst,pr->prevPR[dst],updateDiff,sizeSrc+1);
#if 1
        atomicAdd(pr->currPR+dst,updateDiff);
        //pr->currPR[dst] = pr->prevPR[dst] + updateDiff;
        pr->prevPR[dst] = pr->currPR[dst];
#endif
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
