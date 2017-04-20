#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include <iomanip> 

#include <cub.cuh>
#include <util_allocator.cuh>

#include <device/device_reduce.cuh>
#include <kernel_mergesort.hxx>


#include "update.hpp"
#include "cuStinger.hpp"

#include "operators.cuh"

#include "streaming_page_rank/pr.cuh"

#define PR_UPDATE 1

using namespace cub;
using namespace mgpu;

CachingDeviceAllocator  g_u_allocator(true);  // Caching allocator for device memory

namespace cuStingerAlgs {

void StreamingPageRank::Init(cuStinger& custing){
	hostPRData.nv = custing.nv;
	hostPRData.prevPR  = (prType*) allocDeviceArray(hostPRData.nv+1, sizeof(prType));
	hostPRData.currPR  = (prType*) allocDeviceArray(hostPRData.nv+1, sizeof(prType));
	hostPRData.absDiff = (prType*) allocDeviceArray(hostPRData.nv+1, sizeof(prType));
	hostPRData.contri = (prType*) allocDeviceArray(hostPRData.nv+1, sizeof(prType));

	hostPRData.reductionOut = (prType*) allocDeviceArray(1, sizeof(prType));
	// hostPRData.reduction=NULL;

	devicePRData = (pageRankUpdate*)allocDeviceArray(1, sizeof(pageRankUpdate));
	hostPRData.queue.Init(custing.nv);

	SyncDeviceWithHost();

	//cusLB = new cusLoadBalance(custing.nv); //ERROR!!
	cusLB = new cusLoadBalance(custing);

	Reset();
}

void StreamingPageRank::Reset(){
	hostPRData.iteration = 0;
	hostPRData.queue.resetQueue();
	SyncDeviceWithHost();
}


void StreamingPageRank::Release(){
	free(cusLB);	
	freeDeviceArray(devicePRData);
	freeDeviceArray(hostPRData.currPR);
	freeDeviceArray(hostPRData.prevPR);
	freeDeviceArray(hostPRData.absDiff);
	// freeDeviceArray(hostPRData.reduction);
	freeDeviceArray(hostPRData.reductionOut);
	freeDeviceArray(hostPRData.contri);
}

void StreamingPageRank::Run(cuStinger& custing){

	allVinG_TraverseVertices<StreamingPageRankOperator::init>(custing,devicePRData);
	hostPRData.iteration = 0;

	prType h_out = hostPRData.threshhold+1;

	while(hostPRData.iteration < hostPRData.iterationMax && h_out>hostPRData.threshhold){
		SyncDeviceWithHost();

		allVinA_TraverseVertices<StreamingPageRankOperator::resetCurr>(custing,devicePRData,*cusLB);
		allVinA_TraverseVertices<StreamingPageRankOperator::computeContribuitionPerVertex>(custing,devicePRData,*cusLB);
		allVinA_TraverseEdges_LB<StreamingPageRankOperator::addContribuitionsUndirected>(custing,devicePRData,*cusLB);
		allVinA_TraverseVertices<StreamingPageRankOperator::dampAndDiffAndCopy>(custing,devicePRData,*cusLB);

		allVinG_TraverseVertices<StreamingPageRankOperator::sum>(custing,devicePRData);
		SyncHostWithDevice();

		copyArrayDeviceToHost(hostPRData.reductionOut,&h_out, 1, sizeof(prType));
		// h_out=hostPRData.threshhold+1;
		// cout << "The number of elements : " << hostPRData.nv << endl;

		hostPRData.iteration++;
	}
}

#if PR_UPDATE
void StreamingPageRank::Run2(cuStinger& custing){

//	allVinG_TraverseVertices<StreamingPageRankOperator::init>(custing,devicePRData);
	hostPRData.iteration = 0;

	prType h_out = hostPRData.threshhold+1;

	while(hostPRData.iteration < hostPRData.iterationMax && h_out>hostPRData.threshhold){
		SyncDeviceWithHost();

		allVinA_TraverseVertices<StreamingPageRankOperator::resetCurr>(custing,devicePRData,*cusLB);
		allVinA_TraverseVertices<StreamingPageRankOperator::computeContribuitionPerVertex>(custing,devicePRData,*cusLB);
		allVinA_TraverseEdges_LB<StreamingPageRankOperator::addContribuitionsUndirected>(custing,devicePRData,*cusLB);
		// allVinA_TraverseEdges_LB<StreamingPageRankOperator::addContribuitions>(custing,devicePRData,*cusLB);
		allVinA_TraverseVertices<StreamingPageRankOperator::dampAndDiffAndCopy>(custing,devicePRData,*cusLB);
		allVinG_TraverseVertices<StreamingPageRankOperator::sum>(custing,devicePRData);
		SyncHostWithDevice();

		copyArrayDeviceToHost(hostPRData.reductionOut,&h_out, 1, sizeof(prType));
		// h_out=hostPRData.threshhold+1;
		// cout << "The number of elements : " << hostPRData.nv << endl;

		hostPRData.iteration++;
	}
}

void StreamingPageRank::UpdateDiff(cuStinger& custing, BatchUpdateData& bud) {
#if 0 //in host memory
    length_t batchsize = *(bud.getBatchSize());
    vertexId_t *edgeSrc = bud.getSrc();
    vertexId_t *edgeDst = bud.getDst();
    
	    prType* h_currPR = (prType*)allocHostArray(hostPRData.nv, sizeof(prType));
        copyArrayDeviceToHost(hostPRData.currPR,h_currPR,hostPRData.nv, sizeof(prType));

        printf("----------------------------------------------\n");
        for(int32_t i=0; i<batchsize; i++){
               //undirected graph
               cuStinger::cusVertexData* cushVD = custing.getHostVertexData();
               length_t sizeSrc = cushVD->used[edgeSrc[i]];
               length_t sizeDst = cushVD->used[edgeDst[i]];
               printf("sizeSrc[%d]=%d,\t sizeDst[%d]=%d\t", edgeSrc[i], sizeSrc, edgeDst[i], sizeDst);
               printf("h_currPRSrc[%d]=%e,\t h_currPRDst[%d]=%e\n", edgeSrc[i], h_currPR[edgeSrc[i]], edgeDst[i], h_currPR[edgeDst[i]]);

               //1. vertice update
               prType updatePRDiffVtxSrc = (hostPRData.damp)*(h_currPR[edgeDst[i]]/(sizeDst*(sizeDst+1)));
               prType updatePRDiffVtxDst = (hostPRData.damp)*(h_currPR[edgeSrc[i]]/(sizeSrc*(sizeSrc+1)));
               h_currPR[edgeSrc[i]] -= updatePRDiffVtxSrc;
               h_currPR[edgeDst[i]] -= updatePRDiffVtxDst;
               printf("h_currPRSrc[%d]=%e,\t h_currPRDst[%d]=%e,\t updatePRDiffSrc=%e,\t updatePRDiffDst=%e\n", edgeSrc[i], h_currPR[edgeSrc[i]], edgeDst[i], h_currPR[edgeDst[i]], updatePRDiffVtxSrc, updatePRDiffVtxDst);
               dst
               prType updatePRDiffPropSrc = (hostPRData.damp)*(h_currPR[edgeDst[i]]/(sizeDst+1));
               prType updatePRDiffPropDst = (hostPRData.damp)*(hostPRData.queueh_currPR[edgeSrc[i]]/(sizeSrc+1));

               //2. connected vertices
               //ERROR!!!! no adj information in host memory
               vertexId_t* adjSrc = cushVD->adj[edgeSrc[i]]->dst;
               vertexId_t* adjDst = cushVD->adj[edgeDst[i]]->dst;

               for(length_t e=0; e<sizeSrc; e++){
                 h_currPR[adjSrc[e]] += updatePRDiffPropSrc;
                 printf("upDEBUG_ONdatePRSrc[%d]=%e,\t updatePRDiffPropSrc=%e\n", edgeSrc[i], h_currPR[adjSrc[e]], updatePRDiffPropSrc);
               }

               for(length_t e=0; e<sizeDst; e++){
                 h_currPR[adjSrc[e]] += updatePRDiffPropSrc;
                 printf("updatePRDst[%d]=%e,\t updatePRDiffPropDst=%e\n", edgeDst[i], h_currPR[adjDst[e]], updatePRDiffPropDst);
               }
       }
#else //in device memory
        length_t batchsize = *(bud.getBatchSize());
        vertexId_t *edgeSrc = bud.getSrc();
        vertexId_t *edgeDst = bud.getDst();
        
	    hostPRData.vArraySrc = (vertexId_t*)allocDeviceArray(hostPRData.nv, sizeof(vertexId_t));
	    hostPRData.vArrayDst = (vertexId_t*)allocDeviceArray(hostPRData.nv, sizeof(vertexId_t));
       
        copyArrayHostToDevice(edgeSrc,hostPRData.vArraySrc,batchsize,sizeof(vertexId_t));
        copyArrayHostToDevice(edgeDst,hostPRData.vArrayDst,batchsize,sizeof(vertexId_t));

#if DEBUG_ON
        printf("\n");
        for(length_t i=0; i<batchsize; i++) {
         	printf("edgeSrc[%d]=%d, edgeDst[%d]=%d\n",i,edgeSrc[i],i,edgeDst[i]);
         	//printf("vArraySrc[%d]=%d, vArrayDst[%d]=%d\n",i,hostPRData.vArraySrc[i],i,hostPRData.vArrayDst[i]);
         }
#endif
        
        hostPRData.iteration = 0;
        hostPRData.iterationMax = 1; //added
        prType h_out = hostPRData.threshhold+1;

        printf("\n--------------- recommpute-----------");       
        allVinA_TraverseOneEdge<StreamingPageRankOperator::recomputeContributionUndirected>(custing,devicePRData,
        		hostPRData.vArraySrc,hostPRData.vArrayDst,batchsize);
        printf("\n--------------- update contri---------------\n");
              
        for(length_t i=0; i<batchsize; i++) {
        	hostPRData.queue.enqueueFromHost(edgeSrc[i]);
        }    
        
        length_t prevEnd = batchsize;
        
        SyncDeviceWithHost(); //added for threashold and iteration count

#if DEBUG_ON      
        cout << "11111   hostPRData.queue.getActiveQueueSize():" <<hostPRData.queue.getActiveQueueSize()<< endl;
        cout << "hostPRData.iteration < hostPRData.iterationMax " << hostPRData.iteration << " "<< hostPRData.iterationMax << endl;
        cout << "h_out>hostPRData.threshhold?" << h_out << " " << hostPRData.threshhold << endl; 
#endif
        while((hostPRData.queue.getActiveQueueSize())>0 
        		&& (hostPRData.iteration < hostPRData.iterationMax)
        		&& (h_out>hostPRData.threshhold)){
        	cout << "\n" << "****hostPRData.queue.getActiveQueueSize()=" << hostPRData.queue.getActiveQueueSize() << endl;
            SyncDeviceWithHost();

        	allVinA_TraverseEdges_LB<StreamingPageRankOperator::updateContributionsUndirected>(custing,devicePRData,
            		*cusLB,hostPRData.queue,batchsize);

            hostPRData.queue.setQueueCurr(prevEnd);
    		prevEnd = hostPRData.queue.getQueueEnd();
    		
    		allVinA_TraverseVertices<StreamingPageRankOperator::updateDiffAndCopy>(custing,devicePRData,*cusLB);
    		allVinG_TraverseVertices<StreamingPageRankOperator::updateSum>(custing,devicePRData);
    		
            SyncHostWithDevice();
    		
    		copyArrayDeviceToHost(hostPRData.reductionOut,&h_out, 1, sizeof(prType));
    		hostPRData.iteration++;
#if DEBUG_ON    		
    	    cout << "22222   hostPRData.queue.getActiveQueueSize():" <<hostPRData.queue.getActiveQueueSize()<<endl;
    	    cout << "hostPRData.iteration < hostPRData.iterationMax " << hostPRData.iteration << " "<< hostPRData.iterationMax << endl;
    	    cout << "h_out>hostPRData.threshhold?" << h_out << " " << hostPRData.threshhold <<endl; 
#endif    	    
        }
#if DEBUG_ON        
        cout << "33333    hostPRData.queue.getActiveQueueSize():" <<hostPRData.queue.getActiveQueueSize()<<endl;
        cout << "hostPRData.iteration < hostPRData.iterationMax " << hostPRData.iteration << " "<< hostPRData.iterationMax << endl;
        cout << "h_out>hostPRData.threshhold?" << h_out << " " << hostPRData.threshhold <<endl;       
#endif 
#endif //end of host vs. device memory
}
#endif //end of PR_UPDATE


void StreamingPageRank::setInputParameters(length_t prmIterationMax, prType prmThreshhold,prType prmDamp){
	hostPRData.iterationMax=prmIterationMax;
	hostPRData.threshhold=prmThreshhold;
	hostPRData.damp=prmDamp;
	hostPRData.normalizedDamp=(1-hostPRData.damp)/float(hostPRData.nv);
	SyncDeviceWithHost();
}

length_t StreamingPageRank::getIterationCount(){
	return hostPRData.iteration;
}

#if PR_UPDATE
int fnum = 0;
#endif

void StreamingPageRank::printRankings(cuStinger& custing){
  
	prType* d_scores = (prType*)allocDeviceArray(hostPRData.nv, sizeof(prType));
	vertexId_t* d_ids = (vertexId_t*)allocDeviceArray(hostPRData.nv, sizeof(vertexId_t));

	copyArrayDeviceToDevice(hostPRData.currPR, d_scores,hostPRData.nv, sizeof(prType));

	allVinG_TraverseVertices<StreamingPageRankOperator::setIds>(custing,d_ids);

#if PR_UPDATE
	prType* h_currPr = (prType*)allocHostArray(hostPRData.nv, sizeof(prType));
	copyArrayDeviceToHost(hostPRData.currPR,h_currPr,hostPRData.nv, sizeof(prType));
	
    char nbuff[100];
    sprintf(nbuff, "pr_values_%d.txt", fnum++);
    FILE *fp_npr = fopen(nbuff, "w");
    for (uint64_t v=0; v<hostPRData.nv; v++)
    {
        fprintf(fp_npr,"%d %e\n",v,h_currPr[v]);
    }
   fclose(fp_npr);
#endif //end of PR_UPDATE
	
	standard_context_t context(false);
	mergesort(d_scores,d_ids,hostPRData.nv,greater_t<float>(),context);

	prType* h_scores = (prType*)allocHostArray(hostPRData.nv, sizeof(prType));
	vertexId_t* h_ids    = (vertexId_t*)allocHostArray(hostPRData.nv, sizeof(vertexId_t));

	copyArrayDeviceToHost(d_scores,h_scores,hostPRData.nv, sizeof(prType));
	copyArrayDeviceToHost(d_ids,h_ids,hostPRData.nv, sizeof(vertexId_t));


    for(int v=0; v<10; v++){
            printf("Pr[%d]:= %.10f\n",h_ids[v],h_scores[v]);
    }

#if PR_UPDATE
    char buff[100];
    sprintf(buff, "pr_values_sorted_%d.txt", fnum++);
    FILE *fp_pr = fopen(buff, "w");
    for (uint64_t v=0; v<hostPRData.nv; v++)
    {
        fprintf(fp_pr,"%d %d %e\n",v,h_ids[v],h_scores[v]);
    }
   fclose(fp_pr);
#endif //end of PR_UPDATE

//DO NOT REMOVE currPR FOR UPDATE
//	allVinG_TraverseVertices<StreamingPageRankOperator::resetCurr>(custing,devicePRData);
	allVinG_TraverseVertices<StreamingPageRankOperator::sumPr>(custing,devicePRData);

// SyncHostWithDevice();
	prType h_out;

	copyArrayDeviceToHost(hostPRData.reductionOut,&h_out, 1, sizeof(prType));
	cout << "                     " << setprecision(9) << h_out << endl;

	freeDeviceArray(d_scores);
	freeDeviceArray(d_ids);
	freeHostArray(h_scores);
	freeHostArray(h_ids);
}

}// cuStingerAlgs namespace
