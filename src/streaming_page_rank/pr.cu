

	
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

#if 1 //queue
	hostPRData.queue.Init(custing.nv);
#else
    //allocate memory for array in UpdateDiff	
#endif	
	SyncDeviceWithHost();

	//cusLB = new cusLoadBalance(custing,false,true);
	//cusLB = new cusLoadBalance(custing.nv); //ERROR!!
	cusLB = new cusLoadBalance(custing);


	Reset();
}

void StreamingPageRank::Reset(){
	hostPRData.iteration = 0;
#if 1 //queue	
	hostPRData.queue.resetQueue();
#else
	//array
#endif	
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

#if 0 //queue	
#else
	//release array in UpdateDiff after using them. why? update is not called, they might not be allocated 
#endif
	
}

void StreamingPageRank::Run(cuStinger& custing){
	// cusLoadBalance cusLB(custing);
	// cusLoadBalance cusLB(custing,true,false);
//	cusLoadBalance cusLB(custing,false,true);
//	cout << "The number of non zeros is : " << cusLB.currArrayLen << endl;

	allVinG_TraverseVertices<StreamingPageRankOperator::init>(custing,devicePRData);
	hostPRData.iteration = 0;

	prType h_out = hostPRData.threshhold+1;

	while(hostPRData.iteration < hostPRData.iterationMax && h_out>hostPRData.threshhold){
		SyncDeviceWithHost();

		allVinA_TraverseVertices<StreamingPageRankOperator::resetCurr>(custing,devicePRData,*cusLB);
		allVinA_TraverseVertices<StreamingPageRankOperator::computeContribuitionPerVertex>(custing,devicePRData,*cusLB);
		allVinA_TraverseEdges_LB<StreamingPageRankOperator::addContribuitionsUndirected>(custing,devicePRData,*cusLB);
		// allVinA_TraverseEdges_LB<StreamingPageRankOperator::addContribuitions>(custing,devicePRData,*cusLB);
		allVinA_TraverseVertices<StreamingPageRankOperator::dampAndDiffAndCopy>(custing,devicePRData,*cusLB);

		// allVinG_TraverseVertices<StreamingPageRankOperator::resetCurr>(custing,devicePRData);
		// allVinG_TraverseVertices<StreamingPageRankOperator::computeContribuitionPerVertex>(custing,devicePRData);
		// allVinA_TraverseEdges_LB<StreamingPageRankOperator::addContribuitionsUndirected>(custing,devicePRData,cusLB);
		// allVinG_TraverseVertices<StreamingPageRankOperator::dampAndDiffAndCopy>(custing,devicePRData);

		// copyArrayDeviceToDevice(hostPRData.currPR,hostPRData.prevPR, hostPRData.nv,sizeof(prType));

		// allVinG_TraverseVertices<StreamingPageRankOperator::print>(custing,d_out);

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
	// cusLoadBalance cusLB(custing);
	// cusLoadBalance cusLB(custing,true,false);
//	cusLoadBalance cusLB(custing,false,true);
//	cout << "The number of non zeros is : " << cusLB.currArrayLen << endl;

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

        //array test
//	    hostPRData.vArraySrc = (vertexId_t*)allocHostArray(hostPRData.nv, sizeof(vertexId_t));
//	    hostPRData.vArrayDst = (vertexId_t*)allocHostArray(hostPRData.nv, sizeof(vertexId_t));
//        memcpy(hostPRData.vArraySrc,edgeSrc,batchsize*sizeof(vertexId_t));
//        memcpy(hostPRData.vArrayDst,edgeDst,batchsize*sizeof(vertexId_t));
//        for(length_t i=0; i<batchsize; i++) {
//        	printf("hostPRData.vArraySrc[%d]=%d, hostPRData.vArrayDst[%d]=%d\n",i,hostPRData.vArraySrc[i],i,hostPRData.vArrayDst[i]);
//        }
        
	    hostPRData.vArraySrc = (vertexId_t*)allocDeviceArray(hostPRData.nv, sizeof(vertexId_t));
	    hostPRData.vArrayDst = (vertexId_t*)allocDeviceArray(hostPRData.nv, sizeof(vertexId_t));
       
        copyArrayHostToDevice(edgeSrc,hostPRData.vArraySrc,batchsize,sizeof(vertexId_t));
        copyArrayHostToDevice(edgeDst,hostPRData.vArrayDst,batchsize,sizeof(vertexId_t));

#if 1// DEBUG_ON
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
       
#if 0 //array
        allVinA_TraverseEdges_LB<StreamingPageRankOperator::updateContributionsUndirected>(custing,devicePRData,
        		*cusLB,hostPRData.vArraySrc,batchsize);
#else
        
        for(length_t i=0; i<batchsize; i++) {
        	hostPRData.queue.enqueueFromHost(edgeSrc[i]);
        }    
        
        length_t prevEnd = batchsize;
        
        SyncDeviceWithHost(); //added for threashold and iteration count
#if 1       
        cout << "11111   hostPRData.queue.getActiveQueueSize():" <<hostPRData.queue.getActiveQueueSize()<<endl;
        cout << "hostPRData.iteration < hostPRData.iterationMax " << hostPRData.iteration << " "<< hostPRData.iterationMax << endl;
        cout << "h_out>hostPRData.threshhold?" << h_out << " " << hostPRData.threshhold <<endl; 
        while((hostPRData.queue.getActiveQueueSize())>0 
        		&& (hostPRData.iteration < hostPRData.iterationMax)
        		&& (h_out>hostPRData.threshhold)){
        	cout << "\n" << "****hostPRData.queue.getActiveQueueSize()=" << hostPRData.queue.getActiveQueueSize() << endl;
            //      SyncHostWithDevice();
                    SyncDeviceWithHost(); //added

        	allVinA_TraverseEdges_LB<StreamingPageRankOperator::updateContributionsUndirected>(custing,devicePRData,
            		*cusLB,hostPRData.queue,batchsize);

            hostPRData.queue.setQueueCurr(prevEnd);
    		prevEnd = hostPRData.queue.getQueueEnd();
    		
    		allVinA_TraverseVertices<StreamingPageRankOperator::updateDiffAndCopy>(custing,devicePRData,*cusLB);
    		allVinG_TraverseVertices<StreamingPageRankOperator::updateSum>(custing,devicePRData);
    		
            //SyncDeviceWithHost();
            SyncHostWithDevice(); //added
    		
    		copyArrayDeviceToHost(hostPRData.reductionOut,&h_out, 1, sizeof(prType));
    		hostPRData.iteration++;
    	    cout << "22222   hostPRData.queue.getActiveQueueSize():" <<hostPRData.queue.getActiveQueueSize()<<endl;
    	        cout << "hostPRData.iteration < hostPRData.iterationMax " << hostPRData.iteration << " "<< hostPRData.iterationMax << endl;
    	        cout << "h_out>hostPRData.threshhold?" << h_out << " " << hostPRData.threshhold <<endl; 
    	    
        }
        cout << "33333    hostPRData.queue.getActiveQueueSize():" <<hostPRData.queue.getActiveQueueSize()<<endl;
            cout << "hostPRData.iteration < hostPRData.iterationMax " << hostPRData.iteration << " "<< hostPRData.iterationMax << endl;
            cout << "h_out>hostPRData.threshhold?" << h_out << " " << hostPRData.threshhold <<endl; 
        
#endif        
#endif
/*
 * 	hostPRData.iteration = 0;

	prType h_out = hostPRData.threshhold+1;

	while(hostPRData.iteration < hostPRData.iterationMax && h_out>hostPRData.threshhold){
		SyncDeviceWithHost();

		allVinA_TraverseVertices<StreamingPageRankOperator::resetCurr>(custing,devicePRData,*cusLB);
		allVinA_TraverseVertices<StreamingPageRankOperator::computeContribuitionPerVertex>(custing,devicePRData,*cusLB);
		allVinA_TraverseEdges_LB<StreamingPageRankOperator::addContribuitionsUndirected>(custing,devicePRData,*cusLB);
		// allVinA_TraverseEdges_LB<StreamingPageRankOperator::addContribuitions>(custing,devicePRData,*cusLB);
		allVinA_TraverseVertices<StreamingPageRankOperator::dampAndDiffAndCopy>(custing,devicePRData,*cusLB);

		// allVinG_TraverseVertices<StreamingPageRankOperator::resetCurr>(custing,devicePRData);
		// allVinG_TraverseVertices<StreamingPageRankOperator::computeContribuitionPerVertex>(custing,devicePRData);
		// allVinA_TraverseEdges_LB<StreamingPageRankOperator::addContribuitionsUndirected>(custing,devicePRData,cusLB);
		// allVinG_TraverseVertices<StreamingPageRankOperator::dampAndDiffAndCopy>(custing,devicePRData);

		// copyArrayDeviceToDevice(hostPRData.currPR,hostPRData.prevPR, hostPRData.nv,sizeof(prType));

		// allVinG_TraverseVertices<StreamingPageRankOperator::print>(custing,d_out);

		allVinG_TraverseVertices<StreamingPageRankOperator::sum>(custing,devicePRData);
		SyncHostWithDevice();

		copyArrayDeviceToHost(hostPRData.reductionOut,&h_out, 1, sizeof(prType));
		// h_out=hostPRData.threshhold+1;
		// cout << "The number of elements : " << hostPRData.nv << endl;

		hostPRData.iteration++;
	}
}
 */        
        
//        while(hostPRData.iteration < hostPRData.iterationMax && h_out>hostPRData.threshhold){
//                SyncDeviceWithHost();
//                printf("\n--------------- reset curr---------------");
//                allVinA_TraverseVertices<StreamingPageRankOperator::resetCurr>(custing,devicePRData,*cusLB);
              
        //      allVinA_TraverseVertices<StreamingPageRankOperator::computeContribuitionPerVertex>(custing,devicePRData,cusLB);
        //      allVinA_TraverseEdges_LB<StreamingPageRankOperator::addContribuitionsUndirected>(custing,devicePRData,cusLB);
        //      allVinA_TraverseEdges_LB<StaticPageRankOperator::addContribuitions>(custing,devicePRData,cusLB);
        //      allVinA_TraverseVertices<StreamingPageRankOperator::dampAndDiffAndCopy>(custing,devicePRData,cusLB);

        //      allVinG_TraverseVertices<StaticPageRankOperator::resetCurr>(custing,devicePRData);
        //      allVinG_TraverseVertices<StaticPageRankOperator::computeContribuitionPerVertex>(custing,devicePRData);
        //      allVinA_TraverseEdges_LB<StaticPageRankOperator::addContribuitionsUndirected>(custing,devicePRData,cusLB);
        //      allVinG_TraverseVertices<StaticPageRankOperator::dampAndDiffAndCopy>(custing,devicePRData);

        //      copyArrayDeviceToDevice(hostPRData.currPR,hostPRData.prevPR, hostPRData.nv,sizeof(prType));

        //      allVinG_TraverseVertices<StaticPageRankOperator::print>(custing,d_out);

        //      allVinG_TraverseVertices<StaticPageRankOperator::sum>(custing,devicePRData);
//                SyncHostWithDevice();

//                copyArrayDeviceToHost(hostPRData.reductionOut,&h_out, 1, sizeof(prType));
                // h_out=hostPRData.threshhold+1;
                // cout << "The number of elements : " << hostPRData.nv << endl;

//                hostPRData.iteration++;
//       }
#endif
}
#endif


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
#endif
	
	standard_context_t context(false);
	mergesort(d_scores,d_ids,hostPRData.nv,greater_t<float>(),context);

	prType* h_scores = (prType*)allocHostArray(hostPRData.nv, sizeof(prType));
	vertexId_t* h_ids    = (vertexId_t*)allocHostArray(hostPRData.nv, sizeof(vertexId_t));

	copyArrayDeviceToHost(d_scores,h_scores,hostPRData.nv, sizeof(prType));
	copyArrayDeviceToHost(d_ids,h_ids,hostPRData.nv, sizeof(vertexId_t));


    for(int v=0; v<10; v++){
            printf("Pr[%d]:= %.10f\n",h_ids[v],h_scores[v]);
    }

#if 1//PR_UPDATE
    //std::string fname = "pr_values_" + std::to_string(fnum) + ".txt";
    //std::ofstream fp_pr;
    //fp_pr.open(fname.c_str());
    //FILE *fp_pr = fopen("pr_values.txt", "w");

    char buff[100];
    sprintf(buff, "pr_values_sorted_%d.txt", fnum++);
    FILE *fp_pr = fopen(buff, "w");
    for (uint64_t v=0; v<hostPRData.nv; v++)
    {
        fprintf(fp_pr,"%d %d %e\n",v,h_ids[v],h_scores[v]);
    }
   fclose(fp_pr);
#endif

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





// void StreamingPageRank::Run(cuStinger& custing){
// 	// cusLoadBalance cusLB(custing);
// 	// cusLoadBalance cusLB(custing,true,false);
// 	cusLoadBalance cusLB(custing,false,true);
// 	cout << "The number of non zeros is : " << cusLB.currArrayLen << endl;

// 	allVinG_TraverseVertices<StreamingPageRankOperator::init>(custing,devicePRData);
// 	hostPRData.iteration = 0;

// 	prType h_out = hostPRData.threshhold+1;

// 	while(hostPRData.iteration < hostPRData.iterationMax && h_out>hostPRData.threshhold){
// 		SyncDeviceWithHost();

// 		// allVinA_TraverseVertices<StreamingPageRankOperator::resetCurr>(custing,devicePRData,cusLB);
// 		// allVinA_TraverseVertices<StreamingPageRankOperator::computeContribuitionPerVertex>(custing,devicePRData,cusLB);
// 		// allVinA_TraverseEdges_LB<StreamingPageRankOperator::addContribuitionsUndirected>(custing,devicePRData,cusLB);
// 		// allVinA_TraverseVertices<StreamingPageRankOperator::dampAndDiffAndCopy>(custing,devicePRData,cusLB);

// 		// allVinG_TraverseVertices<StreamingPageRankOperator::resetCurr>(custing,devicePRData);
// 		// allVinG_TraverseVertices<StreamingPageRankOperator::computeContribuitionPerVertex>(custing,devicePRData);
// 		// allVinA_TraverseEdges_LB<StreamingPageRankOperator::addContribuitionsUndirected>(custing,devicePRData,cusLB);
// 		// allVinG_TraverseVertices<StreamingPageRankOperator::dampAndDiffAndCopy>(custing,devicePRData);

// 		// copyArrayDeviceToDevice(hostPRData.currPR,hostPRData.prevPR, hostPRData.nv,sizeof(prType));

//     // void            *d_temp_storage = NULL;
//     // size_t          temp_storage_bytes = 0;
//     // DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, (prType*)hostPRData.absDiff, (prType*)hostPRData.reductionOut, hostPRData.nv);
//     // d_temp_storage = (void*)allocDeviceArray(temp_storage_bytes,1);
//     // DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, (prType*)hostPRData.absDiff, (prType*)hostPRData.reductionOut, hostPRData.nv);


//     // freeDeviceArray(d_temp_storage);
//     // printf("\n%d\n", temp_storage_bytes);
//     // Run

//     // float* h_in = new float[10000];
//     // float  h_reference;
//     // for(int i=0;i<10000; i++)
//     // 	h_in[i]=(float)i/float(10000);
//     // float* d_referernce, *d_in;
//     // d_in=(float*)allocDeviceArray(10000,sizeof(float));
//     // d_referernce=(float*)allocDeviceArray(1,sizeof(float));
//     // copyArrayHostToDevice(h_in,d_in,10000,sizeof(float));

//     // void            *d_temp_storage = NULL;
//     // size_t          temp_storage_bytes = 0;
//     // DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_referernce, hostPRData.nv);
//     // d_temp_storage = (void*)allocDeviceArray(temp_storage_bytes,1);
//     // DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_referernce, hostPRData.nv);


//     // copyArrayDeviceToHost(d_referernce,&h_reference,1, sizeof(float));
//     // cout << "Hello MF "<<  h_reference << endl;

//     int* h_in = new int[10000];
//     int  h_reference;
//     for(int i=0;i<10000; i++)
//     	h_in[i]=(int)i;
//     int *d_out=NULL, *d_in=NULL;
//     // d_in=(int*)allocDeviceArray(10000,sizeof(int));
//     // d_out=(int*)allocDeviceArray(1,sizeof(int));
//     g_allocator.DeviceAllocate((void**)&d_in, sizeof(float) * 10000);
//     g_allocator.DeviceAllocate((void**)&d_out, sizeof(float) * 1);

//     copyArrayHostToDevice(h_in,d_in,10000,sizeof(int));

//     void            *d_temp_storage = NULL;
//     size_t          temp_storage_bytes = 0;
//     DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, 10000);
//     // d_temp_storage = (void*)allocDeviceArray(temp_storage_bytes,1);
//     g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
//     DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, 10000);
//     DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, 10000);
//     DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, 10000);
//     DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, 10000);
//     DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, 10000);

    
// 	allVinG_TraverseVertices<StreamingPageRankOperator::print>(custing,d_out);


//     copyArrayDeviceToHost(d_out,&h_reference,1, sizeof(int));
//     cout << "Hello MF "<<  h_reference << endl;
// 	// allVinG_TraverseVertices<StreamingPageRankOperator::print>(custing,d_out);

//     // freeDeviceArray(d_temp_storage);
//     // freeDeviceArray(d_in);
//     // freeDeviceArray(d_out);
//     if (d_temp_storage) g_allocator.DeviceFree(d_temp_storage);
//     if (d_in) g_allocator.DeviceFree(d_in);
//     if (d_out) g_allocator.DeviceFree(d_out);


// 		// allVinG_TraverseVertices<StreamingPageRankOperator::sum>(custing,devicePRData);

// 		SyncHostWithDevice();

// 		// copyArrayDeviceToHost(hostPRData.reductionOut,&h_out, 1, sizeof(prType));
// 		// h_out=hostPRData.threshhold+1;
// 		// cout << "The number of elements : " << hostPRData.nv << endl;

// 		hostPRData.iteration++;
// 	}
// }
