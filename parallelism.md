DATA 

![image](https://github.com/jinuk0211/llm_project/assets/150532431/a04f0f7d-5a4c-402c-8d3f-833c16c10305)

ZeRO DATA

![image](https://github.com/jinuk0211/llm_project/assets/150532431/016e4b65-bcc2-476a-a4ef-850622d07164)

seq

![image](https://github.com/jinuk0211/llm_project/assets/150532431/91fe2a6c-82e3-4e72-b46d-4b766890af9a)

TENSOR

![image](https://github.com/jinuk0211/llm_project/assets/150532431/9ac7b9cb-3681-4c0a-bd4c-0052c42fd43c)

pipeline

![image](https://github.com/jinuk0211/llm_project/assets/150532431/bc8c1451-ed0e-48b2-ae71-7f574f34b954)

interleaved pipeline

![image](https://github.com/jinuk0211/llm_project/assets/150532431/b471f12e-20ac-4d61-8f5e-67dc6c7f40be)

DeepSpeed, Varuna and SageMaker에서 사용

MoE

![image](https://github.com/jinuk0211/llm_project/assets/150532431/8d0c1c52-cfd3-4615-8e31-a1c72fa37e52)

https://www.threads.net/@rien_n_est/post/C4QefgpysSM

llm 만들기 가이드 in 2024 - 병렬화

4D parallelism

DataParallelism - parameter
PipelineParallelism - layer
TensorParallelism - tensor
SequenceParalllelism - layernorm, activation

파이토치, 허깅페이스 document 

1. 데이터 병렬화 DP, DDP
Data Parallell,Distributed Data Parallel

Zero-DP 시각화 ↓

La | Lb | Lc     
a0 | b0 | c0
a1 | b1 | c1  
a2 | b2 | c2

gpu rank 0 에는 layer A의 a0, layer B의 b0, layer C의 c0
GPU0:
La | Lb | Lc
a0 | b0 | c0

gpu rank 1에는 각 layer의 a1,b1,c1
GPU1:
La | Lb | Lc
a1 | b1 | c1

요약
DataParallel - DP는 학습 시간을 단축한다는 장점이 있지만 매 Weight parameter를 업데이트할 때마다 여러 GPU가 학습한 결과를 종합한 후 다시 나누는 Synchronization이 필요한 단점이 존재.

작동방식 
- 동기화가 진행되는 첫번째 gpu에 부하가 많이가게됨
파이썬의 GIL 특성상 multithread x 
-> multiprocessing을 사용, 하나의 gpu를 위해 하나의 process를 사용(DDP) ->
하지만 gpu 계산한 결과를 합치는 과정이 필요해지기에 gpu끼리 통신하기 위한 백엔드 라이브러리가 필요해짐 ex)'nccl'

밑의 과정이 DP인데 batch별로 gpu들의 복제된 모델에 들어가 이를 forward pass 진행 후 loss를 평균을 내서 구한뒤 첫뻔쨰 gpu에 이를 저장 후 다시 나눈후 backward pass 진행

![image](https://github.com/jinuk0211/llm_project/assets/150532431/46fee905-20c0-42de-892b-83a3970710b8)

FSDP 

- ![image](https://github.com/jinuk0211/llm_project/assets/150532431/55edd15d-9d79-433b-853b-dc92ddd98875)


TensorParallel -커다란 Weight matrix를 여러 GPU로 나누어 연산을 한 후 그 결과값을 합치는(Concatenate)

DDP에서는 각 프로세스가 계산한 gradient를 모두 합산한 후, 전체 프로세스 수로 나누어 평균 gradient를 FSDP 역시 비슷한 방식으로 작동하지만, 메모리 사용량을 최적화하기 위해 gradient와 모델 파라미터를 샤딩하는 추가적인 메커니즘
