DATA 

![image](https://github.com/jinuk0211/llm_project/assets/150532431/a04f0f7d-5a4c-402c-8d3f-833c16c10305)

ZeRO DATA

![image](https://github.com/jinuk0211/llm_project/assets/150532431/016e4b65-bcc2-476a-a4ef-850622d07164)

seq

![image](https://github.com/jinuk0211/llm_project/assets/150532431/91fe2a6c-82e3-4e72-b46d-4b766890af9a)

TENSOR

![image](https://github.com/jinuk0211/llm_project/assets/150532431/9ac7b9cb-3681-4c0a-bd4c-0052c42fd43c)

pipe 

![image](https://github.com/jinuk0211/llm_project/assets/150532431/bc8c1451-ed0e-48b2-ae71-7f574f34b954)

MoE

![image](https://github.com/jinuk0211/llm_project/assets/150532431/8d0c1c52-cfd3-4615-8e31-a1c72fa37e52)

https://www.threads.net/@rien_n_est/post/C4QefgpysSM

DataParallel - DP는 학습 시간을 단축한다는 장점이 있지만 매 Weight parameter를 업데이트할 때마다 여러 GPU가 학습한 결과를 종합한 후 다시 나누는 Synchronization이 필요한 단점이 존재
작동방식 - 동기화가 진행되는 첫번째 gpu에 부하가 많이가게됨
파이썬의 GIL 특성상 multithread x 
multiprocessing을 사용, 하나의 gpu를 위해 하나의 process를 사용(DDP) ->
하지만 gpu 계산한 결과를 합치는 과정이 필요해지기에 gpu끼리 통신하기 위한 백엔드 라이브러리가 필요해짐 ex)'nccl'
![image](https://github.com/jinuk0211/llm_project/assets/150532431/46fee905-20c0-42de-892b-83a3970710b8)


TensorParallel -커다란 Weight matrix를 여러 GPU로 나누어 연산을 한 후 그 결과값을 합치는(Concatenate)

DDP에서는 각 프로세스가 계산한 gradient를 모두 합산한 후, 전체 프로세스 수로 나누어 평균 gradient를 FSDP 역시 비슷한 방식으로 작동하지만, 메모리 사용량을 최적화하기 위해 gradient와 모델 파라미터를 샤딩하는 추가적인 메커니즘
