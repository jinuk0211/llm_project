DATA 

![image](https://github.com/jinuk0211/llm_project/assets/150532431/a04f0f7d-5a4c-402c-8d3f-833c16c10305)

seq

![image](https://github.com/jinuk0211/llm_project/assets/150532431/91fe2a6c-82e3-4e72-b46d-4b766890af9a)

TENSOR

![image](https://github.com/jinuk0211/llm_project/assets/150532431/9ac7b9cb-3681-4c0a-bd4c-0052c42fd43c)

pipe 

![image](https://github.com/jinuk0211/llm_project/assets/150532431/bc8c1451-ed0e-48b2-ae71-7f574f34b954)

MoE

![image](https://github.com/jinuk0211/llm_project/assets/150532431/8d0c1c52-cfd3-4615-8e31-a1c72fa37e52)

https://www.threads.net/@rien_n_est/post/C4QefgpysSM

DataParallel - Data parallelism은 학습 시간을 단축한다는 장점이 있지만 매 Weight parameter를 업데이트할 때마다 여러 GPU가 학습한 결과를 종합한 후 다시 나누는 Synchronization이 필요한 단점이 존재

TensorParallel -커다란 Weight matrix를 여러 GPU로 나누어 연산을 한 후 그 결과값을 합치는(Concatenate)