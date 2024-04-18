
![image](https://github.com/jinuk0211/llm_project/assets/150532431/4bae1f5c-ac33-4e2d-bda7-c473cc4dca68)
![image](https://github.com/jinuk0211/llm_project/assets/150532431/48b4130a-e91c-487f-954c-36e1655ac882)

# llm_project
목표 : bio 관련 llm  

https://arxiv.org/pdf/2403.04652.pdf
01.AI Yi 모델

https://arxiv.org/pdf/2402.16827.pdf
A Survey on Data Selection for Language Models

https://arxiv.org/pdf/2402.00159.pdf
dolma 데이터셋
https://arxiv.org/pdf/2306.01116.pdf
falcon 데이터셋 - 웹데이터

훈련과정 다섯가지
![image](https://github.com/jinuk0211/llm_project/assets/150532431/acb27926-5252-4ab5-9f93-b85e1ba45f4e)
pretraining
instruction training
alignment - 유해성 내용 제거
in-context learning
fine-tuning (task,domain 맞춤형)

![image](https://github.com/jinuk0211/llm_project/assets/150532431/4a3d1253-50eb-49dc-ad57-d8a5804e8c3b)
![image](https://github.com/jinuk0211/llm_project/assets/150532431/44b49189-ac61-4dad-9f1e-fc8da201e493)

데이터 source
![image](https://github.com/jinuk0211/llm_project/assets/150532431/24ff5fa0-d758-4a92-815e-f25f80373c7e)
합성데이터 예시
![image](https://github.com/jinuk0211/llm_project/assets/150532431/511c46a2-0027-43f5-94f9-1d3936d4c98a)
코드 데이터-starcoder2
![image](https://github.com/jinuk0211/llm_project/assets/150532431/ab3dbcdd-b8d8-4edf-80c6-8395947fbbb8)

데이터 필터링 방법
휴리스틱
![image](https://github.com/jinuk0211/llm_project/assets/150532431/19a4f588-95bd-42da-9fc7-e8aa1967add2)
데이터 중복제

2.모델링
![image](https://github.com/jinuk0211/llm_project/assets/150532431/71465a56-1ef2-40a7-adc0-0dd187aac83c)
병렬화 가능한 4가지
![image](https://github.com/jinuk0211/llm_project/assets/150532431/3bd01f19-f633-490c-942d-f236105787b0)
tensor(모델) 병렬화
![image](https://github.com/jinuk0211/llm_project/assets/150532431/bb5a3ae1-72ec-45d8-99d1-0216358f1e10)
파이프 병렬화
![image](https://github.com/jinuk0211/llm_project/assets/150532431/0816f5e6-af00-4897-9b6c-fac6544d4f24)
![image](https://github.com/jinuk0211/llm_project/assets/150532431/f7393db2-acd1-40d2-b90d-9a8c0bc7e397)
sequence 병렬화
![image](https://github.com/jinuk0211/llm_project/assets/150532431/633732d8-6c3e-4e14-9731-c3cd2e3c5504)
flashattention
![image](https://github.com/jinuk0211/llm_project/assets/150532431/ae8c9561-9434-4121-abdd-18d9e4668e7d)
MoE
![image](https://github.com/jinuk0211/llm_project/assets/150532431/02d00603-98cf-4c84-9172-2f6e96c2ef38)
Mamba 
![image](https://github.com/jinuk0211/llm_project/assets/150532431/77a775b7-6c25-4a30-ac1a-310cc6a57dbf)

nanotron?
![image](https://github.com/jinuk0211/llm_project/assets/150532431/91bebd6d-5396-455c-8129-a117727ab1e5)

alignment - 정치, 폭력, 19세 이상, 윤리적 내용 제거 - RLFH
![image](https://github.com/jinuk0211/llm_project/assets/150532431/78b2210b-f2d1-4007-9967-28377aaca9ac)

PPO
![image](https://github.com/jinuk0211/llm_project/assets/150532431/64e6cf90-bb4b-45cc-9f38-369d499e2b10)
DPO
![image](https://github.com/jinuk0211/llm_project/assets/150532431/a9918777-2eca-47dd-9bc1-3acd60fb2190)

추론속도 향상
Quantization
![image](https://github.com/jinuk0211/llm_project/assets/150532431/27a856bb-2a4e-4515-89f9-66818c212cb5)

speculative decoding
https://arxiv.org/pdf/2401.10774.pdf

compiling and cuda graph
https://pytorch.org/blog/accelerating-generative-ai-2/
