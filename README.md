
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


CPU overhead bound 상태가 발생하는 주요 원인은 다음과 같습니다:

컨텍스트 스위칭(Context Switching): 프로세스 간 전환 시 발생하는 오버헤드입니다. 컨텍스트 스위칭이 자주 일어날수록 CPU 오버헤드가 증가합니다.
인터럽트 처리(Interrupt Handling): 하드웨어 인터럽트나 소프트웨어 인터럽트 처리 시 발생하는 오버헤드입니다. 인터럽트 빈도가 높을수록 CPU 오버헤드가 증가합니다.
시스템 호출(System Calls): 프로세스가 운영체제에 서비스를 요청할 때 발생하는 오버헤드입니다. 시스템 호출 횟수가 많을수록 CPU 오버헤드가 증가합니다.
메모리 관리(Memory Management): 가상 메모리 시스템 관리, 페이지 폴트 처리 등의 오버헤드입니다. 메모리 사용량이 많고 페이지 폴트 발생 빈도가 높을수록 CPU 오버헤드가 증가합니다.
동기화 오버헤드(Synchronization Overhead): 멀티스레드 또는 멀티프로세스 환경에서 공유 리소스에 대한 동기화 오버헤드입니다. 락(lock)이나 세마포어(semaphore) 등의 동기화 메커니즘 사용 빈도가 높을수록 CPU 오버헤드가 증가합니다.

-> Torch.compile 큰 region에서 단일 컴파일되는 region 으로 만든다. 특히 mode= 'reduce-overhead'의 경우 cpu overhead 발생을 크게 줄일수 있다

Graph break는 CPU와 GPU 간의 데이터 전송 오버헤드로 인한 성능 저하 현상을 의미하며, 이를 최소화하기 위해서는 배치 크기 조절, 연산 그래프 최적화, 모델 병렬화 

-> full graph= True로 설청해 graph break가 일어나지 않도록 확인함

e.g) torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

문제 2가지
1-1.kv cache , long context가 될수록 kv cache의 메모리를 재할당해야함 ->  비쌈
1-2. kv cache의 동적할당은 overhead 줄이기 힘들게 만듦, cudagraph를 사용할수 없게됨
-> static kv cache를 사용 
효과 : kv cache길이의 한계를 정하고 컴퓨팅할 때 안쓰는 부분은 masking함
2 prefill 문제
prefill은 더 큰 동적시스템을 필요로함 다양한 프롬프트 길이를 가질수 있기 때문에 한계 조정 불가능
