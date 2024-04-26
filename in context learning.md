llm 만들기 가이드 - ICL

in-context learning

1. Why Can GPT Learn In-Context?
Language Models Implicitly Perform Gradient Descent as
Meta-Optimizers
https://arxiv.org/pdf/2212.10559

2. many-shot (2024년 4월 17일 구글 deepmind)
https://arxiv.org/pdf/2404.11018.pdf - context window 가 늘어나며 가능해짐
llama 3 - 8192, chatgpt4 - 128k

3. Chain of Thought without prompting
https://arxiv.org/abs/2402.10200

Large Language Models are Zero-Shot Reasoners 
2022년 5월 24일 - 구글, 도쿄대학
https://arxiv.org/abs/2205.11916

1.
![image](https://github.com/jinuk0211/llm_project/assets/150532431/3dc97499-bfce-41ea-a6fc-c706343efa64)


![image](https://github.com/jinuk0211/llm_project/assets/150532431/dd275103-00f3-4abe-8384-5772fede0744)

![image](https://github.com/jinuk0211/llm_project/assets/150532431/9e1d0218-5e8c-4cfb-a369-05990f35de5d)


performance 향상의 원리  

x는 prompt에 입력한 우리가 풀여야 할 query를
X'은 prompt에 적은 예시들을 demonstration
Error는 backpropagation에서 구해진 error로 이를 value값으로 사용한다

linear layer 의 기존의 gradient를 구하는 것을 attention 메커니즘의 수식으로 바꿔 생각 

(9)의 4-> 5 설명
E = {e1, e2, ...}는 오차 신호 값들의 집합
X' = {x'1, x'2, ...}는 과거 입력 표현들의 집합(키)
x는 현재 입력(쿼리)

알수있는 것 ->
Dual Form Between Attention and Linear
Layers Optimized by Gradient Descent



이를 ICL의 demonstration에 적용
모델의 attention layer를 meta optimizer로 활용해 demonstration의 example을 In-context learning

query token이 demonstration들에 attention을 주는 식과 자기 자신에 attention을 주는 식이 나옴 . 자기 자신한테 attention을 주는 식은 zero-shot learning과 같음.
demonstration에 attention 주는 식을 거꾸로 수행 이를 gradient descent식으로 되돌리면 demonstration들의 token에 대한 query의 attention은 gradient로 해석이 가능해진다. ->

demonstration의 질문 : 답변이라든지
문장 : 라벨의 context에 대한 이해도가 향상된다

Query에도 문장, 질문이 들어오면 라벨, 답변 형식의 답이 나오게 됨

![image](https://github.com/jinuk0211/llm_project/assets/150532431/c2e2ff74-158e-4f28-88ab-f1bae8ade562)


![image](https://github.com/jinuk0211/llm_project/assets/150532431/7d04378a-2d07-4116-b3aa-81d7794250a2)



대규모 언어 모델(LLMs)은 추론 시 컨텍스트에서 제공된 몇 가지 예시로 학습하는 few-shot in-context learning(ICL)에 탁월한 성능을 보임 - 어떠한 가중치 업데이트도 없이 학습 

최근 확장된 컨텍스트 윈도우를 통해 수백 또는 수천 개의 예시로 ICL을 살펴볼 수 있게 됨 - 이를 many-shot이라하는데 Few-shot에서 many-shot으로 넘어가면서 다양한 생성 및 판별(generative, discriminative) 과제에서 상당한 성능 향상을 관찰함. promising한 결과이지만, many-shot ICL은 사용 가능한 인간 생성 출력물의 양에 의해 bottleneck이 발생할 수 있음.

이러한 제한을 완화하기 위해
-> "Reinforced ICL"과 "Unsupervised ICL"이라는 두 가지 새로운 설정 
1. Reinforced ICL은 인간이 작성한 rationale 대신 모델 생성 chain of thought rationale을 사용함
2. Unsupervised ICL은 프롬프트에서 rationale를 완전히 제거하고 모델에 도메인 특화 입력만을 프롬프트

이는 복잡한 reasoning 과제에서 뛰어난 성능을 보임. 또한 few-shot과 달리 pretraining에서의 데이터,학습 편향을 overriding하고 numerical input에서 고차원 함수를 학습하는데 효과적임

![image](https://github.com/jinuk0211/llm_project/assets/150532431/a4666c69-7a05-494c-9745-d7502b849cbc)


unsupervised ICL 예시

![image](https://github.com/jinuk0211/llm_project/assets/150532431/7343fb5a-91f7-4f38-a578-1c1d0518979e)

![image](https://github.com/jinuk0211/llm_project/assets/150532431/c13de153-4a18-470d-b1e7-f78831f85487)

![image](https://github.com/jinuk0211/llm_project/assets/150532431/6c6bc7d6-f8f4-4058-b618-4357a2e67adc)
