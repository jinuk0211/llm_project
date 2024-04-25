llm의 다음 단계, 초인공지능
현재의 llm
1. 정적이고 static
2. 데이터 기반 data-bound
-> dynamic, robust intelligence 시스템으로
PKU lab, 알리바바 survey - 2024/4/22
arxiv.org/pdf/2…
llm이 llm 스스로가 만들어낸 결과(experiment)를 얻어(aquire) 정제(refine)하고 스스로 학습(learn)하게 하는 self-evolution(스스로 발전, 인간이나 외부의 model의 supervision이 아닌) 접근 방식에 대한 연구가 급속하게 늘어나고 있다. 인간의 경험론적 학습에서 영감을 받은 이 새로운 training 패러다임은 LLM을 초인공지능로 scale할 수 있을지 모르는 potential을 제공한다.

![image](https://github.com/jinuk0211/llm_project/assets/150532431/e7382484-c3a5-4f16-b377-6944d333df6f)

![image](https://github.com/jinuk0211/llm_project/assets/150532431/5952880a-acf4-4f33-a903-416704675931)


AI의 빠른 발전으로 GPT-3.5(2022), GPT-4(2023), Gemini(2023), LLaMA(2023a,b) 및 Qwen(2023)의 LLM은 언어 이해 및 생성에 있어 상당한shift를 가져왔다
위의 모델은 3가지 발전 단계를 거친다
1. pretrain(대규모 다양한 corpus로)
언어와 세상 지식에 대한 일반적인 이해를 얻기 위해 - 공부, 언어학습
2. SFT
downstream task를 이끌어 내기 위해 - 번역, 요약, 대화 등
3. alignment
사람처럼 말하고 행동하기 위해 - 도덕, 예의 등
이러한 학습 패러다임을 통해 질의응답(Tan 외, 2023), 수학적 추론(Collins 외, 2023), 코드 생성(Liu 외, 2024b), 환경과 상호작용이 필요한 task 해결(Liu 외, 2023b) 등 다양한 분야에서 놀라운 zero shot and in-context 능력을 발휘할 수 있게 되었다.


그럼에도 불구하고 새로운 세대의 LLM에는 과학 발견(Miret와 Krishnan, 2024), 미래 예측(Schoenegger 외, 2024) 등 더욱 복잡한 작업이 요구되고 있다.
하지만 현재 LLM은 기존 학습 패러다임의 modeling, annotation, 이러한 훈련 패러다임과 연관된 evaluation으로 인해 이러한 정교한 작업에서의 challenge에 직면해 있다(Burns 외, 2023).
더욱이 최근 개발된 Llama-3 모델은 무려 15조 단어(token)로 구성된 corpus에서 학습되었다. real-world 데이터를 더 많이 추가하여 모델 성능을 크게 향상시키는 데는 한계가 있을 수 있음을 제시하는 엄청난 데이터 양이다.
이로 인해 LLM의 자연 진화적 메커니즘(인간의 지능과 유사한)에 대한 관심이 높아지고 있다. alphago에서 alphozero로의 발전(자연친화적 지능에 관한)



AlphaZero’s self-play method -> 라벨링된 데이터 필요없음 -> llm은 사람의 supervision이 매우 필요한 것과 다름
지금까지의 연구
self-instruct(Wang 외, 2023b), self-play(Tu 외, 2024), self-improving(Huang 외, 2022),self-training(Gulcehre 외, 2023) 등
특히 DeepMind의 AMIE 시스템(Tu 외, 2024)은 진단 정확도에서 primary care 의사를 능가하며, 마이크로소프트의 WizardLM-2는 초기 GPT-4 버전보다 뛰어난 성능을 보인다.(natural framework)



4단계로 요약
experience acquisition - 무언가를 경험
experience refinement - 깨달음,
updating
evaluation
four stage 자세히
LLM은 먼저 새로운 과제(new task)를 evolving 하고 해당 솔루션을 생성하여 경험을 얻는다. 이후 이러한 경험을 정제하여 더 나은 supervision signal(지도 신호)를 얻는다. 모델의 in-weight 또는 in-context을 업데이트한 후 LLM은 진행 상황을 측정하고 새로운 목표를 설정하기 위해 평가된다.
~introduction,개요 끝 세부내용 길어서 다음에

![image](https://github.com/jinuk0211/llm_project/assets/150532431/999c8f98-d9a8-44d9-b0e4-4aab4482a322)

특정 evolution goal에 집중해 모델이 관련된 task에 참여하게 함 -> 경험(experience)를 최적화 -> 아키텍쳐 update -> next cycle, iteration으로 가기 전 얼마나 진척했는지 확인

experience aquistion
t 번째 iteration 때마다, evolution objective(goal)를 확인하고 이 objective를 따라 새로운 task를 시작하고 solution을 내는데 이를 environment으로부터 피드백을 받게 된다. 그리고 새로운 experience를 얻는 것으로 끝마친다.

refinement
위의 받은 new experience를 정체하는데 부정확한 데이터 제거, 불완벽한 것을 향상 이를 통해 refined된 T,Y를 받는다.

update 
이 T,Y를 framework에 통합시키는데 이것이 모델이 현재상태에서 optimized되는 과정이다

![image](https://github.com/jinuk0211/llm_project/assets/150532431/745cb593-e096-4fa6-aa29-082f51efa87f)
![image](https://github.com/jinuk0211/llm_project/assets/150532431/82c2359a-f1a5-4a0c-a608-54d97d75f53a)


evaluation
외부 ENV에 의해 모델이 평가되는데 이 다음 iteration의 objective function에 영향을 줌

evolution objetive에 관해
모델의 자가발전을 위한 가이드를 제공하는 미리 정해진 목표로 인간이 필요와 욕구에 따라 목표를 설정하는 것과 마찬가지로 모델이 self 업데이트하는 방식을 결정하기 때문에 중요하다

LLM은 새로운 데이터에서 자율적으로 학습하고, 알고리즘을 최적화하며, 변화하는 환경에 적응할 수 있어 피드백이나 자체 평가에서 자신의 필요를 "느끼고" 인간 개입 없이 기능을 향상시키기 위한 목표를 스스로 설정할 수 있게된다
 
수식화
evolution objective를 evolution abilitiy, evolution direction으로 정의한다. 
evolution ability는 innate and detailed skill을 나타내고, 진화 방향은 evolution objective가 향상시키고자 하는 방향이다

![image](https://github.com/jinuk0211/llm_project/assets/150532431/860ab150-4079-4fdf-afb6-e00503d3f07f)
