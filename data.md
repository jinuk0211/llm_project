llm 만들기 가이드 part 4 - 데이터

solar 10.7b - 2024/4/4
https://arxiv.org/pdf/2312.15166.pdf

yi - 34b - 2024/3/7
https://arxiv.org/pdf/2403.04652.pdf

A Survey on Data Selection for Language Models - 2024/3/8
https://arxiv.org/pdf/2402.16827.pdf

The RefinedWeb Dataset for Falcon LLM - 2023/6/1
https://arxiv.org/pdf/2306.01116.pdf

an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research - 2024/1/31
https://arxiv.org/pdf/2402.00159.pdf


2 page

광범위한 데이터의 selection 연구를 할 수 있는 여건을 갖춘 조직과 기관은 매우 적다. 또한 그들은 그들이 찾아낸 방법, 연구결과를 잘 공유하지 않는 다. 이러한 지식의 격차를 좁히기 위해, 지금까지 데이터 selection method의 related work를 모은 이해하기 쉬운 리뷰를 제공하고자 한다

언어모델의 등장으로 데이터 selection은 특히 더 중요해졌다. llm이 겪는 여러 train In-context learning, instruction tuning, alignment, pretraining, finetuning 각 단계마다 데이터 selection은 매우 중요한 역할을 하게된다

pretrain 과정에서 llm 은 텍스트 데이터의 매우 큰 corpus로 학습되는데 이는 대부분 인터넷으로 현재 2500억개의 웹페이지가 있으며 이는 11페타바이트 정도에 해당한다. 이와 같이 매우 큰 규모의 데이터 때문에 높은 퀄리티의 데이터만을 필터링하는게 목표가 된다

3 page

taxnomy

data point : x(i)
데이터 포인트 x(i)는 모델을 훈련하거나 평가하는 데 사용되는 단일 데이터 샘플을 구성하는 토큰의 정렬된 집합이다. 예를 들어, 언어 모델링에서 x(i)는 인터넷 문서, 책 또는 과학 논문의 토큰 시퀀스일 수 있다. 언어 모델은 훈련 시 입력 시퀀스 길이가 제한되어 있으므로, 긴 문서는 종종 여러 개의 데이터 포인트로 분할된다. 실제로 이것은 x(i)가 완전한 문서가 아닌 문서의 일부를 구성할 수 있음을 의미한다 

ϕ(D) = ϕ1 ◦· · · ◦ϕn(D)
  ϕ 기호는 component 함수로 filtering component(바람직하지 않은 데이터 포인트 자체 제거), cleaning component(각각의 데이터 포인트에서 바람직하지 않은 내용 제거), mixing component(마지막의 dataset에서 얼마나 남은 데이터 포인트가 사용되야 하는지를 결정)


4 page

υ(x(i)) : utility(data point)
유틸리티 함수 υ(x(i))는 데이터 포인트를 실제 숫자로 맵핑하는데
 간단한 binary indicator(예: x(i)의 문자 수가 10보다 큰 경우에만 D에 포함) 또는 확률 함수(예: υ가 x(i)가 위키피디아 기사일 가능성이면 υ(x(i))에 정의된 확률에 따라 x(i)를 D에 포함)가 될 수 있다 D= dataset

selection 메커니즘
유틸리티 함수의 결과값을 사용하여 데이터 포인트가 결과 부분집합(Dataset)에 포함될지 여부(그리고 경우에 따라 데이터 포인트가 몇 번 반복되어야 하는지)를 결정한다. 또한 임계값(threshold)이 필요한 선택 메커니즘에는 선택 민감도(sensitivity)가 필요하다(예: υ(x(i)) > 0.9인 경우에만 x(i)를 포함). 

5 page 

분포 매칭
Distribution Matching의 주요 목표는 모델이 평가되거나 배포될 바람직한 대상 분포와 유사한 특성을 가진 데이터를 선택하는 것이다. 예를 들어, 바람직한 분포는 알려진 고품질 데이터, 특정 언어 또는 대상 도메인(예: 금융, 의료 또는 법률)으로 정의될 수 있다, 
바람직한 대상 분포의 정확한 규격은 잘 정의된 경우(예: 언어 감지 as in detecting a language)에서 모호한 경우(예: "고품질" 데이터 “high quality” data)까지 다양할 수 있다. 

일부 Distribution Matching 방법은 대상 분포와의 유사성이 일반적으로 유틸리티가 되는 데이터 표현 분포를 일치시키려고 한다(예: similarity to Wikipedia data). 
다른 분포 매칭 방법은 대상 데이터 세트에서 샘플링된 데이터의 통계를 유틸리티로 사용한다(예: total number of characters per example).


6 page

분포 다양화
Distribution diversification은 샘플의 이질성(unique,hetero)을 우선시하고 중복을 제거하는 것을 목표로 한다. 이들은 데이터 포인트 간의 유사성을 측정할 수 있는 representation 공간에서 작동한다. 데이터 포인트의 유틸리티는 표현 공간에서 다른 데이터 포인트와의 관계(유사성)에 의해 정의된다. 분포 다양화 방법은 종종 특정 representation 공간(예: 문자 또는 벡터, characters or vectors)에서 유사한 데이터 포인트를 제거한다. 매우 유사한 데이터 포인트를 제거함으로써 다양화 방법은 데이터의 중복을 제거하여 데이터 세트 크기를 줄임으로써 훈련 효율성을 향상시킬 수 있다. 또한 분포 다양화 방법은 데이터 세트 분포를 더 평평하게 함으로써 과잉 적합 감소, 편향 감소 및 robustness 향상을 가져올 수 있다.
