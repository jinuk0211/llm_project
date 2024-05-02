수학(매듭 이론)과 물리학(앤더슨 localization)의 2가지 예시를 활용하여 KAN이 과학자들이 수학과 물리 법칙을 (재)발견하는 데 있어 도움이 되는 "협력자"가 될 수 있음을 보여줌

다층 퍼셉트론(Multi-layer perceptrons, MLPs) 은 fully connected feed forward 신경망으로 알려져 있으며, 오늘날 딥러닝 모델의 근간이 되는 구성 요소. MLPs의 중요성은 과소평가될 수 없는데, 왜냐하면 이들은 범용 근사 정리(universal approximation theorem)에 의해 보장된 표현력(representation) 때문에 비선형 함수를 근사화하는 machine leraning의 기본 모델이기 때문/ 하지만 MLPs가 우리가 구축할 수 있는 최고의 비선형 회귀 모델인가?에 대한 의문점 존재
MLPs의 광범위한 사용에도 불구하고, 이들은 중대한 단점이 있음. 예를 들어 트랜스포머[4]에서 MLPs는 거의 모든 non embeddings 매개변수를 소비하며 일반적으로 (attention) 계층에 비해 해석하기 (less interpretable) 어려움

MLP - inspired by universal approximation theorem
하나의 hidden layer에 수많은 노드를 쌓으면 거의 모든 함수를 근사할수 있다
![image](https://github.com/jinuk0211/llm_project/assets/150532431/e5e75dd1-d5a9-4f2a-bfee-6bece9cf39c9)
딥러닝의 지도학습의 근간
![image](https://github.com/jinuk0211/llm_project/assets/150532431/1ca9507c-0930-467c-bba8-970fb3a01435)


KAN - inspired by Kolmogorov-Arnold representation theorem

실제 실험
일반적으로 MLP보다 훨씬 작은 계산 graph.
예를 들어 우리는 PDE(편미분방정식) 해결을 위해 2 layer,width 10 KAN을 사용했는데 4 layer, width 100 MLP보다 100배 더 정확하고(10-7 vs 10-5 MSE) parameter 효율성도 100배 더 높음(10^2 vs 10^4 params)

splines과 MLPs을 합친 것

spline은 compositional structures 를 사용할 수 없음으로 COD(curse of dimensionality)가 발생
COD - 데이터 학습을 위해 차원이 증가하면서 학습데이터 수가 차원의 수보다 적어져 성능이 저하되는 현상.

MLP는 feature learning 덕분에 COD 덜 시달리지만, univariate 함수를 최적화할 수 없기 때문에 low dimension에서 spline보다 정확도가 낮다.

이를 합친 것 -> KAN
spline : 차원의 저주(COD) 때문에 큰 N에 대해서는 실패할 것. 반면 MLP는 generalized additive structure을 잠재적으로 학습할 수 있지만, 문제가, 예를 들어)ReLU 활성화 함수를 사용할 때 지수 함수와 사인 함수를 근사화하는 데에는 매우 비효율적. 대조적으로 KAN은 합성 구조(compositional structure)와 univariate 함수 모두를 꽤 잘 학습할 수 있기 때문에 MLP 성능을 큰 폭으로 능가


![image](https://github.com/jinuk0211/llm_project/assets/150532431/a206550d-ac78-45d8-9ad0-3325e42bda14)

수학
만약의 근사함수 f(x)가 다변수 연속함수(on a bounded domain)라면 
이는 단일 변수, 이항연산만의 연속함수를 유한하게 composition 하는 것으로 계산할 수 있다
->
고차원 함수 f(x)를 학습하는 것은 여러개의 1차원 함수를 학습하는 것으로 귀결됨.


실제 예시
![image](https://github.com/jinuk0211/llm_project/assets/150532431/f1e8e8b6-ce22-417a-8378-4cd1f06336ac)

but
이러한 1차원 함수들은 매끄럽지 않고(non smooth) 심지어 프랙탈(fractal) 구조를 가질 수 있기 때문에 실제로는 학습하기 어려울 수 있음

-> 위의 단점 때문에 Kolmogorov-Arnold 표현 정리는 기계학습 분야에서 사형 선고를 받음 ( 빚좋은 개살구)
![image](https://github.com/jinuk쳐
