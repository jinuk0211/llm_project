새로운 뉴럴넷 , 딥러닝의 FFN의 대체제 - KAN

Kolmogorov–Arnold Networks
KAN이 학습한 수식(simple symbolic task)들의 network
![image](https://github.com/jinuk0211/llm_project/assets/150532431/f47d95b3-58fb-45ca-938a-44b8074d5d11)

https://arxiv.org/pdf/2404.19756 - 2024년 4월 30일자 논문

수학(매듭 이론)과 물리학(앤더슨 localization)의 2가지 예시를 활용하여 KAN이 과학자들이 수학과 물리 법칙을 (재)발견하는 데 있어 도움이 되는 "협력자"가 될 수 있음을 보여줌

다층 퍼셉트론(Multi-layer perceptrons, MLPs) 은 fully connected feed forward 신경망으로 알려져 있으며, 오늘날 딥러닝 모델의 근간이 되는 구성 요소. MLPs의 중요성은 과소평가될 수 없는데, 왜냐하면 이들은 범용 근사 정리(universal approximation theorem)에 의해 보장된 표현력(representation) 때문에 비선형 함수를 근사화하는 machine leraning의 기본 모델이기 때문 하지만 MLPs가 우리가 구축할 수 있는 최고의 비선형 회귀 모델인가?에 대한 의문점이 여전히 존재
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


![image](https://github.com/jinuk0211/llm_project/assets/150532431/e275b829-cb34-47e5-ab9c-e3454e6216e3)

수학
만약의 근사함수 f(x)가 다변수 연속함수(on a bounded domain)라면 
이는 단일 변수, 이항연산만의 연속함수를 유한하게 composition 하는 것으로 계산할 수 있다
->
고차원 함수 f(x)를 학습하는 것은 여러개(polynomial number)의 1차원 함수를 학습하는 것으로 귀결됨.


실제 예시
![image](https://github.com/jinuk0211/llm_project/assets/150532431/f1e8e8b6-ce22-417a-8378-4cd1f06336ac)

but
이러한 1차원 함수들은 매끄럽지 않고(non smooth) 심지어 프랙탈(fractal) 구조를 가질 수 있기 때문에 실제로는 학습하기 어려울 수 있음

-> 위의 단점 때문에 Kolmogorov-Arnold 표현 정리는 기계학습 분야에서 사형 선고를 받음 ( 빚좋은 개살구)

KAN 아키텍쳐
![KAN layer 1D function](https://github.com/jinuk0211/llm_project/assets/150532431/270a0a7f-1519-43d2-a8a5-cc2090d9eb85)
MLP에서는 한 번에 선형 변환(nn.linear)과 비선형성(activation layer)으로 이루어진 레이어를 정의한 후에 더 많은 레이어를 쌓아 신경망을 더 깊게 만들 수 있음.하지만 KAN은 <-? 깊은 KAN을 구축하기 위해서는 먼저 "KAN 레이어는 무엇인가?"라는 질문에 대답해야 한다. 결과적으로, nin(input_dim) 차원의 입력과 nout(output_dim) 차원의 출력을 갖는 KAN 레이어는 1D 함수의 행렬로 정의될 수 있다.
위의 사례를 보자면 inner function 의 nin은 n, nout은 2n+1이고 outer function의 nin은 2n+1이고 nout은 1이된다. 이게 Kolmogorov-Arnold representations이고 MLP와 똑같이 layer를 더 깊게 쌓으면 됨
(2.2) 각각의 1D function을 B spline curve로 파라미터화시킨다.
![kAN activation](https://github.com/jinuk0211/llm_project/assets/150532431/7edb6fec-3d79-4c58-bb78-058718906711)
대충 phi함수 정의하고 몇번째 layer고 몇번째 input이고 등등 어렵게 수식으로 정의

![image](https://github.com/jinuk0211/llm_project/assets/150532431/f8daf9f9-d924-41b2-b944-367864d6e09e)
![Uploading image.png…]()


approximation ability
![knn approximation ability](https://github.com/jinuk0211/llm_project/assets/150532431/de81738f-04d1-4c0d-abb0-7391059a1cc8)
![knn approximation ability](https://github.com/jinuk0211/llm_project/assets/150532431/d298f348-9714-4677-b104-cb466a11f52f)

grid extension
![image](https://github.com/jinuk0211/llm_project/assets/150532431/30b95b23-4d87-431f-b1d5-944c1c55a71a)

FFN의 L2 regularization을 KAN에 적용
![image](https://github.com/jinuk0211/llm_project/assets/150532431/ec6dc43e-5c14-4503-a437-7590cf4f1a8a)

simplification 기술들 
![image](https://github.com/jinuk0211/llm_project/assets/150532431/30cf9fc6-2f53-4f4b-9412-04a1526b7eea)


interpretable 클릭 한번
![image](https://github.com/jinuk0211/llm_project/assets/150532431/d923727b-6e8b-486d-89cd-7eaef28f5cbf)
![스크린샷 2024-05-02 163113](https://github.com/jinuk0211/llm_project/assets/150532431/32d48c2d-2855-413d-8bed-5ec0f58c3b7f)

