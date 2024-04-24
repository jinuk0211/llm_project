llama 3 8b에서 phi3 3.8b까지 
slm - part 2

Why do small language models underperform? -11 Apr 2024
https://arxiv.org/pdf/2404.07647.pdf
slm 포텐셜에 관한 연구 - 22 Apr 2024
https://arxiv.org/pdf/2404.06395.pdf

수조 개의 파라미터를 가진 대규모 언어 모델(LLM)을 개발하려는 관심이 높아지면서 자원 효율성과 cost에 대한 우려가 커지고 있다. 특히 실험에 드는 엄청난 비용을 고려할 때, 효율적인 대안으로 소규모 언어 모델(SLM)의 잠재력을 탐구하는 것이 더 중요해지고 있다.
 
ex)
iphone 15에서 4bit으로 QLoRA fine-tuning 가능한 llama 8B use case
https://twitter.com/awnihannun/status/1782807288002281916

page 2


대부분의 연구자와 기업에게 있어 대규모 언어모델(LLM)에 대한 실험은 매우 비쌈. 
또 이러한 거대한 모델을 개인 컴퓨터나 스마트폰 등의 일상적인 상황에 배치하는 것은 비효율적이거나 실현 불가능하다.

이러한 두가지 측면이 SLM에 대한 연구에 집중해야할 필요성을 강조함

또한 최근의  Phi 시리즈(Gunasekar et al., 2023; Li et al., 2023b; Javaheripi & Bubeck, 2023), TinyLlama (Zhang et al., 2024a), MobileLLM (Liu et al., 2024), Gemma (Banks & Warkentin, 2024) 등 혁신적인 모델이 등장

하지만 LLM 과 유사한 comprehensive 능력 개발과 (2) SLM과 LLM 모두의 진화를 더욱 촉진할 수 있는 transparent and scalable 훈련 방법론의 숙제가 남음

아래는 iphone15 내장된 A16 bionic chip info



page 5


작은 모델은 일정 수준 이상 훈련을 진행하면 성능이 떨어진 후 정체되는 포화(saturation) 현상이 발생함

모델의 hidden dim과 목표 문맥 확률 분포의 높은 랭크(the high rank of
the target contextual probability distribution) 간의 불일치에서 비롯된다는 것 발견

불일치는 모델의 linear prediction head에서 잘 알려진 softmax bottleneck 현상을 일으켜 성능에 영향을 미침.

우리는 다양한 setting으로 softmax bottleneck 현상의 영향을 측정했고, hidden_layer_dim이 1000 미만인 모델은 late pretrain에서 degraded latent representation을 택하여 평가 성능이 저하되는 것을 발견
