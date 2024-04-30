![image](https://github.com/jinuk0211/llm_project/assets/150532431/38198a6f-aabb-46e7-9309-9037ddcac6d5)
clinical reasoning capability <- self training, web search integration
multimodal performance <- finetuning, customized encoder

환자 상담,정확한 진단, 치료계획, 공감
검사 이미지와 다른 진단 결과로부터의 multimodal reasoning
임상의는 의사 결정을 내리기 위해 최신 의학 정보 출판물에서 procedure video 까지 계속 공부해야함

이전의 노력
med-PaLM 2는 factuality, reasoning,harm, and bias 측면에서 의사들을 능가(Singhal 등, 2023b). 포텐셜은 Q&A 수준을 넘어선다. 플라밍고-CXR 및 Med-PaLM M과 같은 LMM(Li 등, 2024; Moor 등, 2023b)는 통제된(controlled) 환경에서 방사선 보고서를 생성하는 데 있어 방사선 전문의와 비슷한 수준(Huang 등, 2023; Tanno 등, 2024; Tu 등, 2024a). 더 어려운(challenging) 환자 배우와의 텍스트 기반 진단 상담 환경(가상으로 세팅해서 실험)에서 Articulate Medical Intelligence Explorer (AMIE) 모델은 진단 대화에 대한 여러 평가 기준에서 1차 진료 의사를 능가(Tu 등, 2024b).

이러한 promising한 결과를 보이지만 이외에도 성능 개선을 여지가 매우 많음. LLM은 불확실한 상황에서 적절치 않은 임상 추론을 보이며, 허위 정보(없는거 지어냄 , confabulation)과 bias가 여전히 주요 과제(Omiye 등, 2023; Umapathi 등, 2023). 의료 작업을 수행하기 위한 도구 및 최신 의학 정보 활용(Zakka 등, 2024)과 임상의와의 효과적인 협업(McDuff 등, 2023)은 LLM에 대한 과제로 남아 있습니다. 또한 complex multimodal 의료 데이터(예시: 이미지, 비디오, 환자 개인자료 없앤 의료기록을 통합시키는 과정)를 처리하는 능력도 현재는 제한적 LMM의 성능이 아직 좋지 않음(Tu 등, 2024a). 이러한 능력이 의료 분야에서 특히 의미가 있지만, 의료 LLM의 성능 향상은 의료 분야를 넘어 다른 분야에도 광범위한 영향을 미칠 수 있다. 의료 LLM의 progress을 측정하고 가속화하기 위해 개발된 작업 및 벤치마크는 광범위한 영향력을 가질 것

방법
1. self-training 과 web search integration을 통한 Advanced reasoning
medical notes 요약, 진료의뢰서 생성 등의 덜 복잡한 reasoning task
<- Med-Gemini-M 1.0 : 제미니 finetuning한거
복잡한 reasoning과제
<- Med-Gemini-L 1.0 : 제미니 self-training 방식으로 finetuning 한거
 웹에서 서치할 수 있게해 self training 가능하게 함, 추론 시 novel uncertainty-guided search strategy를 개발해냄 <- 복잡한 clinical reasoning task를 수행할 수 있게 하기위해

제미니 자체 ㅅㄹ

![image](https://github.com/jinuk0211/llm_project/assets/150532431/2ff69c1a-aeb9-455b-9889-b784f8b83c47)
