https://www.youtube.com/watch?v=UTuuTTnjxMQ&t=126s
gemini 만든 사람 중 한명 - sholto ( google deepmind ) - 현재 추론 관련 연구
claude 만든 사람 중 한명 - trenton ( anthropic ai, 뇌과학과 AI 사이의 수렴이란 연구를 했음
 ) - 현재 AI interpretation ? 관련 연구
 
large 모델과 small 모델 사이의 경계가 흐릿해지고 있음

뇌와 AI의 association? 의 같은 방식을 사용

finetune이 별로 필요가 없어질 것 - 지금의 특정 task별로 리더보드마다의 SOTA가 1개의 독립적인 모델로
<-- MoD, MoE 같은 연구 같음

long context - infini attention 
만약에 ai에 프론트엔드 백엔드 코드 다 넣고 학습을 진행해 어떤 결과를 얻을 수 있다면
어떤 영화를 학습데이터로 넣어서 뭔가를 얻어낼수 있다면
 
gpt4의 파라미터개수가 약 1조로 추정되는데 인간의 뇌 synopses 수는 30조~300조 (one to one mapping이 아니고 숫자로 토론을 할수는 없지만) 여전히 brain scale 밑이다 (뉴런 100b)
openai의 7T dollars, 9경 프로젝트처럼 국가 레벨을 넘어가는 magnitude의 ai가, 만약에 수십조의 파라미터를 가지는 ai를 돌릴 수 있게 된다면 ?

reasoning이 무엇인가 - 
if 사람이 어떤 게임을 보고 그 게임하는 방법을 깨닫는다 -> 뉴럴넷으로 설명이 가능한가
trenton : 수많은 reasoning circuit(수많은 정보를 연결시키는)을 연결시키는 것, 이미지의 픽셀로 물체들을 latent representaion으로 변환시키고, physics를 배우는 서킷등의 casual 서킷들의 집합
trenton : 
if induction head에 적용시켜보면 짐과 메리가 가게에 갔다. 

Ztrenton : 정해진 숫자의 forward pass가 있다 생각한다. 보통 언어 같은 경우 무한한 forward pass가 있다 생각하지만
"소년이 곰 위로 점프했다. 그 곰은 꿀을 먹고 있었고 낮잠을 잤었는데 배를 긁적였었다" 실제로는 5~7번 정도의 recursion이 끝이다

2. intelligence에 대한 토론
trenton : 뇌의 70퍼센트의 뉴런이 소뇌(cellebrium)있는데 next token prediction 때 이가 매우 활성화 되는 것을 PET데이터에서 확인 가능했다
trenton : trenton의견 : hot take 정도, 지능이란 pattern matching으로 연관된 기억들의 계층을 갖고 있다면 그에 기반해서 가장 적합한 pattern matching을 수행하는 것일 것이다.

feature space
direction and activation space

monosemanticity 로의 연구(흔히 black box라 불리는 뉴럴넷에 대한 interpretation)
새인지 아닌지를 학습하는 것, 혹은 독수리인지 참새인지 까마귀인지, 더 나아가 사랑, 속임수, 
수학공식, 복잡한 proof를 holding 하는 것, 이 모든게 다 feature 인가
trenton : 만약 우리가 찾은 feature가 예측 불가능, 데이터가 인과관계가 없거나 또는 그냥 representation, 데이터의 대표적인 무언가일 때 우리는 단지 데이터를 clustering 모으고
 
sholto :예를 들어 F=ma 식에서 f를 구하는 과정이 일종의 reasoning이라 치면 질량과 가속도를 구해서 힘에 대한 어떤 값을 얻어내는 것 high-level association 갖고 있는 이러한 feature들을 통해 F란 것을 얻어낼수 있음.
