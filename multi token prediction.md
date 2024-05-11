
Better & Faster Large Language Models via Multi-token Prediction
- 2024년 4월 30일
https://arxiv.org/pdf/2404.19737
next token prediction -> mutliple future token prediction

장점
13B모델로 한 실험에서 humaneval에서는 12 %, MBPP에서는 17% 문제를 더 많이 풀었다( next token 모델과 비교해서)

추론(inference)속도가 3배 더 빠르다 

기존의 cross entrophy loss

기존의 1~t번째까지의 토큰들을 통해 t+1의 next token prediction은 결과로 1개의 토큰을 생성한다.

multi token next prediction의 loss

n 개의 토큰들을 한번에 prediction하기 위해 밑의 loss식을 tractable하게 만들어야 한다.
이를 위해 z (t:1)의 잠재표현(latent representation
