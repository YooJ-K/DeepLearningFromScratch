# 5장 오차역전파법

- 계산 그래프를 이용하면 계산 과정에서 시각적으로 파악할 수 있다.

- 계산 그래프의 노드는 국소적 계산으로 구성된다. 국소적 계산을 조합해 전체 계산을 구성한다.

- 계산 그래프의 순전파는 통상의 계산을 수행한다. 한편, 계산 그래프의 역전파로는 각 노드의 미분을 구성할 수 있다.

- 신경망의 구성 요소를 계층으로 구현하여 기울기를 효율적으로 계산할 수 있다.(오차역전파법)

- 수치 미분과 오차역전파법의 결과를 비교하면 오차역전파법의 구현에 잘못이 없는지 확인할 수 있다.(기울기 확인)


![밑바닥부터 시작하는 딥러닝-8](https://user-images.githubusercontent.com/55521930/153741750-8f62fdf7-f1b2-424c-ad46-14dea7365048.jpg)
![밑바닥부터 시작하는 딥러닝-9](https://user-images.githubusercontent.com/55521930/153741747-24e3f296-5a97-48ec-8f4f-cf68ae1393f8.jpg)
![밑바닥부터 시작하는 딥러닝-10](https://user-images.githubusercontent.com/55521930/155278948-2a7bb35e-3252-40e7-bad2-b3fac2a49964.jpg)
