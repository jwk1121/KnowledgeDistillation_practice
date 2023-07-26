# KnowledgeDistillation_practice
Knowledge Distillation 기법을 사용한 실습 정리

## 01) Response-based Knowledge
FashionMNIST Dataset을 사용해 soft logits 을 통한 Knowledge transfer 실습을 진행  
Network의 구조는 단순 DNN으로 진행, Teacher model은 6개의 layer, Student model은 1개의 layer로 구성  

$$Loss Fuction = \sum L_{KD}(S(x,\theta_{S}, T ), T(x, \theta_{T}, T)) T^2 \alpha+ L_{CE}(\hat y_{S}, y)(1-\alpha)$$  
위와 같은 Loss를 사용하여 하이퍼파라미터인 T와 $\alpha$ 를 적절히 변경해가며 최적의 하이퍼파라미터를 탐색  

단순 학습을 시켰을 때 Teacher Model's acc: 88.62%, Student Model's acc: 84.24%

|T value|$\alpha$|acc|
|:---:|:---:|:---:|
|2|0.5|84.52|
|3|0.5|83.84|
|5|0.5|82.43|
|7|0.5|81.29|

위의 실험을 통해 T가 2일때 가장 높은 정확도를 보이는 것을 확인  

|$\alpha$|T value|acc|
|:---:|:---:|:---:|
|0.1|2|84.55|
|0.3|2|84.69|
|0.5|2|84.52|
|0.7|2|84.26|

가장 높은 정확도를 보이는 하이퍼 파라미터는 T = 2, $\alpha$ = 0.3 임을 실험을 통해 파악할 수 있음

Distillation을 함으로써 약 0.5%의 정확도가 올라간 것을 확인할 수 있음

## 02) feature-based Knowledge
CIFAR10 Dataset을 사용해 intermediate layer의 output을 통한 Knowledge transfer 실습 진행  
Network의 구조는 Teacher model은 ResNet18, Student model은 3개의 CNN layer with BatchNormaliztion 으로 구성  

$$Loss Fuction = L_{MSE}(\psi_{Student}(x),\psi_{Teacher}(x)) + L_{MSE}(\phi_{Student}(x),\phi_{Teacher}(x))\cdot \alpha  + L_{CE}(\hat y_{s}, y)$$  

$\psi$ 는 첫 번째 Feature map을 출력하는 함수이고,  $\phi$ 는 두 번째 Feature map을 출력하는 함수를 의미  
*가장 초기의 Loss는 Feature Map을 사용하여 MSE loss를 산출했지만 Attention map을 2개를 사용해 MSE Loss를 산출*  

단순 학습을 시켰을 때 Teacher Model's acc: 91.48%, Student Model's acc: 76.10%
