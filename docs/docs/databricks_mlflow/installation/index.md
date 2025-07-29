---
sidebar_position: 1
---

# 01-Installation

Managed mlflow 는 여러 서비스에서 사용할 수 있습니다.
[공식 문서](https://mlflow.org/docs/latest/genai/#running-anywhere)에 언급된 managed 서비스를 제공하는 회사는 다음과 같습니다.
- [Databricks](https://docs.databricks.com/aws/en/mlflow3/genai/)
- [Amazon Sagemaker](https://aws.amazon.com/ko/sagemaker-ai/experiments/)
- [Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow?view=azureml-api-2)
- [Nebius](https://nebius.com/services/managed-mlflow)
- Kubernetes

이번 튜토리얼에서는 Databricks Free Edition 에서 제공하는 Managed Mlflow 를 사용합니다.

## Databricks Mlflow

### Step 1: Free edition
Free Edition 가입을 위해 [https://www.databricks.com/learn/free-edition](https://www.databricks.com/learn/free-edition) 으로 이동합니다.

![img](databricks_mlflow_0.png)

### Step 2: Sign Up
이메일을 이용해 손쉽게 가입할 수 있습니다.
![img](databricks_mlflow_1.png)

### Step 3: Databricks Workspace
데이터브릭스에 접속하면 다음과 같은 화면이 나옵니다.
![img](databricks_mlflow_2.png)

### Step 4: Expeirments Tab
화면 왼쪽에서 Experiments 탭을 클릭해 mlflow 를 사용할 수 있습니다.
![img](databricks_mlflow_3.png)

## API Key 발급

데이터브릭스 워크스페이스에서 필요한 환경 변수들을 생성합니다.
### Databricks Host
Databricks host 는 다음과 같은 규칙으로 설정됩니다.
```plaintext
https://<UNIQUE_ID>.cloud.databricks.com
```
예를 들어서 제가 생성한 workspace 의 URL 은 다음과 같습니다.
- https://dbc-a3ca0892-0f44.cloud.databricks.com/

### Experiment
우선 실험을 로깅할 experiment 를 생성합니다.
1. 우선 Experiments tab 에서 Custom model training 을 선택합니다.
    ![img](databricks_mlflow_3.png)
2. 이름을 입력하고 experiement 를 생성합니다.
    ![img](databricks_mlflow_4.png)
3. 오른쪽 위의 New run 버튼을 눌러 Experiment ID 를 확인합니다.
    ![img](databricks_mlflow_5.png)

### DATABRICKS TOKEN
다음으로 databricks token 을 발급받습니다.

1. 오른쪽 프로필 아이콘을 클릭해 Settings 로 진입합니다.
    ![img](databricks_mlflow_6.png)
2. User > Developer > Access tokens 의 Manage 를 클릭합니다.
    ![img](databricks_mlflow_7.png)
3. Generate new token 을 눌러서 새로운 API Key 를 발급 받고 저장합니다.
    ![img](databricks_mlflow_8.png)
