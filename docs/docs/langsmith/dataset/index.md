---
sidebar_position: 4
---


# 03-Dataset

## 개요

LangSmith를 사용하여 데이터셋을 생성하고 관리하는 방법에 대해 설명합니다.
LangSmith의 Dataset 기능을 사용하여 평가용 데이터를 중앙에서 관리하고 버전을 추적할 수 있습니다. LangSmith는 LangChain에서 공식적으로 제공하는 개발자 플랫폼으로, 데이터셋 관리 기능도 제공합니다.

튜토리얼에서 사용하는 코드는
[Github](https://github.com/Aiden-Jeon/llm-monitoring/blob/main/notebooks/langsmith/03_dataset.ipynb)
에서 확인할 수 있습니다.

## Requirements

### 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 필요한 환경 변수를 설정합니다.
:::info
[환경 변수 설정](../tracing/#Environments) 을 참조해 설정합니다.
:::

## Code

### Environments

#### 1. 환경 변수 로드

실행을 위해 필요한 환경 변수를 불러옵니다.

```python
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv(dotenv_path=".env", override=True)
```

#### 2. LangSmith 클라이언트 설정

LangSmith 클라이언트를 초기화하고 데이터셋 이름을 지정합니다.

```python
from langsmith import Client

client = Client()
dataset_name = "dataset-exmaple"
```

## Dataset Management

### 1. 데이터셋 생성

LangSmith에 데이터셋을 생성합니다. 이미 존재하는 경우 기존 데이터셋을 사용합니다.

```python
# Create dataset in LangSmith
try:
    # Try to read the dataset first to see if it already exists
    existing_dataset = client.read_dataset(dataset_name=dataset_name)
    print(
        f"Dataset '{dataset_name}' already exists with {len(list(client.list_examples(dataset_name=dataset_name)))} examples"
    )
except:
    # Dataset doesn't exist, create it
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Dataset for evaluating AI explanations",
    )
```

### 2. 예제 데이터 생성

평가용 예제 데이터를 정의하고 LangSmith 형식으로 변환합니다.

```python
records = [
    {
        "input_question": "What is sound?",
        "output_output": "Okay! Imagine you have a drum. When you hit it, the drum shakes, right? That shaking is called a vibration. Now, when the drum shakes, it makes the air around it wiggle too, almost like when you throw a rock in a pond and the water makes little waves. These wiggles in the air are what we call sound! So, sound is like little invisible wiggles in the air that our ears can feel and hear. Just like how you can hear your friend calling you or your favorite song, those are sounds!",
    },
    {
        "input_question": "How does string theory work?",
        "output_output": "Okay! Imagine that everything in the universe, like stars, trees, and even you, is made of tiny building blocks, kind of like LEGO bricks. Now, in string theory, instead of those building blocks being tiny dots, they are super tiny strings, like the strings on a guitar! These strings can wiggle and vibrate. Picture a guitar playing different notes – when a string vibrates in different ways, it makes different sounds. In the same way, when these tiny strings vibrate differently, they become different things, like particles that make up everything you see around you. So, when the strings wiggle and dance around in space, they interact with each other – sometimes they might hug and combine, and other times they might bounce off each other, just like friends playing together. When we look closely at these strings, they act like the little particles we learn about in science. In short, string theory is like saying that everything is made of tiny, wiggly strings, and how they dance and vibrate tells us all about the world we live in!",
    },
    {
        "input_question": "How does a democracy work?",
        "output_output": "Okay! Imagine you and your friends want to decide what game to play. Instead of just one person deciding, everyone gets a turn to say what they want. In a democracy, it's like that, but for the whole country! Here's how it works: 1. **Everyone Has a Voice**: Just like you and your friends all get to say what game you want, in a democracy, everyone gets to share their opinions. This means that everyone gets to vote on important things. 2. **Voting**: When it's time to make a decision, like picking a leader or a new law, everyone votes. Voting is like marking your favorite game on a piece of paper. The choice with the most votes wins! 3. **Leaders**: The people who get the most votes (like class leaders) are called leaders too! They help make the rules and decisions for everyone. 4. **Fairness**: Everyone is treated equally, so whether you're big or small, everyone's voice is important! 5. **Change is Possible**: If people don't like something, they can vote again later or even choose new leaders. It's like changing the game if everyone wants to play something different. So, in a democracy, people work together to decide what's best for everyone by talking, voting, and making sure everyone can be heard!",
    },
]

# Convert to LangSmith format
examples = []
for row in records:
    examples.append(
        {
            "inputs": {"question": row["input_question"]},
            "outputs": {"output": row["output_output"]},
        }
    )
```

### 3. 데이터셋에 예제 추가

변환된 예제 데이터를 데이터셋에 추가합니다.

```python
# Add examples to the dataset
client.create_examples(
    inputs=[ex["inputs"] for ex in examples],
    outputs=[ex["outputs"] for ex in examples],
    dataset_id=dataset.id,
)

print(
    f"Successfully created dataset '{dataset_name}' with {len(examples)} examples"
)
```

### 4. 업로드된 데이터셋 확인

생성된 데이터셋을 확인합니다.

```python
existing_dataset = client.read_dataset(dataset_name=dataset_name)
existing_dataset
```

## LangSmith UI에서 Dataset 확인

1. 브라우저에서 [LangSmith](https://smith.langchain.com/)에 접속합니다.
2. 로그인 후 프로젝트를 선택합니다.
3. 왼쪽 사이드바에서 "Datasets & Experiments" 탭을 클릭합니다.
4. 생성된 데이터셋 목록을 확인할 수 있습니다.
    ![img](langsmith_0.png)
5. 각 데이터셋을 클릭하여 상세 정보와 예제들을 확인할 수 있습니다.
    ![img](langsmith_1.png)
