---
sidebar_position: 1
---

# Self-Hosting Mlflow

Self hosting 하는 mlflow 방법에 대해서 설명합니다.  
해당 문서에서는 mlflow 3.1.1 버전을 기준으로 설명합니다.

## Local

개발하는 환경과 같은 곳에서 사용할 경우 간단하게 local 실행을 할 수 있습니다.
실행 환경은 [공식 문서](https://mlflow.org/docs/latest/genai/getting-started/connect-environment/#setup-instructions)를 참조하였습니다.

### Step 1: Install MLflow
Install MLflow for local development:
```bash
pip install --upgrade "mlflow>=3.1"
```

### Step 2: Start MLflow Tracking Server
```bash
# Start MLflow server (in a separate terminal)
mlflow server --host 0.0.0.0 --port 5000
```

## Remote Server

리모트 서버에서 mlflow 를 실행할 경우 아래 Docker Compose 를 이용해 실행할 수 있습니다.
해당 튜토리얼에서는 langfuse 환경과 같이 실행하기 위해서 포트 정보가 변경되어 있습니다.

### Step 1: mlflow 를 실행하는 Dockerfile 작성
- `mlflow.Dockerfile`
    ```docker
    FROM ghcr.io/mlflow/mlflow:v3.1.1

    # Install PostgreSQL driver
    RUN pip install psycopg2-binary boto3

    # Set the default command
    CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"] 

    ```

### Step 2: backend 와 같이 실행하는 docker-compose 작성
- `docker-compose.yaml`
    ```yaml
    services:
        # PostgreSQL database
        postgres:
        image: postgres:latest
        environment:
            POSTGRES_USER: user
            POSTGRES_PASSWORD: password
            POSTGRES_DB: mlflowdb
        ports:
            - 5432:5432
        volumes:
            - ./postgres-data:/var/lib/postgresql/data
        healthcheck:
            test: ["CMD-SHELL", "pg_isready -U user -d mlflowdb"]
            interval: 5s
            timeout: 5s
            retries: 5
        # MinIO server
        minio:
        image: minio/minio
        expose:
            - "9000"
        ports:
            - "9000:9000"
            # MinIO Console is available at http://localhost:9001
            - "9001:9001"
        environment:
            MINIO_ROOT_USER: "minio_user"
            MINIO_ROOT_PASSWORD: "minio_password"
        healthcheck:
            test: timeout 5s bash -c ':> /dev/tcp/127.0.0.1/9000' || exit 1
            interval: 1s
            timeout: 10s
            retries: 5
        command: server /data --console-address ":9001"
        # Create a bucket named "bucket" if it doesn't exist
        minio-create-bucket:
        image: minio/mc
        depends_on:
            minio:
            condition: service_healthy
        entrypoint: >
            bash -c "
            mc alias set minio http://minio:9000 minio_user minio_password &&
            if ! mc ls minio/bucket; then
            mc mb minio/bucket
            else
            echo 'bucket already exists'
            fi
            "
        mlflow:
        build:
            context: .
            dockerfile: mlflow.Dockerfile
        environment:
            MLFLOW_S3_ENDPOINT_URL: http://minio:9000
            AWS_ACCESS_KEY_ID: minio_user
            AWS_SECRET_ACCESS_KEY: minio_password
        ports:
            - 5000:5000
        volumes:
            - ./mlflow-data:/mlflow-data
        depends_on:
            postgres:
            condition: service_healthy
            minio-create-bucket:
            condition: service_completed_successfully
        command: >
            mlflow server
            --backend-store-uri postgresql://user:password@postgres:5432/mlflowdb
            --artifacts-destination s3://bucket
            --host 0.0.0.0
            --port 5000
    ```

### Step 3: 실행

위 두 파일을 작성 후 실행합니다.
```bash
docker compose up -d
```

실행 후 [http://localhost:5000/](http://localhost:5000/) 에 접속해서 화면을 확인합니다.

![img](./self_hosting_mlflow_0.png)
