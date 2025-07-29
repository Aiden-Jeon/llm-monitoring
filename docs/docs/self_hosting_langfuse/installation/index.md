---
sidebar_position: 1
---

# 01-Installation

Self hosting 하는 langfuse 방법에 대해서 설명합니다.  
해당 문서에서는 langfuse 3 버전을 기준으로 설명합니다.

## Remote Server
### Step 1: langfuse 실행을 위한 docker-compose 파일 작성
이후 실행 재현을 위해서 각 백엔드 버전을 고정하였으며 최신 버전의 langfuse 실행은 [공식 레포지토리의 docker-compose](https://github.com/langfuse/langfuse/blob/main/docker-compose.yml) 에서 확인할 수 있습니다.

고정된 버전은 다음과 같습니다:
- `langfuse/langfuse-worker:3`
- `langfuse/langfuse:3`
- `postgres:17`
- `clickhouse/clickhouse-server:23.8-alpine`

해당 튜토리얼에서는 mlflow 환경과 같이 실행하기 위해서 포트 정보가 변경되어 있습니다.

- `docker-compose.yaml`
    
    ```yaml
    # Make sure to update the credential placeholders with your own secrets.
    # We mark them with # CHANGEME in the file below.
    # In addition, we recommend to restrict inbound traffic on the host to langfuse-web (port 3000) and minio (port 9090) only.
    # All other components are bound to localhost (127.0.0.1) to only accept connections from the local machine.
    # External connections from other machines will not be able to reach these services directly.
    services:
      langfuse-worker:
        image: langfuse/langfuse-worker:3
        restart: always
        depends_on: &langfuse-depends-on
          postgres:
            condition: service_healthy
          minio:
            condition: service_healthy
          redis:
            condition: service_healthy
          clickhouse:
            condition: service_healthy
        ports:
          - 127.0.0.1:3030:3030
        environment: &langfuse-worker-env
          DATABASE_URL: postgresql://postgres:postgres@postgres:5432/postgres # CHANGEME
          SALT: "mysalt" # CHANGEME
          ENCRYPTION_KEY: "0000000000000000000000000000000000000000000000000000000000000000" # CHANGEME: generate via `openssl rand -hex 32`
          TELEMETRY_ENABLED: ${TELEMETRY_ENABLED:-true}
          LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES: ${LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES:-true}
          CLICKHOUSE_MIGRATION_URL: ${CLICKHOUSE_MIGRATION_URL:-clickhouse://clickhouse:9000}
          CLICKHOUSE_URL: ${CLICKHOUSE_URL:-http://clickhouse:8123}
          CLICKHOUSE_USER: ${CLICKHOUSE_USER:-clickhouse}
          CLICKHOUSE_PASSWORD: ${CLICKHOUSE_PASSWORD:-clickhouse} # CHANGEME
          CLICKHOUSE_CLUSTER_ENABLED: ${CLICKHOUSE_CLUSTER_ENABLED:-false}
          LANGFUSE_USE_AZURE_BLOB: ${LANGFUSE_USE_AZURE_BLOB:-false}
          LANGFUSE_S3_EVENT_UPLOAD_BUCKET: ${LANGFUSE_S3_EVENT_UPLOAD_BUCKET:-langfuse}
          LANGFUSE_S3_EVENT_UPLOAD_REGION: ${LANGFUSE_S3_EVENT_UPLOAD_REGION:-auto}
          LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID: ${LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID:-minio}
          LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY: ${LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY:-miniosecret} # CHANGEME
          LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT: ${LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT:-http://minio:9000}
          LANGFUSE_S3_EVENT_UPLOAD_FORCE_PATH_STYLE: ${LANGFUSE_S3_EVENT_UPLOAD_FORCE_PATH_STYLE:-true}
          LANGFUSE_S3_EVENT_UPLOAD_PREFIX: ${LANGFUSE_S3_EVENT_UPLOAD_PREFIX:-events/}
          LANGFUSE_S3_MEDIA_UPLOAD_BUCKET: ${LANGFUSE_S3_MEDIA_UPLOAD_BUCKET:-langfuse}
          LANGFUSE_S3_MEDIA_UPLOAD_REGION: ${LANGFUSE_S3_MEDIA_UPLOAD_REGION:-auto}
          LANGFUSE_S3_MEDIA_UPLOAD_ACCESS_KEY_ID: ${LANGFUSE_S3_MEDIA_UPLOAD_ACCESS_KEY_ID:-minio}
          LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY: ${LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY:-miniosecret} # CHANGEME
          LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT: ${LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT:-http://localhost:9092}
          LANGFUSE_S3_MEDIA_UPLOAD_FORCE_PATH_STYLE: ${LANGFUSE_S3_MEDIA_UPLOAD_FORCE_PATH_STYLE:-true}
          LANGFUSE_S3_MEDIA_UPLOAD_PREFIX: ${LANGFUSE_S3_MEDIA_UPLOAD_PREFIX:-media/}
          LANGFUSE_S3_BATCH_EXPORT_ENABLED: ${LANGFUSE_S3_BATCH_EXPORT_ENABLED:-false}
          LANGFUSE_S3_BATCH_EXPORT_BUCKET: ${LANGFUSE_S3_BATCH_EXPORT_BUCKET:-langfuse}
          LANGFUSE_S3_BATCH_EXPORT_PREFIX: ${LANGFUSE_S3_BATCH_EXPORT_PREFIX:-exports/}
          LANGFUSE_S3_BATCH_EXPORT_REGION: ${LANGFUSE_S3_BATCH_EXPORT_REGION:-auto}
          LANGFUSE_S3_BATCH_EXPORT_ENDPOINT: ${LANGFUSE_S3_BATCH_EXPORT_ENDPOINT:-http://minio:9000}
          LANGFUSE_S3_BATCH_EXPORT_EXTERNAL_ENDPOINT: ${LANGFUSE_S3_BATCH_EXPORT_EXTERNAL_ENDPOINT:-http://localhost:9092}
          LANGFUSE_S3_BATCH_EXPORT_ACCESS_KEY_ID: ${LANGFUSE_S3_BATCH_EXPORT_ACCESS_KEY_ID:-minio}
          LANGFUSE_S3_BATCH_EXPORT_SECRET_ACCESS_KEY: ${LANGFUSE_S3_BATCH_EXPORT_SECRET_ACCESS_KEY:-miniosecret} # CHANGEME
          LANGFUSE_S3_BATCH_EXPORT_FORCE_PATH_STYLE: ${LANGFUSE_S3_BATCH_EXPORT_FORCE_PATH_STYLE:-true}
          LANGFUSE_INGESTION_QUEUE_DELAY_MS: ${LANGFUSE_INGESTION_QUEUE_DELAY_MS:-}
          LANGFUSE_INGESTION_CLICKHOUSE_WRITE_INTERVAL_MS: ${LANGFUSE_INGESTION_CLICKHOUSE_WRITE_INTERVAL_MS:-}
          REDIS_HOST: ${REDIS_HOST:-redis}
          REDIS_PORT: ${REDIS_PORT:-6379}
          REDIS_AUTH: ${REDIS_AUTH:-myredissecret} # CHANGEME
          REDIS_TLS_ENABLED: ${REDIS_TLS_ENABLED:-false}
          REDIS_TLS_CA: ${REDIS_TLS_CA:-/certs/ca.crt}
          REDIS_TLS_CERT: ${REDIS_TLS_CERT:-/certs/redis.crt}
          REDIS_TLS_KEY: ${REDIS_TLS_KEY:-/certs/redis.key}
    
      langfuse-web:
        image: langfuse/langfuse:3.83
        restart: always
        depends_on: *langfuse-depends-on
        ports:
          - 3000:3000
        environment:
          <<: *langfuse-worker-env
          NEXTAUTH_URL: http://localhost:3000
          NEXTAUTH_SECRET: mysecret # CHANGEME
          LANGFUSE_INIT_ORG_ID: ${LANGFUSE_INIT_ORG_ID:-}
          LANGFUSE_INIT_ORG_NAME: ${LANGFUSE_INIT_ORG_NAME:-}
          LANGFUSE_INIT_PROJECT_ID: ${LANGFUSE_INIT_PROJECT_ID:-}
          LANGFUSE_INIT_PROJECT_NAME: ${LANGFUSE_INIT_PROJECT_NAME:-}
          LANGFUSE_INIT_PROJECT_PUBLIC_KEY: ${LANGFUSE_INIT_PROJECT_PUBLIC_KEY:-}
          LANGFUSE_INIT_PROJECT_SECRET_KEY: ${LANGFUSE_INIT_PROJECT_SECRET_KEY:-}
          LANGFUSE_INIT_USER_EMAIL: ${LANGFUSE_INIT_USER_EMAIL:-}
          LANGFUSE_INIT_USER_NAME: ${LANGFUSE_INIT_USER_NAME:-}
          LANGFUSE_INIT_USER_PASSWORD: ${LANGFUSE_INIT_USER_PASSWORD:-}
    
      clickhouse:
        image: clickhouse/clickhouse-server:23.8-alpine
        restart: always
        user: "101:101"
        environment:
          CLICKHOUSE_DB: default
          CLICKHOUSE_USER: clickhouse
          CLICKHOUSE_PASSWORD: clickhouse # CHANGEME
        volumes:
          - langfuse_clickhouse_data:/var/lib/clickhouse
          - langfuse_clickhouse_logs:/var/log/clickhouse-server
        ports:
          - 127.0.0.1:8123:8123
          - 127.0.0.1:9100:9000
        healthcheck:
          test: wget --no-verbose --tries=1 --spider http://localhost:8123/ping || exit 1
          interval: 5s
          timeout: 5s
          retries: 10
          start_period: 1s
    
      minio:
        image: minio/minio
        restart: always
        entrypoint: sh
        # create the 'langfuse' bucket before starting the service
        command: -c 'mkdir -p /data/langfuse && minio server --address ":9000" --console-address ":9001" /data'
        environment:
          MINIO_ROOT_USER: minio
          MINIO_ROOT_PASSWORD: miniosecret # CHANGEME
        ports:
          - 127.0.0.1:9092:9000
          - 127.0.0.1:9093:9001
        volumes:
          - langfuse_minio_data:/data
        healthcheck:
          test: ["CMD", "mc", "ready", "local"]
          interval: 1s
          timeout: 5s
          retries: 5
          start_period: 1s
    
      redis:
        image: redis:7
        restart: always
        # CHANGEME: row below to secure redis password
        command: >
          --requirepass ${REDIS_AUTH:-myredissecret}
        ports:
          - 127.0.0.1:6379:6379
        healthcheck:
          test: ["CMD", "redis-cli", "ping"]
          interval: 3s
          timeout: 10s
          retries: 10
    
      postgres:
        image: postgres:15-alpine
        restart: always
        healthcheck:
          test: ["CMD-SHELL", "pg_isready -U postgres"]
          interval: 3s
          timeout: 3s
          retries: 10
        environment:
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: postgres # CHANGEME
          POSTGRES_DB: postgres
        ports:
          - 127.0.0.1:5433:5432
        volumes:
          - langfuse_postgres_data:/var/lib/postgresql/data
    
    volumes:
      langfuse_postgres_data:
        driver: local
      langfuse_clickhouse_data:
        driver: local
      langfuse_clickhouse_logs:
        driver: local
      langfuse_minio_data:
        driver: local
    ```

### Step 2: langfuse 초기 설정을 위한 환경 변수 작성
초기 설정을 위한 env file 을 작성합니다.

- `init.env`
  ```env
  LANGFUSE_INIT_ORG_ID=my-org
  LANGFUSE_INIT_ORG_NAME=My Org
  LANGFUSE_INIT_PROJECT_ID=my-project
  LANGFUSE_INIT_PROJECT_NAME=My Project
  LANGFUSE_INIT_PROJECT_PUBLIC_KEY=lf_pk_1234567890
  LANGFUSE_INIT_PROJECT_SECRET_KEY=lf_sk_1234567890
  LANGFUSE_INIT_USER_EMAIL=user@example.com
  LANGFUSE_INIT_USER_NAME=John Doe
  LANGFUSE_INIT_USER_PASSWORD=password123
  ```


### Step 3: 실행

위 두 파일을 작성 후 실행합니다.
```bash
 docker compose --env-file init.env up -d
```

실행 후 [http://localhost:3000/](http://localhost:3000/) 에 접속해서 화면을 확인합니다.

![img](./self_hosting_langfuse_0.png)

초기 접속 정보는 Step 2 에서 작성한 정보입니다.

로그인 후 생성되어 있는 My Org 에 접속하면 다음과 같은 langfuse 화면을 확인할 수 있습니다.

![img](./self_hosting_langfuse_1.png)
