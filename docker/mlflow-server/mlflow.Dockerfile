FROM ghcr.io/mlflow/mlflow:v3.1.1

# Install PostgreSQL driver
RUN pip install psycopg2-binary boto3

# Set the default command
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"] 
