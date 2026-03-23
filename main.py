import mlflow
from dotenv import load_dotenv

load_dotenv()


def main():
    print("Starting MLflow project...")
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.openai.autolog()


if __name__ == "__main__":
    main()
