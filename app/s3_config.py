import boto3
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

s3 = boto3.resource(
    service_name="s3",
    region_name="us-east-2",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

s3.Bucket("dummy").upload_file(Filename="database.py", Key="dummy")