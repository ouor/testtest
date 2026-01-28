import os
import boto3
from botocore.client import Config

bucket_name = os.getenv("R2_BUCKET_NAME")
if not bucket_name:
    raise RuntimeError("R2_BUCKET_NAME is required")

# get s3 client
s3 = boto3.client(
    's3',
    endpoint_url=f'https://{os.getenv("R2_ACCOUNT_ID")}.r2.cloudflarestorage.com',
    aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
    config=Config(signature_version='s3v4'),
)

input('Press Enter to continue...')

file_name = 'test.txt'
# create a test file
with open(file_name, 'w') as f:
    f.write('This is a test file for CloudFlare R2.\n')

# upload file to bucket
s3.upload_file(file_name, bucket_name, file_name)

print(f'Uploaded {file_name} to bucket {bucket_name}')
input('Press Enter to continue...')

# list objects in bucket
objects = s3.list_objects_v2(Bucket=bucket_name)
for k, v in objects.items():
    print(f'{k}: {v}')

input('Press Enter to continue...')

# generate expiring download link
url = s3.generate_presigned_url(
    ClientMethod='get_object',
    Params={
        'Bucket': bucket_name,
        'Key': file_name,
    },
    ExpiresIn=3600*24
)
print(f'Presigned URL: {url}')
input('Press Enter to continue...')

# download file from bucket
s3.download_file(bucket_name, file_name, 'downloaded_' + file_name)

print(f'Downloaded {file_name} from bucket {bucket_name}')
input('Press Enter to exit...')

# delete file from bucket

s3.delete_object(Bucket=bucket_name, Key=file_name)