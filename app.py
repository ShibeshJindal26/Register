import os
import cv2
import boto3
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Specify the region
region_name = 'us-west-1'
# Initialize Boto3 clients
s3 = boto3.client('s3', region_name=region_name)
rekognition = boto3.client('rekognition', region_name=region_name)
dynamodb = boto3.client('dynamodb', region_name=region_name)

# Capture an image using OpenCV
def capture_image():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Failed to open camera")
        return None

    cv2.namedWindow("Capture")
    print("Camera opened, press SPACE to capture image, ESC to exit")

    img_name = "captured_image.jpg"
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Capture", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            cv2.imwrite(img_name, frame)
            print(f"{img_name} written!")
            break

    cam.release()
    cv2.destroyAllWindows()
    return img_name

# Upload the captured image to S3
def upload_image_to_s3(image_path, bucket_name, s3_key, full_name):
    try:
        with open(image_path, 'rb') as file:
            s3.put_object(Bucket=bucket_name, Key=s3_key, Body=file, Metadata={'fullname': full_name})
        print(f"Image uploaded to {bucket_name}/{s3_key}")
    except Exception as e:
        print(f"Failed to upload image to S3: {e}")

# Index faces using Amazon Rekognition
def index_faces(bucket, key):
    response = rekognition.index_faces(
        Image={"S3Object": {"Bucket": bucket, "Name": key}},
        CollectionId="family_collection"
    )
    return response

# Update DynamoDB with faceId and fullName
def update_index(tableName, faceId, fullName):
    response = dynamodb.put_item(
        TableName=tableName,
        Item={
            'RekognitionId': {'S': faceId},
            'FullName': {'S': fullName}
        }
    )
    return response

# Search for faces using Amazon Rekognition
def search_faces_by_image(bucket, key):
    response = rekognition.search_faces_by_image(
        CollectionId="family_collection",
        Image={"S3Object": {"Bucket": bucket, "Name": key}},
        MaxFaces=1,
        FaceMatchThreshold=95
    )
    return response

# Lambda handler function
def lambda_handler(event, context):
    print(event)
    # Get the object from the event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    try:
        # Check if face already exists in the collection
        search_response = search_faces_by_image(bucket, key)

        if search_response['FaceMatches']:
            face_id = search_response['FaceMatches'][0]['Face']['FaceId']
            print(f"Face already registered with FaceId: {face_id}")
            return {'message': 'Face already registered', 'FaceId': face_id, 'ResponseMetadata': {'HTTPStatusCode': 200}}

        # Calls Amazon Rekognition IndexFaces API to detect faces in S3 object
        # to index faces into specified collection
        response = index_faces(bucket, key)

        # Commit faceId and full name object metadata to DynamoDB
        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            face_id = response['FaceRecords'][0]['Face']['FaceId']

            ret = s3.head_object(Bucket=bucket, Key=key)
            person_full_name = ret['Metadata']['fullname']

            update_index('family_collection', face_id, person_full_name)

        # Print response to console
        print(response)
        return response
    except Exception as e:
        print(e)
        print(f"Error processing object {key} from bucket {bucket}.")
        raise e


@app.route('/register', methods=['POST'])
def capture():
    bucket_name = 'face1-bucket'
    s3_key = 'captured_image.jpg'
    user_name = request.form.get('Username')
    if not user_name:
        return jsonify({'error': 'Full name is required'}), 400

    img_name = capture_image()
    if img_name:
        upload_image_to_s3(img_name, bucket_name, s3_key, user_name)
        
        # Invoke the lambda function
        event = {
            'Records': [{
                's3': {
                    'bucket': {'name': bucket_name},
                    'object': {'key': s3_key}
                }
            }]
        }
        response = lambda_handler(event, None)

        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            if 'message' in response and response['message'] == 'Face already registered':
                return jsonify({'message': 'You have already registered'}), 200
            else:
                return jsonify({'message': 'Registration Done Successfully'}), 200
        else:
            return jsonify({'error': 'Failed to process image'}), 500
    else:
        return jsonify({'error': 'No image captured'}), 500

if __name__ == "__main__":
    port = 10000
    app.run(debug=True, host='0.0.0.0', port=port)
