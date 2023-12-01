# Use the official TensorFlow image as a base image
FROM tensorflow/tensorflow:2.4.0

# Set the working directory in the container
WORKDIR /app

# Copy the necessary files to the container
COPY model/ /app/model/
COPY train.py /app

# Install required dependencies
RUN pip install Flask librosa numpy

# Expose the port that Flask will run on
EXPOSE 5000

ENTRYPOINT ["python3.7", "/opt/ml/code/train.py"]

# Command to run the application
CMD ["python", "inference_api.py"]
