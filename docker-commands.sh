# Build the Docker image
docker build -t ai-intern-3tasks .

# Run the container
docker run -it -p 7860:7860 ai-intern-3tasks
