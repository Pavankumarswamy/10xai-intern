# Build and Run AI Intern - 3 Tasks in 1 Docker Container
# This script builds the Docker image and runs the container

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  AI Intern - 3 Tasks in 1 Docker  " -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if Docker is running
Write-Host "[Step 1/3] Checking Docker status..." -ForegroundColor Yellow
$dockerInfo = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker is not running!" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again." -ForegroundColor Red
    Write-Host ""
    pause
    exit 1
}
Write-Host "Docker is running" -ForegroundColor Green
Write-Host ""

# Step 2: Build the Docker image
Write-Host "[Step 2/3] Building Docker image..." -ForegroundColor Yellow
Write-Host "Image name: ai-intern-3tasks" -ForegroundColor Gray
Write-Host "This may take 5-10 minutes on first build..." -ForegroundColor Gray
Write-Host ""

docker build -t ai-intern-3tasks .

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Docker build failed!" -ForegroundColor Red
    Write-Host "Check the error messages above for details." -ForegroundColor Red
    Write-Host ""
    pause
    exit 1
}

Write-Host ""
Write-Host "Docker image built successfully!" -ForegroundColor Green
Write-Host ""

# Step 3: Run the container
Write-Host "[Step 3/3] Starting container..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Application will be available at: http://localhost:7860" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the container" -ForegroundColor Yellow
Write-Host ""

docker run -it --rm -p 7860:7860 --name ai-intern-container ai-intern-3tasks
