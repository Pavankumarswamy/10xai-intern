# Run the AI Intern Docker Container
# Make sure you've built the image first using build-docker.ps1

Write-Host "Starting AI Intern container..." -ForegroundColor Cyan
Write-Host ""
Write-Host "Application will be available at: http://localhost:7860" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

docker run -it --rm -p 7860:7860 --name ai-intern-container ai-intern-3tasks
