# Build Docker Image Only
# Use this if you just want to build the image without running it

Write-Host "Building Docker image 'ai-intern-3tasks'..." -ForegroundColor Cyan
Write-Host "This may take 5-10 minutes..." -ForegroundColor Yellow
Write-Host ""

docker build -t ai-intern-3tasks .

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✓ Build successful!" -ForegroundColor Green
    Write-Host ""
    Write-Host "To run the container, use:" -ForegroundColor Cyan
    Write-Host "  .\run-docker.ps1" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "✗ Build failed!" -ForegroundColor Red
}

pause
