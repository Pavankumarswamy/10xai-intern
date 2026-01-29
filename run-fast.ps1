# Run with local code injected (No Build Needed)
Write-Host "Starting AI Intern with Local Code..." -ForegroundColor Green
Write-Host "Skipping install/build steps for speed." -ForegroundColor Gray

docker run -it -p 7860:7860 `
  -v "${PWD}/app.py:/app/app.py" `
  -v "${PWD}/school_information.pdf:/app/school_information.pdf" `
  ai-intern-final
