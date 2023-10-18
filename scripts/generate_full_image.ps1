# Change directory to autogoal-contrib/
Set-Location -Path .\autogoal-contrib\

# Get the list of directories starting with 'autogoal_' and exclude 'autogoal_contrib'
$contribs = Get-ChildItem -Directory | Where-Object { $_.Name -like "autogoal_*" -and $_.Name -ne "autogoal_contrib" } | ForEach-Object { $_.Name.Replace("autogoal_", "") }

# Change back to the original directory
Set-Location -Path ..

# Build the Docker image
docker build . -t autogoal/autogoal:full-latest -f dockerfiles/development/dockerfile --build-arg extras="common $($contribs -join ' ') remote" --no-cache
