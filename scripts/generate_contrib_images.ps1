param (
    [switch]$p = $false
)

$contribs = Get-ChildItem -Path autogoal-contrib/ -Filter autogoal_* | Where-Object { $_.Name -ne 'autogoal_contrib' } | ForEach-Object { $_.Name.Replace('autogoal_', '') }

foreach ($contrib in $contribs)
{
    make docker-contrib CONTRIB=$contrib
    if ($p)
    {
        docker push autogoal/autogoal:$contrib
    }
}
