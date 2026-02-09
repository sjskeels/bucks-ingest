# test_salesclass_by_date.ps1
$ErrorActionPreference = "Stop"

cd C:\dev\bucks-ingest\bucks-ingest

$BASE = "https://bucks-ingest-webhook-production.up.railway.app"

# Read token from file (avoids clipboard confusion)
$tokFile = "$env:TEMP\bucks_parse_token.txt"
if (!(Test-Path $tokFile)) { throw "Token file missing: $tokFile" }

$TOKEN = ((Get-Content $tokFile -Raw) -replace "\s","").Trim()
if ($TOKEN.Length -lt 20) { throw "Token looks wrong (len=$($TOKEN.Length)). Check $tokFile" }

Write-Host ("token_len={0}" -f $TOKEN.Length)

# Health check
Remove-Item -ErrorAction SilentlyContinue ".\health_headers.txt",".\health_body.json"
& curl.exe -sS "$BASE/healthz" --dump-header ".\health_headers.txt" --output ".\health_body.json" -w "`nHTTP_CODE=%{http_code}`n"
Get-Content ".\health_headers.txt" -TotalCount 1
Get-Content ".\health_body.json"

$dates = @(
  "2026-01-27",
  "2026-01-29",
  "2026-01-30",
  "2026-01-31",
  "2026-02-02",
  "2026-02-03",
  "2026-02-04"
)

foreach ($d in $dates) {
  Write-Host "`n=== $d ==="

  $uri = "$BASE/parse/salesclass/by-date?business_date=$d"
  Remove-Item -ErrorAction SilentlyContinue "headers_$d.txt","body_$d.json"

  $args = @(
    "-sS",
    "-X","POST",
    $uri,
    "-H", ("X-Parse-Token: {0}" -f $TOKEN),
    "-H", "X-Debug: 1",
    "--dump-header", ("headers_{0}.txt" -f $d),
    "--output", ("body_{0}.json" -f $d),
    "-w", "`nHTTP_CODE=%{http_code}`n"
  )

  & curl.exe @args

  if (Test-Path "headers_$d.txt") {
    Get-Content "headers_$d.txt" -TotalCount 1
  }

  if (Test-Path "body_$d.json") {
    $raw = Get-Content "body_$d.json" -Raw
    try { $json = $raw | ConvertFrom-Json } catch { $json = $null }

    if ($json -and $null -ne $json.ok) {
      $ok       = $json.ok
      $rows     = $json.parse.rows_parsed
      $nonNull  = $json.parse.ext_price_non_null
      $skipped  = $json.parse.rows_skipped
      $dbSkip   = $json.db.daily.skipped_db
      $inserted = $json.inserted_rows

      Write-Host ("ok={0} inserted={1} rows_parsed={2} ext_price_non_null={3} rows_skipped={4} db_skipped={5}" -f `
        $ok,$inserted,$rows,$nonNull,$skipped,$dbSkip)

      if (-not $ok) {
        Write-Host $raw
      }
    } else {
      Write-Host $raw
    }
  }
}
