\# Buck’s Exec Dashboard — Ingest + Parse + Metabase Handoff (2026-01-26)



\## What we built (current working state)

We have an end-to-end pipeline that:

1\) Receives emailed TransAct PDFs (via Mailgun inbound webhook)

2\) Stores the raw attachment in Postgres (`raw.inbound\_files`)

3\) Infers `report\_type` + `business\_date` from filename/subject

4\) Parses the latest Sales Statistics PDF and upserts daily KPIs into `analytics.salesstat\_daily`

5\) Metabase reads from `analytics.salesstat\_daily` for charts/dashboards



\### Confirmed working (today)

\- In Metabase, `raw.inbound\_files` shows:

&nbsp; - `filename = salesstatreport012526020108.PDF`

&nbsp; - `report\_type = salesstat`

&nbsp; - `business\_date = 2026-01-25`

\- Parse endpoint returns JSON including upserted fields like:

&nbsp; - `net\_sales = 3628.71`

&nbsp; - `gross\_sales = 3856.42`

&nbsp; - `transactions = 97`

&nbsp; - `avg\_ticket = 37.41`



\## Repo + deploy

\- Repo: `sjskeels/bucks-ingest`

\- Service: `bucks-ingest-webhook` on Railway (deploys from GitHub)

\- NOTE: editing variables in Railway does NOT deploy code changes; code changes require deploy from GitHub integration.



\## Key services (Railway)

\- `bucks-metabase` (largest cost driver, RAM)

\- `bucks-postgres`

\- `bucks-ingest-webhook` (Flask app receiving email + parsing)

\- (optional/older) `bucks-ingest` service existed; webhook is the one being used.



\## Flask endpoints (bucks-ingest-webhook)

\- `GET /health`

&nbsp; - returns `{"ok": true}`

\- `POST /mailgun/inbound`

&nbsp; - Mailgun posts inbound email + attachments here

&nbsp; - Stores each attachment in `raw.inbound\_files`

&nbsp; - Computes `sha256` for de-dupe and uses upsert to fill `report\_type` and `business\_date`

\- `POST /parse/salesstat/latest`

&nbsp; - Requires header: `X-Parse-Token: <PARSE\_TOKEN>`

&nbsp; - Finds the latest `raw.inbound\_files` row for `report\_type='salesstat'` (and/or newest PDF)

&nbsp; - Extracts KPIs from PDF text via `pypdf`

&nbsp; - Upserts into `analytics.salesstat\_daily`



\## Database tables

\### Raw storage

\- `raw.inbound\_files`

&nbsp; - `id`

&nbsp; - `received\_at`

&nbsp; - `mail\_from`

&nbsp; - `mail\_subject`

&nbsp; - `mail\_message\_id`

&nbsp; - `filename`

&nbsp; - `content\_type`

&nbsp; - `file\_bytes` (bytea)

&nbsp; - `sha256` (unique key / conflict key)

&nbsp; - `report\_type`

&nbsp; - `business\_date`



\### Analytics

\- `analytics.salesstat\_daily`

&nbsp; - `business\_date` (PK)

&nbsp; - `net\_sales` numeric

&nbsp; - `gross\_sales` numeric

&nbsp; - `transactions` int

&nbsp; - `avg\_ticket` numeric

&nbsp; - `created\_at`, `updated\_at`



\## Metabase (what matters in Metabase naming)

\### Collection recommendation

\- Put everything under: \*\*Our Analytics → Buck’s - Sales\*\*

&nbsp; - Raw log question (debug): “Inbound Files — Raw Log”

&nbsp; - Authoritative table: “Daily Sales — Summary (Authoritative)”

&nbsp; - Line charts:

&nbsp;   - “SalesStat Daily — Net Sales (Line)”

&nbsp;   - “SalesStat Daily — Gross Sales (Line)”

&nbsp;   - “SalesStat Daily — Transactions (Line)”

&nbsp;   - “SalesStat Daily — Avg Ticket (Line)”

&nbsp; - Dashboard:

&nbsp;   - “Sales Dashboard”



\## Scheduling / automation (NOT local machine)

Emails arrive ~1:00 AM daily.

We want parse runs:

\- \*\*3:00 AM daily\*\* (main run)

\- \*\*12:00 PM daily\*\* (retry run)



\### Lowest-cost approach: GitHub Actions calling the parse endpoint

Use GitHub Actions cron to `curl` the Railway endpoint twice a day.

\- Pros: essentially free, no extra Railway service runtime.

\- Auth: send header `X-Parse-Token`



Example call:

curl -s -X POST -H "X-Parse-Token: <token>" https://bucks-ingest-webhook-production.up.railway.app/parse/salesstat/latest



Success response includes `"ok": true`.

Failure response includes `"ok": false` and `"error"` with a sample of extracted PDF text.



\## Cost control notes (Railway)

\- Biggest spend is \*\*Metabase RAM\*\*, billed continuously.

\- We reduced Metabase memory limit to \*\*0.5 GB\*\* (as low as available in UI).

\- Postgres and ingest webhook are comparatively small costs.

\- There is no “replicas = 0” / “pause” button in this UI; best lever is lowering memory/CPU and keeping services minimal.



\## Next build steps (expand dashboard)

1\) Identify the remaining daily TransAct reports we care about (names + expected KPIs).

2\) For each report type:

&nbsp;  - Add inference rules for `report\_type` / `business\_date` if needed

&nbsp;  - Add parser that reads PDF bytes from `raw.inbound\_files`

&nbsp;  - Create `analytics.<report>\_daily` table

&nbsp;  - Upsert extracted KPIs

&nbsp;  - Create Metabase questions + dashboard cards

3\) Extend automation to parse all required report types (same 3AM + noon retry windows).



\## Operating rule

One step at a time:

\- change code → commit/push → deploy in Railway → test endpoint → confirm row(s) in Postgres → confirm card in Metabase



