# FastAPI AI Endpoints – cURL Test Suite

Run these examples from the FastAPI project root on the EC2 instance (same folder as `main.py`). All commands assume:

- FastAPI is running locally on `http://127.0.0.1:8063`
- `.env` already contains the Neo4j + Gemini credentials
- No API key is required (`API_KEY` left blank). If you later set one, add `-H "X-API-Key: <your-key>"` to every request.

Use the active Python virtualenv or install `jq`/`curl` globally as you prefer. Responses are piped through `jq` for readability; remove `| jq` if it is not available.

> **Tip:** When pasting the multi-line `curl` examples below, ensure every line ending with a backslash (`\`) has **no** trailing spaces and that the JSON payload remains inside the opening/closing single quotes. If your shell complains about “command not found” or “JSON decode error,” paste the command as a single line instead (the docs include both styles).

---

## 1. Health Check

```bash
curl -s http://127.0.0.1:8063/health | jq
```

Expected output:

```json
{
  "status": "healthy",
  "service": "drug-target-ai",
  "environment": "production"
}
```

---

## 2. Single Classification

Classify a drug–target pair (uses Gemini + stores results in Neo4j).

```bash
curl -s -X POST http://127.0.0.1:8063/classification/classify \
  -H "Content-Type: application/json" \
  -d '{
        "drug_name": "aspirin",
        "target_name": "PTGS2",
        "additional_context": "COX-2 enzyme inhibition",
        "force_reclassify": false
      }' | jq
```

Single-line equivalent:

```bash
curl -s -X POST http://127.0.0.1:8063/classification/classify -H "Content-Type: application/json" -d '{"drug_name":"aspirin","target_name":"PTGS2","additional_context":"COX-2 enzyme inhibition","force_reclassify":false}' | jq
```

Key response fields:

- `success` – `true` on success
- `classification.relationship_type` – e.g. `Primary/On-Target`
- `classification.mechanism` – e.g. `Irreversible Inhibitor`
- `classification.timestamp` – ISO datetime of the stored record

---

## 3. Batch Classification

Classify the same drug against multiple targets in one call. The API iterates sequentially so each target results in a separate Gemini call, respecting rate limits.

```bash
curl -s -X POST http://127.0.0.1:8063/classification/batch \
  -H "Content-Type: application/json" \
  -d '{
        "drug_name": "aspirin",
        "targets": ["PTGS1", "PTGS2"],
        "additional_context": "Cyclooxygenase inhibition",
        "force_reclassify": false
      }' | jq
```

Each entry in `results[]` contains:

- `target` – name of the target processed
- `success` – `true` or `false`
- `classification` – same structure as the single classify endpoint
- `error` – populated only when a target fails

---

## 4. Auto-Classify All Targets for a Drug (Neo4j-assisted)

Fetches targets for the given drug directly from Neo4j (`MATCH (Drug)-[:TARGETS]->(Target)`), classifies each in turn, and stores the results. By default it only classifies relationships that are not already stored; set `force_reclassify` to `true` to recompute everything.

```bash
curl -s -X POST http://127.0.0.1:8063/classification/auto \
  -H "Content-Type: application/json" \
  -d '{
        "drug_name": "aspirin",
        "additional_context": "Cyclooxygenase inhibition",
        "force_reclassify": false,
        "only_unclassified": true
      }' | jq
```

- Omit `limit` to classify every known target; include it (e.g., `\"limit\": 50`) to chunk very large jobs.
- `only_unclassified: true` keeps previously-classified pairs untouched; set `false` with `force_reclassify: true` to refresh stored results
- Response shape matches the manual batch endpoint (`BatchClassificationResponse`)

---

## 4. Classification Status

Check the cached Neo4j record without calling Gemini.

```bash
curl -s http://127.0.0.1:8063/classification/status/aspirin/PTGS2 | jq
```

Returns:

- `success` – `true` if a stored classification exists
- `classification` – `null` if not found

---

## 5. Cascade Prediction

Predict downstream cascade effects (stores the cascade in Neo4j). Depth can be 1–3.

```bash
curl -s -X POST http://127.0.0.1:8063/cascade/predict \
  -H "Content-Type: application/json" \
  -d '{
        "drug_name": "aspirin",
        "target_name": "PTGS2",
        "depth": 2,
        "additional_context": "",
        "force_repredict": false
      }' | jq
```

Important response fields inside `data`:

- `direct_effects[]`, `secondary_effects[]`, `tertiary_effects[]`
  - `entity_name`, `entity_type`, `effect_type`, `reasoning`, `confidence`, `depth`
- `warning` – contains text when the drug–target relationship was not validated
- `stored_in_db` – `true` if Neo4j write succeeded

---

## 6. Force Reclassification / Reprediction (Optional)

Set `force_reclassify` or `force_repredict` to `true` to bypass cached Neo4j results:

```bash
curl -s -X POST http://127.0.0.1:8063/classification/classify \
  -H "Content-Type: application/json" \
  -d '{
        "drug_name": "aspirin",
        "target_name": "PTGS2",
        "force_reclassify": true
      }' | jq
```

```bash
curl -s -X POST http://127.0.0.1:8063/cascade/predict \
  -H "Content-Type: application/json" \
  -d '{
        "drug_name": "aspirin",
        "target_name": "PTGS2",
        "depth": 2,
        "force_repredict": true
      }' | jq
```

---

## 7. Using an API Key (Optional)

If you populate `API_KEY` in `.env`, include the header on every request:

```bash
curl -s http://127.0.0.1:8063/health \
  -H "X-API-Key: $FASTAPI_API_KEY" | jq
```

Replace `$FASTAPI_API_KEY` with your actual key or export it beforehand.

---

## 8. Troubleshooting

- **401 Invalid API key:** Ensure the header uses the same value as `API_KEY` in `.env`.
- **503 Classifier / Cascade predictor unavailable:** Usually indicates missing Gemini or Neo4j credentials; re-check `.env` and restart the service.
- **Connection refused:** Confirm FastAPI is running (`screen -ls`, `curl http://127.0.0.1:8063/health`).
- **Rate limits / Google API errors:** If Gemini throttles, wait and retry; batch classification delays internally (`time.sleep`) but very large batches may still hit quotas.

---

These commands provide a quick manual regression test of every FastAPI endpoint and match the payload shapes expected by the NestJS integration. Adapt drug/target names as needed for your data.


