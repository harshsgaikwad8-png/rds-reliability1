# RDS Reliability Dashboard

Single-page Flask dashboard for the reliability of a Radial Distribution System.

- `reliability.py` - analytical (FMEA) model + Monte Carlo validation, upgrade scenarios, sensitivity
- `app.py` - Flask API (`/api/overview, network, indices, validation, scenarios, predict`)
- `templates/index.html` - the dashboard (Chart.js)

Run locally: `pip install -r requirements.txt && python app.py` then open http://localhost:5000

Deploy on Render: push to GitHub, create a Web Service, build `pip install -r requirements.txt`,
start `gunicorn app:app`. (`render.yaml` is included.)

All network data at the top of `reliability.py` is synthetic sample data - replace with your feeder's data.
