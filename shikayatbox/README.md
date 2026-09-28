# ShikayatBox — Citizen Grievance Portal (india.gov.in-style UI)

Open `index.html` (or `python3 -m http.server 8080`, or drag the folder onto Netlify Drop). Needs internet for the React/Babel/font CDNs.
Municipal Desk demo password: `nagar123`.

## Features
- Government-portal UI: tricolour bar, A-/A/A+ text size, EN/हिन्दी/मराठी switcher, hero search, stats band, red "Report a Problem" panel
- Sign up / Sign in, Profile (edit, my complaints, stats), About & Help, Contact, helplines
- Report: categories (+custom), photos (compressed), GPS, **voice input** (browser Web Speech: en-IN/hi-IN/mr-IN), priority
- **Smart category suggestion** (multilingual keyword classifier — baseline, not a trained model)
- **Duplicate detection** (same category within 100 m or similar text → offer to upvote existing)
- **Explainable priority score** (priority + upvotes + animal/drainage + near school/hospital + age) with reasons shown
- Community feed (filter/search/sort), upvote (1/device), WhatsApp share
- Municipal Desk: queue ranked by priority score, status updates, CSV export
- Email/SMS open the user's own mail/SMS app (mailto:/sms:) — no server

## Honest limits
- Data + accounts live in browser localStorage (per device). Password hashing is NOT secure. A real backend (FastAPI + Postgres/PostGIS) is needed for shared data.
- No trained ML model or Bhashini yet (needs a labelled dataset and Bhashini API keys).

## v3 additions
- Moving alert ticker, Track-by-Ticket page (5-stage timeline + SLA timer), department routing.
- Transparency page: Leaflet + OpenStreetMap heatmap, department performance, SLA breaches.
- Officer desk: before/after photo (visual comparison; automatic CLIP check is roadmap).
- Real trained model: `ml/train_text_model.py` (char n-gram Naive Bayes, EN/HI/MR/Hinglish) -> `ml/model.json`, embedded in index.html.
  **Trained on synthetic template data** (held-out accuracy is optimistic). Retrain on real complaints when available.
- 28 demo Pune complaints are seeded on first load. Reset by clearing site data.
## Banner carousel
img/b1-b4.jpg are the supplied banners, unmodified except cropped apart; shown in a fixed 2:1 stage (contain + blurred fill), auto-advance 5 s, swipe, arrows, dots, pause button, keyboard, reduced-motion aware. Replace files to change banners.
## v6
A-/A/A+ now zooms the whole page (persisted); dark mode toggle (persisted, follows OS by default); tap-to-pin OSM map on the report form with Nominatim address lookup; heatmap auto-fits points; live status: tabs sync via storage event + 4 s refresh.
Open via a local server (python3 -m http.server) - OSM tiles may refuse file:// pages.
## Not done (be honest with judges)
Real backend/Supabase, Bhashini keys, image models (CLIP/MobileNet), XGBoost+SHAP, load testing. Data is per-browser (localStorage).
Government-style design only; no national emblem, not an official service.
