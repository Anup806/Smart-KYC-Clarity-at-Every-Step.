# Smart KYC: Clarity at Every Step. (Streamlit)

A lightweight demo for Challenge 4: Intelligent KYC Experience - Smart Form Assistance and Support Ticket Reduction.

## What this shows
- Stepper-based KYC flow: Personal Info, Address, Document Upload, Review/Submit, Status Tracker
- Mock OCR auto-fill and quality checks
- Status transparency and contextual help to reduce support tickets

## Run locally (Windows)
1) Create a virtual environment:

```
py -m venv .venv
.venv\Scripts\activate
```

2) Install dependencies:

```
pip install -r requirements.txt
```

3) Launch the demo:

```
streamlit run app.py
```

## Demo tips
- Use the sidebar stepper to jump between steps while recording.
- Upload any small image or PDF to show mocked OCR results.
- Use "Simulate outcomes" to show approved or rejected states quickly.
