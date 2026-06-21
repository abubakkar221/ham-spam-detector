from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib

app = FastAPI()

# Model আর Vectorizer load করছি
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": None}
    )


@app.post("/", response_class=HTMLResponse)
def predict(request: Request, email_text: str = Form(...)):
    text_vector = vectorizer.transform([email_text])
    prediction = model.predict(text_vector)[0]

    if prediction == "spam":
        result = "🚫 SPAM"
    else:
        result = "✅ HAM (Not Spam)"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": result, "email_text": email_text}
    )