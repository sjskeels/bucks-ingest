from flask import Flask, request

app = Flask(__name__)

@app.get("/")
def health():
    return {"ok": True}

@app.post("/mailgun/inbound")
def inbound():
    # for now: just acknowledge receipt
    return "ok", 200
