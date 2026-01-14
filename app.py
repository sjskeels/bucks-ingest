import os
from flask import Flask, request

app = Flask(__name__)

@app.get("/")
def health():
    return {"ok": True}

@app.post("/mailgun/inbound")
def inbound():
    # we'll parse Mailgun payload later
    return "ok", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
