from flask import Flask, request, redirect, session, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return
