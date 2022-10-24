#!/usr/bin/python3
# This Python file uses the following encoding: utf-8
# from distutils.log import debug
# import os,sys
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS, cross_origin
from pose import generateImages
from pose import  video as Contadores
from pose import finished as EndWebcam
import config1

app = Flask(__name__)
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"/": {"origins": "http://localhost:port"}})

# Counter = []

@app.route("/")
def beging():
    return """
    <html> 
        <body>
                <h1> AI - Visão computacional: </h1>
                    <u1>
                        <a href= "/data"> Iniciar Webcam </a>

                    

        </body> 
    
    </html> """

@app.route("/data", methods=['GET'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def data():
        config1.kill_cam = False
        Counter = Contadores()

        # if Counter is None:
        #     abort(404, description='Resource not avalible')
        print(Counter)

        shoulder_L = Counter[0]
        shoulder_R = Counter[1]
        elbow_L = Counter[2]
        elbow_R = Counter[3]
        hip_L = Counter[4]
        hip_R = Counter[5]
        neck = Counter[6]
        
        return jsonify(
        ombro_1 = shoulder_L,
        ombro_2 = shoulder_R,
        cotovelo_1 = elbow_L,
        cotovelo_2 = elbow_R,
        quadril_1 = hip_L,
        quadril_2 = hip_R,
        pescoco = neck 
    )
   


@app.route("/end", methods=['GET'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def endWebcam():

    return jsonify(EndWebcam())


@app.route("/video_feed")
def video_feed():
	return Response(generateImages(),
		mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':

    app.run(host="127.0.0.1", port=5001, debug =True)


