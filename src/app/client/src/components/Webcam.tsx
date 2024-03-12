import { useEffect, useRef, useState } from 'react'
import React, { createContext } from 'react';
import iQuiz from '../interfaces/iQuiz'
import quizRaw from "../data/additionQuiz.json"
import { io, Socket } from "socket.io-client";
import Peer from "simple-peer";
import '../styles/Webcam.css'

 import { Camera } from '@mediapipe/camera_utils';
 import { Hands } from "@mediapipe/hands";
import getFeatures from '../util/getFeatures';
import useWebcam from '../hooks/useWebcam';

const Webcam = ({ text, setText, run, isDynamic }: {text: string, setText: React.Dispatch<React.SetStateAction<string>>, run: boolean, isDynamic: boolean}) => {
  const { captureState, setCaptureState, webcamVideoRef } = useWebcam({ 
    numHands: 1, 
    dynamic: isDynamic, 
    onCaptureError: () => {}, 
    handedness: "right", 
    onResult: (result: any) => {
      console.log(result)
      setText(result);
    }
  })
  
  return (
    <div className="webcam-container">
      <div>
        {
          isDynamic ? (
            <div>
              {
                captureState ? (             
                  <div> 
                    <button className='Record-Button' onClick={() => setCaptureState(true)}>Recording</button>
                    <button className='Record-Button Stop' onClick={() => setCaptureState(false)}>Stop</button>
                  </div>
                ) : (
                  <div> 
                    <button className='Record-Button' onClick={() => setCaptureState(true)}>Start Recording</button>
                  </div>
                )
              }
              </div>
          ) :
          (
            <div>
            </div>
          )
        }
      </div>
      <div>
        <video className='webcam' autoPlay muted playsInline ref={webcamVideoRef} />
      </div>
    </div>
  )
  
}

export default Webcam