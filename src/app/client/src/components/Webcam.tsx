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

const Webcam = ({ text, setText, run }: {text: string, setText: React.Dispatch<React.SetStateAction<string>>, run: boolean}) => {
  const { captureState, setCaptureState, webcamVideoRef } = useWebcam({ 
    numHands: 1, 
    dynamic: true, 
    onCaptureError: () => {}, 
    handedness: "right", 
    onResult: (result: any) => {
      setText(result);
    }
  })
  
  return (
    <div className="webcam-container">
      <video className='webcam' autoPlay muted playsInline ref={webcamVideoRef} />
      <button disabled={captureState} onClick={() => setCaptureState(true)}>{captureState ? "Recording" : "Start Recording"}</button>
    </div>
  )
  
}

export default Webcam

//export {}