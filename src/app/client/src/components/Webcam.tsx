import { useEffect, useRef, useState } from 'react'
import React from 'react';
import '../styles/Webcam.css'

import useWebcam from '../hooks/useWebcam';

const Webcam = ({ text, setText, run }: {text: string, setText: React.Dispatch<React.SetStateAction<string>>, run: boolean}) => {
  const { captureState, setCaptureState, webcamVideoRef } = useWebcam({ 
    numHands: 1, 
    dynamic: true, 
    onCaptureError: () => {}, 
    handedness: "right", 
    onResult: (data: any) => {
      const { result, confidence } = data;
      setText(result);
      console.log("CONFIDENCE:", confidence);
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