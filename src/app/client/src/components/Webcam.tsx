import { useEffect, useRef, useState } from 'react'
import React from 'react';
import '../styles/Webcam.css'

import useWebcam from '../hooks/useWebcam';

const Webcam = ({ text, setText, run, isDynamic }: {text: string, setText: React.Dispatch<React.SetStateAction<string>>, run: boolean, isDynamic: boolean}) => {
  const { captureState, setCaptureState, webcamVideoRef, teardown } = useWebcam({ 
    numHands: 1, 
    dynamic: isDynamic, 
    onCaptureError: () => {}, 
    handedness: "right", 
    onResult: (data: any) => {
      const { result, confidence } = data;
      setText(result);
      console.log("CONFIDENCE:", confidence);
    }
  })

  useEffect(() => {
    return teardown;
  }, [])
  
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