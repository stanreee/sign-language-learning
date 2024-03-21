import { useEffect, useRef, useState } from 'react'
import React from 'react';
import '../styles/Webcam.css'

import useWebcam from '../hooks/useWebcam';

const Webcam = ({ text, setText, setConfidence, isDynamic }: {text: string, setText: React.Dispatch<React.SetStateAction<string>>, setConfidence: React.Dispatch<React.SetStateAction<string>>, isDynamic: boolean}) => {
  
  const { captureState, setCaptureState, setDynamic, webcamVideoRef, teardown, recordingState } = useWebcam({ 
    numHands: 1,  
    onCaptureError: () => {}, 
    handedness: "right", 
    onResult: (data: any) => {
      const { result, confidence } = data;
      setText(result);
      setConfidence(confidence);
      console.log("CONFIDENCE:", confidence);
    }
  })

  useEffect(() => {
    return teardown;
  }, [])
  

  useEffect(() => {
    setDynamic(isDynamic);
  }, [isDynamic])
  
  
  return (
    <div className="webcam-container">
      <div>
        {
          isDynamic ? (
            <div>
              {
                captureState ? (    
                  <div>         
                    <div> 
                      <button className='Record-Button' onClick={() => setCaptureState(true)} disabled>Recording</button>
                    </div>
                    <div>
                      <progress value={recordingState} />
                    </div>
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