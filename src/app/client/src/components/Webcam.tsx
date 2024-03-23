import { useEffect, useRef, useState } from 'react'
import React from 'react';
import '../styles/Webcam.css'

import useWebcam from '../hooks/useWebcam';

const Webcam = ({ text, setText, setConfidence, isDynamic }: {text: string, setText: React.Dispatch<React.SetStateAction<string>>, setConfidence: React.Dispatch<React.SetStateAction<string>>, isDynamic: boolean}) => {
  
  const [detected, setDetected] = useState(false);

  const { 
    captureState, 
    setCaptureState, 
    setDynamic, 
    webcamVideoRef, 
    teardown, 
    recordingState,
  } = useWebcam({ 
    numHands: 1,  
    onCaptureError: () => {
       console.log("error capturing");
    }, 
    onHandDetection: (flag: boolean) => {
      setDetected(flag);
    },
    handedness: "right", 
    onResult: (data: any) => {
      const { result, confidence } = data;
      setText(result);
      setConfidence(confidence);
      console.log("CONFIDENCE:", confidence);
    },
    debug: false
  })

  useEffect(() => {
    return teardown;
  }, [])
  

  useEffect(() => {
    setDynamic(isDynamic);
    if(!isDynamic) {
      setCaptureState(false);
    }
  }, [isDynamic])

  useEffect(() => {
    if(detected) console.log("detected hand");
    else console.log("no longer detecting hand")
  }, [detected])
  
  
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
        <video className={detected ? 'webcam webcam__detected' : 'webcam'} autoPlay muted playsInline ref={webcamVideoRef} />
      </div>
    </div>
  )
  
}

export default Webcam