import { useEffect, useRef, useState } from 'react'
import React from 'react';
import '../styles/Webcam.css'

import useWebcam from '../hooks/useWebcam';

import toast, { Toaster } from 'react-hot-toast';

const Webcam = ({ text, setText, setConfidence, isDynamic }: {text: string, setText: React.Dispatch<React.SetStateAction<string>>, setConfidence: React.Dispatch<React.SetStateAction<string>>, isDynamic: boolean}) => {
  
  const [detected, setDetected] = useState(false);
  const [countdown, setCountdown] = useState(0);

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
    return () => {
      teardown();
      clearInterval(countdown);
    };
  }, [])
  

  useEffect(() => {
    setDynamic(isDynamic);
    if(!isDynamic) {
      setCaptureState(false);
    }
  }, [isDynamic])

  const startCountdown = (callback: Function) => {
    if(countdown > 0) return;
    setCountdown(3);
    const countdownInterval = setInterval(() => {
        setCountdown((prevCount) => {
            if (prevCount <= 1) {
                clearInterval(countdownInterval);
                callback();
                return 0;
            }
            return prevCount - 1;
        });
    }, 1000);
};

  const startRecording = () => {
    if(detected) startCountdown(() => setCaptureState(true));
    else {
      toast("Make sure your hands are being detected before clicking record!");
    }
  }
  
  
  return (
    <div className="webcam-container">
      <Toaster />
      <div>
        {
          isDynamic && (
            <div>
                <div style={{display: "flex", placeItems: "center", marginTop: "20px"}}> 
                  <button className={captureState ? 'Record-Button Record-Button__disabled' : 'Record-Button'} disabled={captureState} onClick={startRecording}>{captureState ? "Recording" : "Start Recording"}</button>
                  {captureState && (
                    <div style={{height: "100%"}}>
                      <progress value={recordingState} />
                    </div>
                  )}
                </div>
              </div>
          )
        }
      </div>
      <div style={{width: "646px", height: "486px", marginTop: "10px", marginBottom: "10px", position: "relative"}}>
        {<div style={{opacity: countdown > 0 ? "1" : "0"}} className="webcam-countdown">
          <div className="webcam-countdown-counter">
            {countdown}
          </div>
        </div>}
        <video className={detected ? 'webcam webcam__detected' : 'webcam'} autoPlay muted playsInline ref={webcamVideoRef} />
      </div>
    </div>
  )
  
}

export default Webcam