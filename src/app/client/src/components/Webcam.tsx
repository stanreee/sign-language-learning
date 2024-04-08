import { useEffect, useRef, useState } from 'react'
import React from 'react';
import '../styles/Webcam.css'

import useWebcam from '../hooks/useWebcam';

import toast, { Toaster } from 'react-hot-toast';

const Webcam = ({ text, setText, setConfidence, isDynamic, hands, canChangeType }: {text: string, setText: React.Dispatch<React.SetStateAction<string>>, setConfidence: React.Dispatch<React.SetStateAction<string>>, isDynamic: boolean, hands: number, canChangeType: boolean}) => {
  
  const [detectedState, setDetectedState] = useState(false);
  const detected = useRef(false);
  const [countdown, setCountdown] = useState(0);
  const [numHands, setNumHands] = useState(hands);

  const { 
    captureState, 
    setCaptureState, 
    dynamicState,
    setDynamic, 
    webcamVideoRef, 
    teardown, 
    recordingState,
  } = useWebcam({ 
    numHands: numHands,  
    onCaptureError: () => {
       console.log("error capturing");
    }, 
    onHandDetection: (flag: boolean) => {
      detected.current = flag;
      setDetectedState(flag);
    },
    handedness: "right", 
    onResult: (data: any) => {
      const { result, confidence } = data;
      setText(result);
      setConfidence(confidence);
      console.log("CONFIDENCE:", confidence);
    },
    debug: true,
    isDynamic: isDynamic,
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
      setNumHands(1);
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
    startCountdown(() => {
      if(detected.current) setCaptureState(true)
      else {
        toast("Make sure your hands are being detected as indicated by the green border before the countdown ends!");
      }
    });
  }

  // useEffect(() => {
  //   console.log(detected);
  // }, [detected])
  
  return (
    <div className="webcam-container">
      <Toaster />
      <div>
        <div>
          {canChangeType && (
            <div>
                <button style={{marginRight: "12px"}} className={dynamicState ? "Button active" : "Button disabled"} onClick={() => {setDynamic(false)}}>Static (Non-Moving)</button>
                <button className={dynamicState ? "Button disabled" : "Button active"} onClick={() => {setDynamic(true)}}>Dynamic (Moving)</button>
            </div>
          )}
        </div>
        {
          dynamicState && (
            <div style={{marginTop: "20px"}}>
              <div style={{display: "flex", placeItems: "center", marginTop: "10px"}}> 
                <button className={captureState ? 'Record-Button Record-Button__disabled' : 'Record-Button'} disabled={captureState} onClick={startRecording}>{captureState ? "Recording" : "Start Recording"}</button>
                <button onClick={() => setNumHands(1)} disabled={captureState || countdown > 0} className={numHands === 1 || captureState || countdown > 0 ? "Hands-Button active" : "Hands-Button"}>1 Hand</button>
                <button onClick={() => setNumHands(2)} disabled={captureState || countdown > 0} className={numHands === 2 || captureState || countdown > 0 ? "Hands-Button active" : "Hands-Button"}>2 Hands</button>
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
        <video className={detectedState ? 'webcam webcam__detected' : 'webcam'} autoPlay muted playsInline ref={webcamVideoRef} />
      </div>
    </div>
  )
  
}

export default Webcam