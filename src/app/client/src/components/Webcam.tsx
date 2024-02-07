import { useEffect, useRef, useState } from 'react'
import React, { createContext } from 'react';
import iQuiz from '../interfaces/iQuiz'
import quizRaw from "../data/additionQuiz.json"
import { io, Socket } from "socket.io-client";
import Peer from "simple-peer";
import '../styles/Webcam.css'

import { Camera } from '@mediapipe/camera_utils';
import { Hands } from "@mediapipe/hands";

const Webcam = ({ text, setText, run }: {text: string, setText: React.Dispatch<React.SetStateAction<string>>, run: boolean}) => {
  const webcamVideo = useRef<any>(null);
  const mediapipeCamera = useRef<any>(null);
  const hands = useRef<any>(null);

  // communicate with web socket on backend
  const socket = io("http://127.0.0.1:5000"),
  SocketContext = createContext<Socket>(socket);

  const onResults = (results: any) => {
    const { multiHandLandmarks } = results;
    if(multiHandLandmarks.length >= 1 && multiHandLandmarks[0].length >= 21) {
      socket.emit('stream', { landmarks: multiHandLandmarks[0] });
    }
  }
  
  const loadHands = () => {
    hands.current = new Hands({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
      }
    })
    hands.current.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    })
    hands.current.onResults(onResults);
  }

  useEffect(() => {
    async function initCamera() {
      mediapipeCamera.current = new Camera(webcamVideo.current, {
        onFrame: async () => {
          await hands.current.send({ image: webcamVideo.current });
        },
      })
      mediapipeCamera.current.start();
    }

    socket.on("stream", (data) => {
      const deserialized = JSON.parse(data);
      const { result } = deserialized;
      setText(result)
    });

    initCamera();
    loadHands();
  }, [])
  
  return (
    <div className="webcam-container">
      <video className='webcam' autoPlay muted playsInline ref={webcamVideo} />
    </div>
  )
  
}

export default Webcam

//export {}