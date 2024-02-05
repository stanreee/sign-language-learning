import { useEffect, useRef, useState } from 'react'
import React, { createContext } from 'react';
import iQuiz from '../interfaces/iQuiz'
import quizRaw from "../data/additionQuiz.json"
import { io, Socket } from "socket.io-client";
import Peer from "simple-peer";
import '../styles/Webcam.css'

const Webcam = ({ text, setText, close }: {text: string, setText: React.Dispatch<React.SetStateAction<string>>, close: boolean}) => {
  const [activeQuestion, setActiveQuestion] = useState<number>(0)
  const [selectedAnswer, setSelectedAnswer] = useState<boolean>()
  const [showResult, setShowResult] = useState<boolean>(false)
  const [selectedAnswerIndex, setSelectedAnswerIndex] = useState<number|string>('')
  const [result, setResult] = useState<{score: number; correctAnswers: number; wrongAnswers: number}>({
    score: 0,
    correctAnswers: 0,
    wrongAnswers: 0,
  })

  const [streamTimer, setTimer] = useState<NodeJS.Timeout>();
  const [stream, setStream] = useState<MediaStream>();
  const webcamVideo = useRef<HTMLVideoElement | null>(null);
  const serverStream = useRef<string | null>(null);
  const [imgSrc, setImgSrc] = useState('');
  const [signResult, setSignResult] = useState<string>('');

  // communicate with web socket on backend
  const socket = io("http://127.0.0.1:5000"),
  SocketContext = createContext<Socket>(socket);


  // listens to whenever the backend sends frame data back through web socket
  socket.on("stream", (data) => {
    const deserialized = JSON.parse(data);
    const { frame, result } = deserialized;
    //console.log(result);
    //console.log(frame);
    //console.log(data);
    var image = new Image();
    image.src = frame;
    setSignResult(result);
    setText(result);
    // serverStream.current! = image.src;
    // setImgSrc(image.src); // this is a **very** bad way of doing this, it's essentially getting each frame from the backend and setting the img
                          // src to that frame using React's useState hook. this causes multiple rerenders every frame, resulting in performance issues
                          // we need a better way of handling processed images sent from the backend
  });

  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext("2d");

  let numFrames = 0;

  // send webcam snapshot through web socket
  const sendSnapshot = () => {
    if(stream){
      const video = webcamVideo.current;
      ctx?.drawImage(video!, 0, 0, video!.videoWidth, video!.videoHeight * 5, 0, 0, 300, 800);
      let dataURL = canvas.toDataURL('image/jpeg');
      socket.emit('stream', { image: dataURL, frame: numFrames });
      //console.log("Sending frame ", numFrames);
      numFrames += 1;
    }
  }

  const startConnection = () => {
    return navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        height: 350,
        width: 350,
      },
    })
    .then((stream: MediaStream) => {
      setStream(stream);
      webcamVideo.current!.srcObject = stream;
      // sends snapshot of webcam 60 times a second (60 fps)
      const timer = setInterval(() => sendSnapshot(), 1000/10);
      // console.log(timer);
      setTimer(timer);
    })
  }

  const stopConnection = () => {
    if(stream){
      console.log('here!!')
      stream!.getTracks().forEach((track) => {
        console.log("stopping track");
        track.stop();
      })
      setStream(undefined);
      //console.log(localVideoStream.current);
      // localVideoStream.current = null;
    }
  }



  useEffect(() => {
    stopConnection();
  }, [close])

  // useEffect(() => {
  //   if(!mediaStream) startConnection();
  //   return function cleanup() {
  //     console.log("exiting");
  //     stopConnection();
  //   }
  // }, [])

  useEffect(() => {
    startConnection();
    return () => clearInterval(streamTimer);
  }, [])

  
  return (
    <div className="webcam-container">
      <video className='webcam' autoPlay muted playsInline ref={webcamVideo} />
    </div>
  )
  
}

export default Webcam

//export {}