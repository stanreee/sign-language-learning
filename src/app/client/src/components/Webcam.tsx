import { useEffect, useRef, useState } from 'react'
import { io } from "socket.io-client";
import '../styles/Webcam.css'

import aslLetters from '../images/ASLLetters.png';

console.log(aslLetters);

const Webcam = () => {

  const [streamTimer, setTimer] = useState<NodeJS.Timeout>();
  const [stream, setStream] = useState<MediaStream>();
  const webcamVideo = useRef<HTMLVideoElement | null>(null);
  const [signResult, setSignResult] = useState<string>('');

  // communicate with web socket on backend
  const socket = io("http://127.0.0.1:5000")

  // listens to whenever the backend sends frame data back through web socket
  socket.on("stream", (data) => {
    const deserialized = JSON.parse(data);
    const { frame, result } = deserialized;
    console.log(result);
    console.log(frame);
    console.log(data);
    var image = new Image();
    image.src = frame;
    setSignResult(result);
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
    const video = webcamVideo.current;
    ctx?.drawImage(video!, 0, 0, video!.videoWidth, video!.videoHeight * 5, 0, 0, 300, 800);
    let dataURL = canvas.toDataURL('image/jpeg');
    socket.emit('stream', { image: dataURL, frame: numFrames });
    console.log("Sending frame ", numFrames);
    numFrames += 1;
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

  // const stopConnection = () => {
  //   if(mediaStream){
  //     mediaStream!.getTracks().forEach((track) => {
  //       console.log("stopping track");
  //       track.stop();
  //     })
  //     mediaStream = null;
  //     console.log(localVideoStream.current);
  //     // localVideoStream.current = null;
  //   }
  // }

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
      <div>
        <h2>Live Sign Language Webcam</h2>
        <h2>Result: {signResult}</h2>
      </div>
      <video className='webcam' autoPlay muted playsInline ref={webcamVideo} />

      <img src={aslLetters} width={500} height={500} alt="aslLetters" />
    </div>
  )
}

export default Webcam

//export {}