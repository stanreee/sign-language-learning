import { useEffect, useRef, useState } from 'react'
import iQuiz from '../interfaces/iQuiz'
import quizRaw from "../data/additionQuiz.json"
import { io } from "socket.io-client";
import Peer from "simple-peer";


import '../styles/Quiz.css'
import iQuizQuestions from '../interfaces/iQuizQuestions';
import { Link } from 'react-router-dom';

type QuizProps = {
    title: string
  quizQuestions: iQuizQuestions[];
};

const Quiz = ({
    title,
    quizQuestions,
}: QuizProps) => {
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

  // communicate with web socket on backend
  const socket = io("http://127.0.0.1:5000")

  // listens to whenever the backend sends frame data back through web socket
  socket.on("stream", (frame) => {
    var image = new Image();
    image.src = frame;
    // serverStream.current! = image.src;
    setImgSrc(image.src); // this is a **very** bad way of doing this, it's essentially getting each frame from the backend and setting the img
                          // src to that frame using React's useState hook. this causes multiple rerenders every frame, resulting in performance issues
                          // we need a better way of handling processed images sent from the backend
    console.log("frame:", frame);
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
      const timer = setInterval(() => sendSnapshot(), 1000/60);
      console.log(timer);
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

  const questions = quizQuestions

  // const { question, choices, correctAnswer } = questions[activeQuestion]

  // const onClickNext = () => {
  //   setSelectedAnswerIndex('')
  //   setResult((prev) =>
  //     selectedAnswer
  //       ? {
  //           ...prev,
  //           score: prev.score + 1,
  //           correctAnswers: prev.correctAnswers + 1,
  //         }
  //       : { ...prev, wrongAnswers: prev.wrongAnswers + 1 }
  //   )
  //   if (activeQuestion !== questions.length - 1) {
  //     setActiveQuestion((prev) => prev + 1)
  //   } else {
  //     setActiveQuestion(0)
  //     setShowResult(true)
  //   }
  // }

  // const onAnswerSelected = (answer: string, index: number) => {
  //   setSelectedAnswerIndex(index)
  //   if (answer === correctAnswer) {
  //     setSelectedAnswer(true)
  //   } else {
  //     setSelectedAnswer(false)
  //   }
  // }

  // const addLeadingZero = (number: number) => (number > 9 ? number : `0${number}`)

  return (
    <div className="quiz-container">
      {!showResult ? (
        <div>
          <div>
            <h3>{title} Quiz</h3>
          </div>
          <video style={{visibility: "hidden"}} autoPlay muted playsInline ref={webcamVideo} />
          <img src={imgSrc} />
          {/* <div>
            <span className="active-question-no">{addLeadingZero(activeQuestion + 1)}</span>
            <span className="total-question">/{addLeadingZero(questions.length)}</span>
          </div>
          <h2>{question}</h2>
          <ul>
            {choices.map((answer: string, index: number) => (
              <li
                onClick={() => onAnswerSelected(answer, index)}
                key={answer}
                className={selectedAnswerIndex === index ? 'selected-answer' : ''}>
                {answer}
              </li>
            ))}
          </ul> */}
          <div className="flex-right">
            <button onClick={sendSnapshot} disabled={selectedAnswerIndex === null}>
              {activeQuestion === questions.length - 1 ? 'Finish' : 'Next'}
            </button>
          </div>
        </div>
      ) : (
        <div className="result">
          <h3>Result</h3>
          <p>
            Total Question: <span>{questions.length}</span>
          </p>
          <p>
            Total Score:<span> {result.score}</span>
          </p>
          <p>
            Correct Answers:<span> {result.correctAnswers}</span>
          </p>
          <p>
            Wrong Answers:<span> {result.wrongAnswers}</span>
          </p>
          <Link reloadDocument to={'/Exercises'}>
            <button>Return to exercises</button> 
          </Link>
        </div>
      )}
    </div>
  )
}

export default Quiz

//export {}