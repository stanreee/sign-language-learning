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

  const [stream, setStream] = useState<MediaStream>();
  const webcamVideo = useRef<HTMLVideoElement | null>(null);

  const socket = io("http://127.0.0.1:5000")

  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext("2d");

  const sendSnapshot = () => {
    const video = webcamVideo.current;
    ctx?.drawImage(video!, 0, 0, video!.videoWidth, video!.videoHeight, 0, 0, 300, 150);
    let dataURL = canvas.toDataURL('image/jpeg');
    socket.emit('stream', dataURL);
  }

  const startConnection = () => {
    navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        height: 350,
        width: 350,
      },
    })
    .then((stream: MediaStream) => {
      setStream(stream);
      webcamVideo.current!.srcObject = stream;
      sendSnapshot();
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
          <video autoPlay muted playsInline ref={webcamVideo} />
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
          </ul>
          <div className="flex-right">
            <button onClick={onClickNext} disabled={selectedAnswerIndex === null}>
              {activeQuestion === questions.length - 1 ? 'Finish' : 'Next'}
            </button>
          </div> */}
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