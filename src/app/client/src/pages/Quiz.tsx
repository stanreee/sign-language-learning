import { SetStateAction, useEffect, useReducer, useRef, useState } from 'react'
import iQuiz from '../interfaces/iQuiz'
import quizRaw from "../data/additionQuiz.json"

import '../styles/Quiz.css'
import iQuizQuestions from '../interfaces/iQuizQuestions';
import iQuizASL from '../interfaces/iQuizASL';

import { Link } from 'react-router-dom';
import Timer from '../components/Timer';
import Webcam from '../components/Webcam';

import { Check } from 'react-bootstrap-icons';

type QuizProps = {
    title: string
  quizQuestions: iQuizASL[];
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
  const [sign, setSign] = useState("A")
  const [isCorrectSign, setIsCorrectSign] = useState(false);
  const [isTimeExpired, setIsTimeExpired] = useState(false);

  // timer countdown
  const [timerReset, resetTimer] = useState(false);

  const questions = quizQuestions

  //const { question, choices, correctAnswer } = questions[activeQuestion]
  const { question, correctAnswer } = questions[activeQuestion]

  const onClickNext = () => {
    setSelectedAnswerIndex('');
    setIsCorrectSign(false);
    setIsTimeExpired(false);
    resetTimer(true);
    setResult((prev) =>
      selectedAnswer
        ? {
            ...prev,
            score: prev.score + 1,
            correctAnswers: prev.correctAnswers + 1,
          }
        : { ...prev, wrongAnswers: prev.wrongAnswers + 1 }
    )
    if (activeQuestion !== questions.length - 1) {
      setActiveQuestion((prev) => prev + 1)
    } else {
      setActiveQuestion(0)
      setShowResult(true)
    }
  }

  const onAnswerSelected = (answer: string, index: number) => {
    setSelectedAnswerIndex(index)
    if (answer === correctAnswer) {
      setSelectedAnswer(true)
    } else {
      setSelectedAnswer(false)
    }
  }

  const addLeadingZero = (number: number) => (number > 9 ? number : `0${number}`)

  // get ASL results from Webcam module and go to next
  useEffect(() => {
    if(sign === question){
      setIsCorrectSign(true);
    }
  }, [sign]);

  // hook from Timer module for expired time
  useEffect(() => {
    if(isTimeExpired) {
        setSelectedAnswer(false);
        onClickNext();
      }
  }, [isTimeExpired])

   // Reset timer when next is clicked
   useEffect(() => {
      if (timerReset) {
        resetTimer(false);
      }
    }, [timerReset]);

  return (
    <div className="quiz-container">
      <div className='quiz-container-column'>
      {!showResult ? (
        <div>
          <h3>{title} Quiz</h3>
          <div className='container-row'>
            <div className='container-column'>
              <div>
                <span className="active-question-no">{addLeadingZero(activeQuestion + 1)}</span>
                <span className="total-question">/{addLeadingZero(questions.length)}</span>
              </div>
            </div>
            <div className='container-column'>
            <Timer time={10} setIsExpired={setIsTimeExpired} timerRes={timerReset}/>
            </div>
            <div className='container-column'></div>
          </div>
          <div>
            <span className="text-question-prompt">Show sign for: </span>
            <span className="text-question">{question}</span>
          </div>
          <div>
            <span className="text-answer-prompt-black">Result: </span>
            <span className="text-question">{sign}</span>
          </div>
          <div className="button-row">
              <button className="skip-quit-button" onClick={onClickNext} disabled={selectedAnswerIndex === null}>
                {'Quit'}
              </button>
            <button className="skip-quit-button" onClick={onClickNext} disabled={selectedAnswerIndex === null}>
              {activeQuestion === questions.length - 1 ? 'Finish' : 'Skip'}
            </button>
            {
              isCorrectSign ? (
                <button onClick={onClickNext} disabled={selectedAnswerIndex === null || !isCorrectSign}>
                <Check />
                {activeQuestion === questions.length - 1 ? 'Finish' : 'Next'}
              </button>
              ) : (
                <button onClick={onClickNext} disabled>
                {activeQuestion === questions.length - 1 ? 'Finish' : 'Next'}
              </button>
              )
            }
          </div>

          {/* <ul>
            {choices.map((answer: string, index: number) => (
              <li
                onClick={() => onAnswerSelected(answer, index)}
                key={answer}
                className={selectedAnswerIndex === index ? 'selected-answer' : ''}>
                {answer}
              </li>
            ))}
          </ul> */}
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
      <div className="quiz-container-column">
        <Webcam text={sign} setText={setSign}/> 
      </div>
    </div>
  )
}

export default Quiz

//export {}