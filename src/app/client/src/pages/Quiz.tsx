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
import React from 'react';

type QuizProps = {
    title: string;
    timePerQuestion: number;
    quizQuestions: iQuizASL[];
    userEmail: string;
};

type resultAnswers = {
  index: number,
  question: string,
  guessedAnswer: string,
  correctAnswer: string,
  isCorrect: boolean
}

type userAnswers = {
  answer: string,
  isCorrect: boolean,
}

const saveResults = (results: resultAnswers[], userEmail: string) => {

  fetch('http://localhost:5001/post-quiz', {
  method: 'POST',
  headers: {
      'Content-Type': 'application/json',
  },
  body: JSON.stringify({userEmail, results}),
  })
  .then((r) => r.json())
}

const Quiz = ({
    title,
    timePerQuestion,
    quizQuestions,
    userEmail
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
  const [sign, setSign] = useState("")

  //temp
  // const [signIndex, setSignIndex] = useState(0);
  // const signHardCoded: string[] = ["A", "B", "F", "D", "G"];


  const [isCorrectSign, setIsCorrectSign] = useState(false);
  const [isTimeExpired, setIsTimeExpired] = useState(false);
  const [closeStream, setCloseStream] = useState(false);

  // timer countdown
  const [timerReset, resetTimer] = useState(false);

  // results array
  const [resultsQuestAns, setResultsQuestAns] = useState<resultAnswers[]>([]);

  const questions = quizQuestions

  //const { question, choices, correctAnswer } = questions[activeQuestion]
  const { question, correctAnswer } = questions[activeQuestion]

  const onClickNext = () => {
    setSelectedAnswerIndex('');
    setIsCorrectSign(false);
    setIsTimeExpired(false);
    resetTimer(true);
    // setResult((prev) =>
    //   selectedAnswer
    //     ? {
    //         ...prev,
    //         score: prev.score + 1,
    //         correctAnswers: prev.correctAnswers + 1,
    //       }
    //     : { ...prev, wrongAnswers: prev.wrongAnswers + 1 }
    // )
    if (activeQuestion !== questions.length - 1) {
      setActiveQuestion((prev) => prev + 1)
    } else {
      setActiveQuestion(0)
      setShowResult(true)
    }

    //temp
    //setSignIndex(signIndex + 1);
  }

  const onClickSkip = () => {
    setSelectedAnswerIndex('');
    setIsCorrectSign(false);
    setIsTimeExpired(false);
    resetTimer(true);
    // setResult((prev) => 
    //   ({ ...prev, wrongAnswers: prev.wrongAnswers + 1 })
    // )
    setResultsQuestAns(
      [
        ...resultsQuestAns,
        {index: activeQuestion, 
          question: questions[activeQuestion].question, 
          guessedAnswer: 'SKIPPED', 
          correctAnswer: questions[activeQuestion].correctAnswer, 
          isCorrect: false}
      ]
    )
    if (activeQuestion !== questions.length - 1) {
      setActiveQuestion((prev) => prev + 1)
    } else {
      setActiveQuestion(0)
      setShowResult(true)
      setCloseStream(true);
    }

    //temp
    //setSignIndex(signIndex + 1);
  }

  const onClickViewHistorical = () => {

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
      // array storing user answers to view what was wrong
      setResultsQuestAns(
        [
          ...resultsQuestAns,
          {index: activeQuestion, 
            question: questions[activeQuestion].question, 
            guessedAnswer: sign, 
            correctAnswer: questions[activeQuestion].correctAnswer, 
            isCorrect: true}
        ]
      )
      //selected indiviual answer in real time
      setSelectedAnswer(true);
      setIsCorrectSign(true);
      onClickNext();
      // temp
      //setSignIndex(signIndex + 1);
    }
  }, [sign]);

  //temp
  // useEffect(() => {
  //   setSign(signHardCoded[signIndex]);
  // }, [signIndex]);

  // hook from Timer module for expired time
  useEffect(() => {
    if(isTimeExpired) {
        setSelectedAnswer(false);
        onClickSkip();
      }
  }, [isTimeExpired])

   // Reset timer when next is clicked
   useEffect(() => {
      if (timerReset) {
        resetTimer(false);
      }
    }, [timerReset]);

    // make results array show results
    useEffect(() => {
      //calculate score
      let correctAnswers: number = 0
      const totalQuestions: number = resultsQuestAns.length;

      resultsQuestAns.forEach((answer: resultAnswers) => {
        console.log(answer)
        if(answer.isCorrect){
          correctAnswers += 1
        }
      })
      setResult({
        score: correctAnswers,
        correctAnswers: correctAnswers,
        wrongAnswers: totalQuestions,
      })

      saveResults(resultsQuestAns, userEmail);

    }, [showResult]);


  return (
    <div className="quiz-container">
      {!showResult ? (
        <div className='container-row'>
          <div className='quiz-container-column'>
            <h3>{title} Quiz</h3>
            <div className='container-row'>
              <div className='container-column'>
                <div>
                  <span className="active-question-no">{addLeadingZero(activeQuestion + 1)}</span>
                  <span className="total-question">/{addLeadingZero(questions.length)}</span>
                </div>
              </div>
              <div className='container-column'>
              <Timer time={timePerQuestion} setIsExpired={setIsTimeExpired} timerRes={timerReset}/>
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
                <Link reloadDocument to={'/Exercises'}>
                  <button className="skip-quit-button">
                    {'Quit'}
                  </button>
                </Link>
              <button className="skip-quit-button" onClick={onClickSkip} disabled={selectedAnswerIndex === null}>
                {'Skip'}
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
          </div>
          <div className="quiz-container-column">
            <Webcam text={sign} setText={setSign} run={true}/> 
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
          <div className="container-row">
            <div className="container-column-3">
              <div className="result">
                <h3>Result</h3>
                <p>
                  Score:<span> {result.score}</span>
                </p>
                <p>
                  Questions: <span>{questions.length}</span>
                </p>
                <Link reloadDocument to={'/Exercises'}>
                  <button>Return to exercises</button> 
                </Link>
                <div>

                </div>
                {/* <button onClick={onClickViewResults}>View Results</button>  */}
              </div>
            </div>
            <div className="container-column">
              <div className="container-row result">
                <div className="container-column">
                    <h3>Prompt</h3>
                      {resultsQuestAns.map((question: resultAnswers) => (
                            <ul>
                              <li
                                key={question.question}>
                                {question.question}
                              </li>
                            </ul>))
                      }
                  </div>
                  <div className="container-column">
                    <h3>Your Answer</h3>
                    {resultsQuestAns.map((question: resultAnswers) => (
                      <ul>
                        <li
                          key={question.guessedAnswer}
                          className={question.isCorrect ? 'correct-answer' : 'incorrect-answer'}
                          >
                          {question.guessedAnswer}
                        </li>
                        </ul>))
                      }
                  </div>
                  </div>
              </div>
            </div>
        )}
      </div>
  )
}

export default Quiz