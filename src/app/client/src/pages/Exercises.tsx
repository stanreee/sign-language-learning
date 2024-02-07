import {useState } from 'react';
import Excercise from "../components/Exercise";

import '../styles/Exercise.css';

import iQuiz from '../interfaces/iQuiz'
import Quiz from './Quiz';
import React from 'react';
import getUserQuizzes from '../class/getUserQuizzes';


const Exercises = () => {
    // store index of quiz selection
    const [selectedQuizIndex, setSelectedQuizIndex] = useState<number>();
    const [selectedQuizDifficulty, setSelectedQuizDifficulty] = useState<string>();

    //DATA CALL
    // GET QUIZ LIST
    const quizVowelArray: iQuiz[] = getUserQuizzes.getVowelQuiz();
    const quizEasyArray: iQuiz[] = getUserQuizzes.getEasyQuizzes();
    const quizEasyVowelArray = quizEasyArray.concat(quizVowelArray);

    const quizMediumArray: iQuiz[] = getUserQuizzes.getMediumQuizzes();
    const quizHardArray: iQuiz[] = getUserQuizzes.getHardQuizzes();

    // const quizArray: iQuiz[] = quizEasyVowelArray.concat(quizMediumArray, quizHardArray);


    return (
    <div className="Exercise-Page">
      <div className="Exercise-header">
        <h1>Exercises</h1>
      </div>
      <div className="Exercise-content">
        <div className='container-row'>
        {selectedQuizIndex === undefined ? (
          <div className='container-row'>
            <div className='container-column-3'>
              <h2>Easy</h2>
              {
              quizEasyVowelArray.map((quiz: iQuiz, index: number) => {
                return(
                  <Excercise 
                    title={quiz.topic}
                    desc={quiz.description}
                    difficulty={quiz.level}
                    timePerQuestion={quiz.timePerQuestion}
                    questionAmount={quiz.totalQuestions}
                    quiz={quiz.questions}
                    selected={selectedQuizIndex === index}
                    onClick={() => {
                      setSelectedQuizIndex(index)
                      setSelectedQuizDifficulty('Easy');
                    }}
                  />
                )
                })
              }
            </div>
            <div className='container-column-3'>
              <h2>Medium</h2>
              {
              quizMediumArray.map((quiz: iQuiz, index: number) => {
                return(
                  <Excercise 
                    title={quiz.topic}
                    desc={quiz.description}
                    difficulty={quiz.level}
                    timePerQuestion={quiz.timePerQuestion}
                    questionAmount={quiz.totalQuestions}
                    quiz={quiz.questions}
                    selected={selectedQuizIndex === index}
                    onClick={() => {
                      setSelectedQuizIndex(index)
                      setSelectedQuizDifficulty('Medium');
                    }}
                  />
                )
                })
              }
            </div>
            <div className='container-column-3'>
              <h2>Hard</h2>
              {
              quizHardArray.map((quiz: iQuiz, index: number) => {
                return(
                  <Excercise 
                    title={quiz.topic}
                    desc={quiz.description}
                    difficulty={quiz.level}
                    timePerQuestion={quiz.timePerQuestion}
                    questionAmount={quiz.totalQuestions}
                    quiz={quiz.questions}
                    selected={selectedQuizIndex === index}
                    onClick={() => {
                      setSelectedQuizIndex(index);
                      setSelectedQuizDifficulty('Hard');
                    }}
                  />
                )
                })
              }
            </div>
          </div>

        )
          
          : (
              (selectedQuizDifficulty === "Easy" && <Quiz title={quizEasyVowelArray[selectedQuizIndex].topic} timePerQuestion={quizEasyVowelArray[selectedQuizIndex].timePerQuestion} quizQuestions={quizEasyVowelArray[selectedQuizIndex].questions}/>)
              || (selectedQuizDifficulty === "Medium" && <Quiz title={quizMediumArray[selectedQuizIndex].topic} timePerQuestion={quizMediumArray[selectedQuizIndex].timePerQuestion} quizQuestions={quizMediumArray[selectedQuizIndex].questions}/>)
              || (selectedQuizDifficulty === "Hard" && <Quiz title={quizHardArray[selectedQuizIndex].topic} timePerQuestion={quizHardArray[selectedQuizIndex].timePerQuestion} quizQuestions={quizHardArray[selectedQuizIndex].questions}/>)
          )
        }
        </div>
      </div>
    </div>
    );
}

export default Exercises;
