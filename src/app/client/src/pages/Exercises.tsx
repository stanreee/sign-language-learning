import {useState } from 'react';
import Excercise from "../components/Excercise";

import '../styles/Exercise.css';

import iQuiz from '../interfaces/iQuiz'
import additionQuizRaw from "../data/additionQuiz.json"
import subtractionQuizRaw from "../data/subtractionQuiz.json"
import multiplicationQuizRaw from "../data/multiplicationQuiz.json"
import Quiz from './Quiz';


const Exercises = () => {

    const [selectedQuiz, setSelectedQuiz] = useState<string>();

    //DATA CALL
    const additionQuiz: iQuiz = additionQuizRaw;
    const subtractionQuiz: iQuiz = subtractionQuizRaw;
    const multiplicationQuiz: iQuiz = multiplicationQuizRaw;

    const quizArray: iQuiz[] = [additionQuiz, subtractionQuiz, multiplicationQuiz]

    return (
    <div className="Exercise-Page">
      <div className="Exercise-header">
        <h1>Exercises</h1>
      </div>
      <div className="Exercise-content">
        {selectedQuiz === undefined ? quizArray.map((quiz: iQuiz) => {
          return(
            <Excercise 
              title={quiz.topic}
              desc={quiz.description}
              difficulty={quiz.level}
              questionAmount={quiz.totalQuestions}
              quiz={quiz.questions}
              selected={selectedQuiz === quiz.topic}
              onClick={() => setSelectedQuiz(quiz.topic)}
            />
          )
          }) : (
            (selectedQuiz === "Addition" && <Quiz title={additionQuiz.topic} quizQuestions={additionQuiz.questions}/>)
            || (selectedQuiz === "Subtraction" && <Quiz title={subtractionQuiz.topic}  quizQuestions={subtractionQuiz.questions}/>)
            || (selectedQuiz === "Multiplication" && <Quiz title={multiplicationQuiz.topic}  quizQuestions={multiplicationQuiz.questions}/>)

          )
        }
      </div>
      {/* <div>
        {selectedQuiz === "Addition" && <Quiz quizQuestions={additionQuiz.questions}/>}
        {selectedQuiz === "Subtraction" && <Quiz quizQuestions={subtractionQuiz.questions}/>}
        {selectedQuiz === "Multiplication" && <Quiz quizQuestions={multiplicationQuiz.questions}/>}
      </div> */}
    </div>
    );
}

export default Exercises;
