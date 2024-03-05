import iQuiz from "../interfaces/iQuiz";
import iQuizEasy from "../interfaces/iQuizEasy";
import iQuizMedium from "../interfaces/iQuizMedium";
import iQuizHard from "../interfaces/iQuizHard";
import iQuizASL from "../interfaces/iQuizASL";
import letters from "../data/letters.json"

const selectRandomQuestions = (difficulty: string, numQuestions: number) => {

    let questions: iQuizASL[] = [];

    // Shuffle array

    let shuffled: string[] = [];

    if(difficulty === "Easy"){
        shuffled = letters.easyLetters.sort(() => 0.5 - Math.random()).slice(0, numQuestions);
    }
    else if (difficulty === "Medium"){
        shuffled = letters.mediumLetters.sort(() => 0.5 - Math.random()).slice(0, numQuestions);
    }
    else if (difficulty === "Hard"){
        shuffled = letters.allLetters.sort(() => 0.5 - Math.random()).slice(0, numQuestions);
    }
    else if (difficulty === "Vowel"){
        shuffled = letters.vowelLetters.sort(() => 0.5 - Math.random()).slice(0, numQuestions);
    }

    shuffled.forEach((letter: string) => {

        const question: iQuizASL = {
            question: letter,
            type: "demo",
            correctAnswer: letter
        }

        questions.push(
            question
        )
    })
  
    return questions;
  
  }


const getQuizzes = (
    numQuizzes: number,
    difficulty: string,
    questions: number,
    timePerQuestion: number
) => {

    const quizzes: iQuiz[] = [];

    for(let i = 1; i < numQuizzes+1; i++){
        const quiz: iQuiz = {
            topic: difficulty + " Quiz " + i,
            level: difficulty,
            description: "Random ASL Letters",
            totalQuestions: questions,
            timePerQuestion: timePerQuestion,
            questions: selectRandomQuestions(difficulty, questions)
        };

        quizzes.push(quiz);
    }

    return quizzes;

}

const getVowelQuiz = () => {
    // 2 easy
    const vowelQuiz: iQuiz[] = getQuizzes(
        1, "Vowel", 4, 30
    );

    return vowelQuiz
}


const getEasyQuizzes = () => {

    // 2 easy
    const easyQuizzes: iQuiz[] = getQuizzes(
        2, "Easy", 4, 30
    );

    return easyQuizzes
}

const getMediumQuizzes = () => {

    // 2 medium
    const mediumQuizzes: iQuiz[] = getQuizzes(
        2, "Medium", 5, 15
    );

    return mediumQuizzes
}

const getHardQuizzes = () => {

    // 2 hard
    const hardQuizzes: iQuiz[] = getQuizzes(
        1, "Hard", 6, 10
    );

    return hardQuizzes
}

const allQuizzes = {
    getEasyQuizzes,
    getMediumQuizzes,
    getHardQuizzes,
    getVowelQuiz
  }

export default allQuizzes;