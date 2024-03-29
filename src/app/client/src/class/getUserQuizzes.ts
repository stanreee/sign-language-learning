import iQuiz from "../interfaces/iQuiz";
import iQuizEasy from "../interfaces/iQuizEasy";
import iQuizMedium from "../interfaces/iQuizMedium";
import iQuizHard from "../interfaces/iQuizHard";
import iQuizASL from "../interfaces/iQuizASL";
import ASLSigns from "../data/ASLSigns.json"

type letterArray = {
    letter: string,
    isStatic: string
}

const selectRandomQuestions = (difficulty: string, numQuestions: number) => {

    let questions: iQuizASL[] = [];

    // Shuffle array

    let shuffled: string[][] = [];

    if(difficulty === "Easy"){
        shuffled = ASLSigns.easyLetters.sort(() => 0.5 - Math.random()).slice(0, numQuestions);
    }
    else if (difficulty === "Medium"){
        shuffled = ASLSigns.mediumLetters.sort(() => 0.5 - Math.random()).slice(0, numQuestions);
    }
    else if (difficulty === "Hard"){
        shuffled = ASLSigns.hardLetters.sort(() => 0.5 - Math.random()).slice(0, numQuestions);
    }
    else if (difficulty === "Vowel"){
        shuffled = ASLSigns.vowelLetters.sort(() => 0.5 - Math.random()).slice(0, numQuestions);
    }
    else if (difficulty === "Easy Static"){
        shuffled = ASLSigns.easyStaticLetters.sort(() => 0.5 - Math.random()).slice(0, numQuestions);
    }
    else if (difficulty === "Medium Static"){
        shuffled = ASLSigns.mediumStaticLetters.sort(() => 0.5 - Math.random()).slice(0, numQuestions);
    }
    else if (difficulty === "Medium Dynamic"){
        shuffled = ASLSigns.mediumDynamicLetters.sort(() => 0.5 - Math.random()).slice(0, numQuestions);
    }
    else if (difficulty === "Hard Dynamic"){
        shuffled = ASLSigns.hardDynamicLetters.sort(() => 0.5 - Math.random()).slice(0, numQuestions);
    }

    shuffled.forEach((letter: string[]) => {
        const question: iQuizASL = {
            question: letter[0],
            type: "demo",
            isDynamic: letter[1],
            correctAnswer: letter[0]
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
    timePerQuestion: number,
    description: string,
    title: string
) => {

    const quizzes: iQuiz[] = [];

    for(let i = 1; i < numQuizzes+1; i++){
        const quiz: iQuiz = {
            topic: title,
            level: difficulty,
            description: description,
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
        1, "Vowel", 4, 30, "Vowels in ASL", "Vowels Quiz"
    );

    return vowelQuiz
}


const getEasyStaticQuizzes = () => {

    // 2 easy
    const easyQuizzes: iQuiz[] = getQuizzes(
        1, "Easy Static", 4, 30, "Random assortment of easy static signs", "Easy Static Sign Quiz"
    );

    return easyQuizzes
}

const getEasyQuizzes = () => {

    // 2 easy
    const easyQuizzes: iQuiz[] = getQuizzes(
        1, "Easy", 4, 30, "Random assortment of easy static and dynamic signs", "Easy Quiz"
    );

    return easyQuizzes
}

const getMediumQuizzes = () => {

    // 2 medium
    const mediumQuizzes: iQuiz[] = getQuizzes(
        1, "Medium", 5, 15, "Random assortment of medium difficulty static and dynamic signs", "Medium Quiz"
    );

    return mediumQuizzes
}

const getMediumStaticQuizzes = () => {

    // 2 medium
    const mediumQuizzes: iQuiz[] = getQuizzes(
        1, "Medium Static", 5, 15, "Random assortment of medium difficulty static signs", "Medium Static Sign Quiz"
    );

    return mediumQuizzes
}

const getMediumDynamicQuizzes = () => {

    // 2 medium
    const mediumQuizzes: iQuiz[] = getQuizzes(
        1, "Medium Dynamic", 5, 15, "Random assortment of medium difficulty dynamic signs", "Medium Dynamic Sign Quiz"
    );

    return mediumQuizzes
}

const getHardQuizzes = () => {

    // 2 hard
    const hardQuizzes: iQuiz[] = getQuizzes(
        1, "Hard", 6, 10, "Random assortment of easy, medium, and hard difficulty static and dynamic signs", "Hard Quiz"
    );

    return hardQuizzes
}

const getHardDynamicQuizzes = () => {

    // 2 hard
    const hardQuizzes: iQuiz[] = getQuizzes(
        1, "Hard Dynamic", 6, 10, "Random assortment of exclusively hard difficulty dynamic signs and words", "Hard Dynamic Sign Quiz"
    );

    return hardQuizzes
}

const allQuizzes = {
    getEasyQuizzes,
    getMediumQuizzes,
    getHardQuizzes,
    getVowelQuiz,
    getEasyStaticQuizzes,
    getMediumStaticQuizzes,
    getMediumDynamicQuizzes,
    getHardDynamicQuizzes
  }

export default allQuizzes;