import iQuizASL from "./iQuizASL";

interface iQuiz {
    topic: string,
    level: string,
    description: string,
    totalQuestions: number,
    timePerQuestion: number,
    questions: iQuizASL[] 
}


export default iQuiz;
