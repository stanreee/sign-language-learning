import iQuizASL from "./iQuizASL";

interface iQuiz {
    topic: string,
    level: string,
    description: string,
    totalQuestions: number,
    questions: iQuizASL[] 
}

export default iQuiz;
