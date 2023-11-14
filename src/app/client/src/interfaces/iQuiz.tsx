import iQuizQuestions from "./iQuizQuestions"

interface iQuiz {
    topic: string,
    level: string,
    description: string,
    totalQuestions: number,
    questions: iQuizQuestions[] 
}

export default iQuiz;
