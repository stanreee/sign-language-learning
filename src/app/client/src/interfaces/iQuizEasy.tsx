import iQuiz from "./iQuiz";
import iQuizASL from "./iQuizASL";

export default interface iQuizEasy extends iQuiz {
    topic: string,
    level: "Easy",
    description: "Easy ASL Letters",
    totalQuestions: 4,
    timePerQuestion: 30,
    questions: iQuizASL[] 
}
